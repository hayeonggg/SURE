"""SEEDA meta-evaluation of SURE (Table 1 of the paper).

Clone SEEDA (Kobayashi et al., 2024; https://github.com/tmu-nlp/SEEDA) and pass its path with
--seeda-dir. The script runs system-level and sentence-level meta-evaluation for the four SURE
outputs (overall, grammaticality, faithfulness, fluency).

  - System-level:   mean sentence score vs. human TrueSkill ratings -> Pearson r / Spearman ρ
  - Sentence-level: agreement with the SEEDA pairwise judgments -> Accuracy / Kendall τ
  - SEEDA-E = edit-based judgments, SEEDA-S = sentence-based judgments
  - Base     = 12 GEC systems (without INPUT, REF-F, GPT-3.5)
    +Fluency = Base + REF-F, GPT-3.5

Usage (run from the repository root):
  git clone https://github.com/tmu-nlp/SEEDA
  python evaluation/seeda_eval.py --seeda-dir SEEDA
  python evaluation/seeda_eval.py --seeda-dir SEEDA --ckpt runs/my_run/checkpoints/rm_best_ep4_acc0.812
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sure.scorer import RewardScorer, detokenize, resolve_checkpoint  # noqa: E402


# ============================================================================
# Constants
# ============================================================================
# Systems in alphabetical order (SEEDA convention; same as the line order of scores/human/*.txt)
ALL_SYSTEMS = [
    "BART", "BERT-fuse", "GECToR-BERT", "GECToR-ens", "GPT-3.5",
    "INPUT", "LM-Critic", "PIE", "REF-F", "REF-M",
    "Riken-Tohoku", "T5", "TemplateGEC", "TransGEC", "UEDIN-MS",
]

# setting -> systems to exclude
SETTINGS: dict[str, list[str]] = {
    "Base":     ["GPT-3.5", "INPUT", "REF-F"],
    "+Fluency": ["INPUT"],
}

# subset → (judgments XML, system-level human rating)
SUBSETS: dict[str, tuple[str, str]] = {
    "SEEDA-E": ("judgments_edit.xml", "TS_edit"),
    "SEEDA-S": ("judgments_sent.xml", "TS_sent"),
}

# SURE outputs reported as metrics
FIELDS = ["overall", "grammaticality", "faithfulness", "fluency"]


# ============================================================================
# SEEDA loader
# ============================================================================
def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_seeda(seeda_dir: Path) -> dict:
    """391-sentence subset: 15 system outputs + system-level human ratings."""
    hypotheses = {
        name: _read_lines(seeda_dir / "outputs" / "subset" / f"{name}.txt")
        for name in ALL_SYSTEMS
    }
    sources = hypotheses["INPUT"]          # INPUT.txt is the uncorrected source
    for name, lines in hypotheses.items():
        assert len(lines) == len(sources), f"{name}: {len(lines)} != {len(sources)}"

    human_ratings: dict[str, dict[str, float]] = {}
    for _, rating in SUBSETS.values():
        values = [float(x) for x in _read_lines(seeda_dir / "scores" / "human" / f"{rating}.txt")]
        human_ratings[rating] = dict(zip(ALL_SYSTEMS, values))

    return {"sources": sources, "hypotheses": hypotheses, "human_ratings": human_ratings}


# ============================================================================
# SURE scoring (cached per system)
# ============================================================================
def score_all(seeda: dict, ckpt_path: Path, cache_dir: Path | None) -> dict[str, list[dict]]:
    """Score every (system, sentence). Returns {system: [{overall, grammaticality, ...}, ...]}."""
    results: dict[str, list[dict]] = {}
    scorer: RewardScorer | None = None      # the model is not loaded if every system is cached

    for system in ALL_SYSTEMS:
        cp = cache_dir / f"{system}.json" if cache_dir else None
        if cp is not None and cp.exists():
            results[system] = json.loads(cp.read_text(encoding="utf-8"))
            print(f"  {system:<14} n={len(results[system])} (cache)")
            continue

        if scorer is None:
            print(f"[scorer] loading SURE from {ckpt_path}")
            scorer = RewardScorer(ckpt_path)

        t0 = time.time()
        scores: list[dict] = []
        for src, hyp in zip(seeda["sources"], seeda["hypotheses"][system]):
            # SEEDA text is tokenized; detokenize it to match the training data
            out = scorer.score(detokenize(src), detokenize(hyp))
            scores.append({"overall": out["overall_reward"], **out["axis_scores"]})
        results[system] = scores
        if cp is not None:
            cp.write_text(json.dumps(scores), encoding="utf-8")
        print(f"  {system:<14} n={len(scores)} ({time.time()-t0:.1f}s)")

    return results


# ============================================================================
# Correlation
# ============================================================================
def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _rank(x: list[float]) -> list[float]:
    n = len(x)
    idx = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rank(x), _rank(y))


# ============================================================================
# SEEDA pairwise matrices (human judgments from the XML + SURE scores)
# ============================================================================
def parse_human_xml(xml_path: Path, drop_systems: set[str]) -> dict[str, list[list[int | None]]]:
    sys2id = {s: i for i, s in enumerate(ALL_SYSTEMS)}
    N = len(ALL_SYSTEMS)
    root = ET.parse(xml_path).getroot()

    h_mtx: dict[str, list[list[int | None]]] = {}
    for child in root.find("error-correction-ranking-result"):
        src_id = str(int(child.attrib["src-id"]) - 1)
        judg_set: dict[int, list[int]] = {}
        for trans in child:
            for sn in trans.attrib["system"].split():
                if sn in drop_systems or sn not in sys2id:
                    continue
                judg_set.setdefault(int(trans.attrib["rank"]), []).append(sys2id[sn])
        sub_mtx: list[list[int | None]] = [[None] * N for _ in range(N)]
        for r1, r2 in itertools.combinations(sorted(judg_set.keys()), 2):
            for t1 in judg_set[r1]:
                for t2 in judg_set[r2]:
                    if t1 < t2:
                        sub_mtx[t1][t2] = 1 if r1 < r2 else -1
                    else:
                        sub_mtx[t2][t1] = -1 if r1 < r2 else 1
        while src_id in h_mtx:              # the same source judged by several annotators
            src_id += "rep"
        h_mtx[src_id] = sub_mtx
    return {k: h_mtx[k] for k in sorted(h_mtx.keys())}


def build_metric_matrix(scored: dict[str, list[dict]], field: str,
                        src_id_order: list[int], drop_systems: set[str]) -> dict[str, list[list[int]]]:
    N = len(ALL_SYSTEMS)
    drop_id = {i for i, s in enumerate(ALL_SYSTEMS) if s in drop_systems}

    m_mtx: dict[str, list[list[int]]] = {}
    for i, conll_src in enumerate(src_id_order):
        sub_mtx = [[0] * N for _ in range(N)]
        for t1, t2 in itertools.combinations(range(N), 2):
            if t1 in drop_id or t2 in drop_id:
                continue
            s1 = scored[ALL_SYSTEMS[t1]][i][field]
            s2 = scored[ALL_SYSTEMS[t2]][i][field]
            if s1 == s2:
                sub_mtx[t1][t2] = 0
            elif s1 > s2:
                sub_mtx[t1][t2] = 1
            else:
                sub_mtx[t1][t2] = -1
        m_mtx[str(conll_src)] = sub_mtx
    return m_mtx


def calc_corr(h_mtx: dict[str, list[list[int | None]]],
              m_mtx: dict[str, list[list[int]]]) -> tuple[float, float, int]:
    """(accuracy, kendall tau, n_pairs), following corr_sentence.py of SEEDA."""
    cc = dc = 0
    for src_id, h in h_mtx.items():
        m = m_mtx.get(src_id.replace("rep", ""))
        if m is None:
            continue
        N = len(h)
        for i in range(N):
            for j in range(N):
                hij = h[i][j]
                if hij is None:
                    continue
                mij = m[i][j]
                if hij == mij and mij != 0:
                    cc += 1
                elif hij == -mij and mij != 0:
                    dc += 1
    if cc + dc == 0:
        return float("nan"), float("nan"), 0
    return cc / (cc + dc), (cc - dc) / (cc + dc), cc + dc


def derive_src_id_order(xml_path: Path) -> list[int]:
    """src-id in the XML (CoNLL-2014 sentence index) -> line order of the subset files."""
    root = ET.parse(xml_path).getroot()
    return sorted({
        int(c.attrib["src-id"]) - 1
        for c in root.find("error-correction-ranking-result")
    })


# ============================================================================
# Meta-evaluation
# ============================================================================
def meta_evaluate(seeda: dict, scored: dict[str, list[dict]], seeda_dir: Path) -> list[dict]:
    """System-level (r, ρ) and sentence-level (Acc, τ) for each field × subset × setting."""
    system_scores = {
        field: {s: sum(row[field] for row in scored[s]) / len(scored[s]) for s in ALL_SYSTEMS}
        for field in FIELDS
    }
    src_id_order = derive_src_id_order(seeda_dir / "data" / "judgments_edit.xml")
    if len(src_id_order) != len(seeda["sources"]):
        print(f"[WARN] XML src_id {len(src_id_order)} ≠ subset {len(seeda['sources'])}")

    results: list[dict] = []
    for subset, (xml_name, rating) in SUBSETS.items():
        for setting, drop in SETTINGS.items():
            drop_set = set(drop)
            systems = [s for s in ALL_SYSTEMS if s not in drop_set]
            human = [seeda["human_ratings"][rating][s] for s in systems]
            h_mtx = parse_human_xml(seeda_dir / "data" / xml_name, drop_set)
            for field in FIELDS:
                metric = [system_scores[field][s] for s in systems]
                m_mtx = build_metric_matrix(scored, field, src_id_order, drop_set)
                acc, tau, n_pairs = calc_corr(h_mtx, m_mtx)
                results.append({
                    "field": field, "subset": subset, "setting": setting,
                    "n_systems": len(systems), "n_pairs": n_pairs,
                    "sys_pearson": _pearson(metric, human),
                    "sys_spearman": _spearman(metric, human),
                    "sent_accuracy": acc,
                    "sent_kendall": tau,
                })
    return results


def format_table(results: list[dict]) -> str:
    index = {(r["field"], r["subset"], r["setting"]): r for r in results}
    cols = [(sub, st) for sub in SUBSETS for st in SETTINGS]
    lines = []
    for title, a, b, ha, hb in (
        ("System-level", "sys_pearson", "sys_spearman", "r", "rho"),
        ("Sentence-level", "sent_accuracy", "sent_kendall", "Acc.", "tau"),
    ):
        lines.append(title)
        lines.append(f"{'':<16}" + "".join(f"{sub + ' ' + st:^18}" for sub, st in cols))
        lines.append(f"{'':<16}" + "".join(f"{ha:>8} {hb:>8} " for _ in cols))
        for field in FIELDS:
            row = f"{field:<16}"
            for sub, st in cols:
                r = index[(field, sub, st)]
                row += f"{r[a]:>8.3f} {r[b]:>8.3f} "
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--seeda-dir", type=str, required=True,
                   help="path to the SEEDA repository (https://github.com/tmu-nlp/SEEDA)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint dir or Hugging Face repo id (default: released checkpoint)")
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "seeda"))
    p.add_argument("--no-cache", action="store_true",
                   help="ignore the per-system score cache in <output-dir>/scored/")
    args = p.parse_args()

    seeda_dir = Path(args.seeda_dir)
    ckpt_path = resolve_checkpoint(args.ckpt)
    print(f"[ckpt] {ckpt_path}")

    # the cache is kept per checkpoint
    out_dir = Path(args.output_dir)
    cache_dir = None
    if not args.no_cache:
        cache_dir = out_dir / "scored" / ckpt_path.name
        cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeda = load_seeda(seeda_dir)
    print(f"[data] {len(seeda['sources'])} sentences × {len(ALL_SYSTEMS)} systems")

    t0 = time.time()
    scored = score_all(seeda, ckpt_path, cache_dir)
    print(f"[scoring done in {time.time()-t0:.0f}s]\n")

    results = meta_evaluate(seeda, scored, seeda_dir)
    table = format_table(results)
    print(table)

    out_json = out_dir / f"{ckpt_path.name}.json"
    out_json.write_text(json.dumps({
        "dataset": "SEEDA",
        "checkpoint": str(ckpt_path),
        "n_sentences": len(seeda["sources"]),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{ckpt_path.name}.txt").write_text(table, encoding="utf-8")
    print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
