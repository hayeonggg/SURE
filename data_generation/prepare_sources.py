"""Build the source JSON files consumed by the pipeline (data/sources/).

BEA-2019  -> data/sources/bea2019_dev_gold.json
  W&I+LOCNESS dev set: source + each annotator's corrected sentence
  (derived by applying the gold M2 edits).

JFLEG     -> data/sources/jfleg.json
  source + 4 fluency-oriented human references per sentence (dev + test).

Inputs (download separately, see README):
  <bea_m2_dir>/ABCN.dev.gold.bea19.m2          (wi+locness/m2/)
  <jfleg_dir>/{dev,test}/{dev,test}.{src,ref0..ref3}

Usage:
  python prepare_sources.py --bea-m2-dir /path/to/wi+locness/m2 --jfleg-dir /path/to/jfleg
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import config


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


# ---------------------------------------------------------------------------
# BEA-2019 dev: source + multi-annotator gold edits + corrected text
# ---------------------------------------------------------------------------

_EDIT_RE = re.compile(
    r"^A\s+(-?\d+)\s+(-?\d+)\|\|\|([^|]*)\|\|\|([^|]*)\|\|\|([^|]*)\|\|\|([^|]*)\|\|\|(\d+)\s*$"
)


def parse_m2(path: str) -> List[Tuple[str, List[dict]]]:
    """Parse an M2 file into a list of (source, [edits]) tuples.

    Each edit dict: {start, end, type, correction, annotator}.
    "noop" (start==-1) edits are kept so callers can distinguish "no errors"
    from "no annotation".
    """
    entries: List[Tuple[str, List[dict]]] = []
    src: Optional[str] = None
    edits: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if src is not None:
                    entries.append((src, edits))
                src = line[2:]
                edits = []
            elif line.startswith("A "):
                m = _EDIT_RE.match(line)
                if not m:
                    continue
                start, end, etype, corr, _req, _nn, ann = m.groups()
                edits.append({
                    "start": int(start),
                    "end": int(end),
                    "type": etype,
                    "correction": corr,
                    "annotator": int(ann),
                })
            elif line == "" and src is not None:
                entries.append((src, edits))
                src, edits = None, []
        if src is not None:
            entries.append((src, edits))
    return entries


def apply_edits(source: str, edits: List[dict]) -> str:
    """Apply a single annotator's edits to a tokenised source sentence.

    Edits use token indices into source.split(' ').  noop edits (start==-1)
    are ignored.  Edits are applied right-to-left so earlier indices stay
    valid.
    """
    real_edits = sorted(
        (e for e in edits if e["start"] >= 0),
        key=lambda e: e["start"],
        reverse=True,
    )
    tokens = source.split(" ")
    for e in real_edits:
        s, t = e["start"], e["end"]
        repl = e["correction"]
        replacement = repl.split(" ") if repl else []
        tokens[s:t] = replacement
    return " ".join(tokens)


def build_bea2019_dev(m2_dir: str) -> List[dict]:
    abcn = os.path.join(m2_dir, "ABCN.dev.gold.bea19.m2")
    records = []
    for i, (src, edits) in enumerate(parse_m2(abcn)):
        # Group edits by annotator and compute a corrected string for each.
        by_ann: Dict[int, List[dict]] = defaultdict(list)
        for e in edits:
            by_ann[e["annotator"]].append(e)
        annotators = {
            str(ann): {"corrected": apply_edits(src, es)}
            for ann, es in sorted(by_ann.items())
        }
        records.append({
            "id": i,
            "source": src,
            "annotators": annotators,
            # Convenience: annotator 0's corrected string (most common path).
            "corrected": annotators.get("0", {}).get("corrected", src),
        })
    return records


# ---------------------------------------------------------------------------
# JFLEG
# ---------------------------------------------------------------------------

def build_jfleg(jfleg_dir: str) -> List[dict]:
    records = []
    rid = 0
    for split in ("dev", "test"):
        base = os.path.join(jfleg_dir, split)
        src = read_lines(os.path.join(base, f"{split}.src"))
        refs = [read_lines(os.path.join(base, f"{split}.ref{k}")) for k in range(4)]
        for r in refs:
            assert len(r) == len(src), f"ref length mismatch in {split}"
        for i, s in enumerate(src):
            cands = {"source_copy": s}
            for k in range(4):
                cands[f"human_correction_{k}"] = refs[k][i]
            records.append({"id": rid, "split": split, "source": s, "candidates": cands})
            rid += 1
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--bea-m2-dir", required=True,
                   help="directory containing ABCN.dev.gold.bea19.m2 (wi+locness/m2)")
    p.add_argument("--jfleg-dir", required=True,
                   help="JFLEG repository root (contains dev/ and test/)")
    args = p.parse_args()

    config.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("bea2019", build_bea2019_dev(args.bea_m2_dir)),
        ("jfleg",   build_jfleg(args.jfleg_dir)),
    ):
        out = config.SOURCES[name]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  -> {out}: {len(records)} sentences")


if __name__ == "__main__":
    main()
