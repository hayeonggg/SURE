"""Score GEC outputs with SURE (reference-free).

Single pair:
  python score.py --source "He go to school yesterday." \
                  --candidate "He went to school yesterday."

Line-aligned files (one sentence per line):
  python score.py --sources src.txt --candidates hyp.txt --output scores.jsonl

`--ckpt` takes a local checkpoint dir or a Hugging Face repo id (default: download the released checkpoint).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sure.scorer import RewardScorer, detokenize

FIELDS = ["overall", "grammaticality", "faithfulness", "fluency"]


def _flatten(out: dict) -> dict:
    return {"overall": out["overall_reward"], **out["axis_scores"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--source", type=str, help="a single source sentence")
    p.add_argument("--candidate", type=str, help="a single candidate correction")
    p.add_argument("--sources", type=str, help="file with one source sentence per line")
    p.add_argument("--candidates", type=str, help="file with one candidate per line")
    p.add_argument("--output", type=str, default=None,
                   help="write per-sentence scores to this JSONL file")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint dir or Hugging Face repo id")
    p.add_argument("--detokenize", action="store_true",
                   help="detokenize Treebank-style tokenized input before scoring")
    args = p.parse_args()

    single = args.source is not None and args.candidate is not None
    batch = args.sources is not None and args.candidates is not None
    if single == batch:
        p.error("give either --source/--candidate or --sources/--candidates")

    if single:
        sources, candidates = [args.source], [args.candidate]
    else:
        sources = Path(args.sources).read_text(encoding="utf-8").splitlines()
        candidates = Path(args.candidates).read_text(encoding="utf-8").splitlines()
        if len(sources) != len(candidates):
            p.error(f"line count mismatch: {len(sources)} sources vs {len(candidates)} candidates")

    if args.detokenize:
        sources = [detokenize(s) for s in sources]
        candidates = [detokenize(c) for c in candidates]

    scorer = RewardScorer(args.ckpt)
    scores = [_flatten(scorer.score(s, c)) for s, c in zip(sources, candidates)]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for s, c, sc in zip(sources, candidates, scores):
                f.write(json.dumps({"source": s, "candidate": c, **sc}, ensure_ascii=False) + "\n")

    if single:
        print(json.dumps(scores[0], indent=2))
    else:
        # system-level score = mean over sentences
        mean = {k: sum(sc[k] for sc in scores) / max(len(scores), 1) for k in FIELDS}
        print(json.dumps({"n_sentences": len(scores), **mean}, indent=2))


if __name__ == "__main__":
    main()
