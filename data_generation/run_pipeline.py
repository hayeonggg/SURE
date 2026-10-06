"""Preference data generation pipeline.

Usage (run inside data_generation/):
  python run_pipeline.py                      # run all stages in order
  python run_pipeline.py --from error_spans   # resume from a stage
  python run_pipeline.py --only finalize      # run a single stage

Stages: sources -> candidates -> error_spans -> pairs -> finalize.
The LLM judges (llm_judge.py) are called inside the `pairs` stage.
"""
from __future__ import annotations

import argparse

import candidate_pool
import error_identification
import finalize
import pair_consensus
import source_extraction
from utils import get_logger, timed

log = get_logger("pipeline")


STAGES = {
    "sources":     source_extraction.main,      # source extraction + ERRANT filtering
    "candidates":  candidate_pool.main,         # human references + GPT-4o corrections
    "error_spans": error_identification.main,   # ERRANT error spans
    "pairs":       pair_consensus.main,         # pair sampling + LLM judges + consensus
    "finalize":    finalize.main,               # training instances + statistics
}


def main() -> None:
    names = list(STAGES)
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start_at", choices=names, default=names[0],
                        help="stage to start from")
    parser.add_argument("--only", dest="only", choices=names, default=None,
                        help="run only this stage")
    args = parser.parse_args()

    selected = [args.only] if args.only else names[names.index(args.start_at):]
    for name in selected:
        with timed(log, name):
            STAGES[name]()


if __name__ == "__main__":
    main()
