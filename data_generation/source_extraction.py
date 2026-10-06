"""Source extraction.

Extracts source sentences from existing GEC datasets (BEA-2019, JFLEG)
and keeps those that satisfy:
  - source length ≥ MIN_TOKENS (whitespace tokens, default 10)
  - number of ERRANT edits between the source and the gold reference ≥ MIN_ERRANT_ERRORS (default 2)

The gold reference used for ERRANT is:
  - BEA-2019: the corrected sentence of annotator 0
  - JFLEG: human_correction_0 (the first of the four references)
"""
from __future__ import annotations

from typing import Any

import config
from utils import (
    dump_json, get_logger, load_json, normalize_ws,
    timed, whitespace_token_count,
)

log = get_logger("source_extraction")


def _gold_for(dataset: str, item: dict) -> str | None:
    """Select the gold reference for each dataset."""
    if dataset == "bea2019":
        # annotators dict -> first annotator
        ann = item.get("annotators") or {}
        if ann:
            first = sorted(ann.keys())[0]
            return ann[first].get("corrected") or item.get("corrected")
        return item.get("corrected")

    if dataset == "jfleg":
        c = item.get("candidates", {})
        return c.get("human_correction_0")

    return None


def _iter_items(dataset: str, path) -> list[dict]:
    items = load_json(path)
    if not isinstance(items, list):
        raise ValueError(f"{dataset}: expected JSON list, got {type(items).__name__}")
    return items


def _load_errant():
    """Loading ERRANT pulls in a spaCy model, so it is imported lazily."""
    import errant  # noqa: WPS433
    return errant.load("en")


def _count_errant_edits(annotator, source: str, gold: str) -> int:
    try:
        orig = annotator.parse(source)
        cor = annotator.parse(gold)
        edits = annotator.annotate(orig, cor)
    except Exception as e:  # treat spaCy/parse failures as 0 edits
        log.warning("ERRANT parse failed (%s): %.80s", e, source)
        return 0
    # 'noop' edits carry no correction, so they are excluded
    return sum(1 for e in edits if e.type and not e.type.startswith("noop"))


def extract_filtered_sources() -> list[dict]:
    """Iterate over all datasets and return the sources that satisfy the filters."""
    annotator = _load_errant()
    out: list[dict] = []
    sid = 0

    for dataset, path in config.SOURCES.items():
        target = config.SOURCE_TARGETS.get(dataset)
        items = _iter_items(dataset, path)
        log.info("dataset=%s loaded=%d target=%s", dataset, len(items), target)

        kept_for_dataset = 0
        for item in items:
            source = normalize_ws(item.get("source") or "")
            if not source:
                continue
            if whitespace_token_count(source) < config.MIN_TOKENS:
                continue

            gold = _gold_for(dataset, item)
            if not gold:
                continue
            gold = normalize_ws(gold)

            n_edits = _count_errant_edits(annotator, source, gold)
            if n_edits < config.MIN_ERRANT_ERRORS:
                continue

            out.append({
                "sid": sid,
                "dataset": dataset,
                "orig_id": item.get("id"),
                "source": source,
                "gold_reference": gold,
                "n_errant_edits": n_edits,
            })
            sid += 1
            kept_for_dataset += 1

            if target and kept_for_dataset >= target:
                break

        log.info("dataset=%s kept=%d", dataset, kept_for_dataset)

    return out


def main() -> None:
    with timed(log, "source extraction + ERRANT filtering"):
        filtered = extract_filtered_sources()
    dump_json(filtered, config.PATHS["filtered_sources"])
    log.info("total kept = %d → %s",
             len(filtered), config.PATHS["filtered_sources"])


if __name__ == "__main__":
    main()
