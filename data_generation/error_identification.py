"""Extract the error spans of each source with ERRANT.

ERRANT is applied to the (source, gold_reference) pair.
  - The gold reference is a minimal edit, so the surface alignment of ERRANT is reliable.
  - Each span gets a label e1, e2, ... that is quoted directly in the judge prompt.

Each edit has the following fields:
  - eid: 1-based label such as "e1"
  - type: ERRANT error type (e.g. "R:VERB:SVA", "M:DET", ...)
  - o_start/o_end: token offsets in the source
  - source_span: substring of the source ("" for a missing-token edit)
  - correction: corrected token(s) proposed by ERRANT
"""
from __future__ import annotations

from typing import Any

import config
from utils import detokenize, dump_json, get_logger, load_json, timed

log = get_logger("error_identification")


def _load_errant():
    import errant  # noqa: WPS433
    return errant.load("en")


def _extract_spans(annotator, source: str, gold: str) -> list[dict[str, Any]]:
    """Extract the edit list from the ERRANT alignment of source vs. gold."""
    try:
        orig = annotator.parse(source)
        cor = annotator.parse(gold)
        edits = annotator.annotate(orig, cor)
    except Exception as e:
        log.warning("ERRANT failed: %s — source=%.80s", e, source)
        return []

    spans: list[dict[str, Any]] = []
    eid = 0
    for ed in edits:
        if not ed.type or ed.type.startswith("noop"):
            continue
        eid += 1
        # ERRANT edit object: o_start/o_end (source token indices), c_str (correction)
        o_start = getattr(ed, "o_start", None)
        o_end = getattr(ed, "o_end", None)
        source_substr = ""
        try:
            source_tokens = [t.text for t in orig]
            if o_start is not None and o_end is not None and o_end > o_start:
                source_substr = " ".join(source_tokens[o_start:o_end])
        except Exception:
            source_substr = ""

        correction = getattr(ed, "c_str", "") or ""
        spans.append({
            "eid": f"e{eid}",
            "type": ed.type,
            "o_start": o_start,
            "o_end": o_end,
            # source_span/correction are quoted in the judge prompt, so detokenize them
            "source_span": detokenize(source_substr),
            "correction": detokenize(correction),
        })
    return spans


def build_error_spans() -> list[dict]:
    pool = load_json(config.PATHS["candidate_pool"])
    annotator = _load_errant()
    out: list[dict] = []
    for row in pool:
        spans = _extract_spans(annotator, row["source"], row["gold_reference"])
        out.append({
            "sid": row["sid"],
            "dataset": row["dataset"],
            "source": row["source"],
            "gold_reference": row["gold_reference"],
            "error_spans": spans,
            "n_error_spans": len(spans),
        })
    return out


def main() -> None:
    with timed(log, "ERRANT error span identification"):
        rows = build_error_spans()
    dump_json(rows, config.PATHS["error_spans"])
    n_zero = sum(1 for r in rows if r["n_error_spans"] == 0)
    log.info("error spans extracted for %d sources (%d had 0 spans) → %s",
             len(rows), n_zero, config.PATHS["error_spans"])


if __name__ == "__main__":
    main()
