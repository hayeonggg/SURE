"""Serialize the accepted pairs as training instances.

Spec:
  (source, error_spans, y⁺, y⁻,
   span_labels_y⁺, span_labels_y⁻,
   axis_preferences, overall_preference)

The accepted pairs are converted into a schema that is easy to use for training,
and summary statistics are saved alongside.
"""
from __future__ import annotations

from collections import Counter

import config
from utils import dump_json, get_logger, load_json, timed

log = get_logger("finalize")


def to_training_instance(row: dict) -> dict:
    return {
        "sid": row["sid"],
        "dataset": row["dataset"],
        "source": row["source"],
        "source_detok": row.get("source_detok") or row["source"],
        "error_spans": row["error_spans"],
        "y_plus":  {
            "cid": row["y_plus"]["cid"],
            "text": row["y_plus"]["text"],
            "source_tag": row["y_plus"]["source_tag"],
            "style_hint": row["y_plus"].get("style_hint"),
        },
        "y_minus": {
            "cid": row["y_minus"]["cid"],
            "text": row["y_minus"]["text"],
            "source_tag": row["y_minus"]["source_tag"],
            "style_hint": row["y_minus"].get("style_hint"),
        },
        "span_labels_y_plus":  row["span_labels_y_plus"],
        "span_labels_y_minus": row["span_labels_y_minus"],
        "axis_preferences":    row["axis_preferences"],
        "overall_preference":  row["overall_preference"],
        "meta": {
            "axis_votes": row["axis_votes"],
            "overall_votes": row["overall_votes"],
            "judge_models": row["judge_models"],
        },
    }


def summarize(rows: list[dict]) -> dict:
    style_pairs = Counter()
    axis_pref = {
        "grammaticality": Counter(),
        "faithfulness": Counter(),
        "fluency": Counter(),
    }
    dataset_counts = Counter()
    n_resolved_plus = n_resolved_minus = 0
    n_total_spans = 0

    for r in rows:
        dataset_counts[r["dataset"]] += 1
        sp = r["y_plus"].get("style_hint", "?")
        sm = r["y_minus"].get("style_hint", "?")
        style_pairs[(sp, sm)] += 1
        for k in axis_pref:
            axis_pref[k][r["axis_preferences"][k]] += 1
        for v in r["span_labels_y_plus"].values():
            n_total_spans += 1
            if v == "resolved":
                n_resolved_plus += 1
        for v in r["span_labels_y_minus"].values():
            if v == "resolved":
                n_resolved_minus += 1

    return {
        "n_pairs": len(rows),
        "by_dataset": dict(dataset_counts),
        "style_pair_counts": {f"{a}->{b}": c for (a, b), c in style_pairs.items()},
        "axis_preferences": {k: dict(v) for k, v in axis_pref.items()},
        "span_resolution": {
            "n_spans_total":        n_total_spans,
            "y_plus_resolved_rate":  (n_resolved_plus / n_total_spans) if n_total_spans else 0.0,
            "y_minus_resolved_rate": (n_resolved_minus / n_total_spans) if n_total_spans else 0.0,
        },
    }


def main() -> None:
    with timed(log, "finalize dataset"):
        rows = [to_training_instance(r) for r in load_json(config.PATHS["pairs"])]
        stats = summarize(rows)
    dump_json(rows, config.PATHS["final_dataset"])
    dump_json(stats, config.OUTPUT_DIR / "preference_pairs_stats.json")
    log.info("final dataset: %d pairs → %s",
             len(rows), config.PATHS["final_dataset"])
    log.info("summary: %s", stats)


if __name__ == "__main__":
    main()
