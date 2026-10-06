"""Pair sampling and unanimous consensus of the three judges.

Algorithm:
  for each source with N candidates:
    1) Randomly sample PAIR_OVERSAMPLE_FACTOR * MAX_PAIRS_PER_SOURCE (unordered) candidate
       pairs out of N choose 2.
    2) Query all three judges for each pair.
       The candidate with the smaller cid is always A (for consistent cache keys).
    3) Accept the pair only if the three overall verdicts are identical,
       i.e. (A,A,A) or (B,B,B).
    4) Decide y⁺ / y⁻:
              A,A,A → y⁺ = A, y⁻ = B
              B,B,B → y⁺ = B, y⁻ = A
    5) Move to the next source once MAX_PAIRS_PER_SOURCE pairs are accepted.

Criteria preferences and span-resolution labels of the accepted pairs are
aggregated over the three judges by majority vote.
"""
from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections import Counter
from typing import Any

import config
from llm_judge import judge_all, load_judge_cache
from utils import dump_json, get_logger, load_json, timed

log = get_logger("pair_consensus")


def _majority(values: list[str]) -> str:
    """Majority vote over criterion verdicts. Returns 'tie' if no label has two votes."""
    counts = Counter(values)
    top, n = counts.most_common(1)[0]
    return top if n >= 2 else "tie"


# Span label space = {resolved, partial, missed}.
# When the three judges all disagree, the median label is used.
# best → worst: resolved > partial > missed
_SPAN_ORDER = ["resolved", "partial", "missed"]


def _span_majority(values: list[str]) -> str:
    """Majority vote over span labels. Label space = {resolved, partial, missed}.
    If the three judges all disagree, the median label is used (= 'partial')."""
    counts = Counter(values)
    top, n = counts.most_common(1)[0]
    if n >= 2:
        return top
    # all three judges gave different labels: use the median label
    sorted_vals = sorted(values, key=lambda x: _SPAN_ORDER.index(x) if x in _SPAN_ORDER else 99)
    return sorted_vals[1] if len(sorted_vals) == 3 else "partial"


def _aggregate_judges(
    judge_outputs: dict[str, dict | None],
    eids: list[str],
) -> dict[str, Any] | None:
    """Aggregate consensus / criteria majority / span majority from the three judge outputs.

    Returns None if any judge failed.
    """
    # every judge must have responded
    if any(v is None for v in judge_outputs.values()):
        return None

    overalls = [v["overall"] for v in judge_outputs.values()]
    axes_keys = ("grammaticality", "faithfulness", "fluency")
    axis_verdicts = {
        k: [v["axes"][k] for v in judge_outputs.values()]
        for k in axes_keys
    }

    unanimous = len(set(overalls)) == 1 and overalls[0] in ("A", "B")

    # Span majority: majority vote of the three judges for each eid
    # (falls back to the median label when they all disagree)
    span_majority: dict[str, dict[str, str]] = {"A": {}, "B": {}}
    for side in ("A", "B"):
        for eid in eids:
            labels = [
                v["span_resolution"].get(side, {}).get(eid, "missed")
                for v in judge_outputs.values()
            ]
            span_majority[side][eid] = _span_majority(labels)

    # Criteria: plain majority vote
    axis_majority = {k: _majority(v) for k, v in axis_verdicts.items()}

    return {
        "overall_votes": overalls,
        "unanimous": unanimous,
        "axis_votes": axis_verdicts,
        "axis_majority": axis_majority,
        "span_majority": span_majority,
        "per_judge": judge_outputs,
    }


def _sample_pairs(n_candidates: int, n_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample up to n_pairs unique unordered (cid_i, cid_j) pairs."""
    all_pairs = list(itertools.combinations(range(n_candidates), 2))
    rng.shuffle(all_pairs)
    return all_pairs[:n_pairs]


async def collect_consensus_pairs() -> list[dict]:
    """Sample candidate pairs and keep those with a unanimous judgment of the three judges.

    Concurrency:
      - one task per source, run with asyncio.gather
      - pairs within a source are judged sequentially (to count MAX_PAIRS_PER_SOURCE)
      - the number of in-flight pairs is limited by the PAIR_CONCURRENCY semaphore
      - per-dataset quotas and TARGET_PAIRS are checked on shared state under a lock

    The pool is shuffled with RANDOM_SEED so that the datasets are processed in mixed order.
    """
    pool = load_json(config.PATHS["candidate_pool"])
    spans_by_sid = {
        r["sid"]: r["error_spans"]
        for r in load_json(config.PATHS["error_spans"])
    }
    cache = load_judge_cache()
    rng = random.Random(config.RANDOM_SEED)

    # shuffle the pool so that the datasets are processed in mixed order
    pool = list(pool)
    rng.shuffle(pool)

    accepted: list[dict] = []
    dataset_counts: Counter[str] = Counter()
    n_sources_tried = 0
    n_pairs_tried = 0
    n_pairs_unanimous = 0

    pair_sem = asyncio.Semaphore(config.PAIR_CONCURRENCY)
    state_lock = asyncio.Lock()       # protects accepted/counters

    log.info("pool=%d sources (shuffled) | target=%d pairs | quotas=%s | "
             "pair_concurrency=%d | judges=%s | cache=%d entries",
             len(pool), config.TARGET_PAIRS, dict(config.DATASET_PAIR_QUOTAS),
             config.PAIR_CONCURRENCY,
             [j.model_id for j in config.JUDGES], len(cache))

    def _global_done() -> bool:
        return len(accepted) >= config.TARGET_PAIRS

    def _dataset_done(ds: str) -> bool:
        return dataset_counts[ds] >= config.DATASET_PAIR_QUOTAS.get(ds, 0)

    async def _process_source(entry: dict) -> None:
        nonlocal n_sources_tried, n_pairs_tried, n_pairs_unanimous

        if _global_done() or _dataset_done(entry["dataset"]):
            return
        cands = entry["candidates"]
        if len(cands) < 2:
            return

        n_sources_tried += 1
        error_spans = spans_by_sid.get(entry["sid"], [])
        eids = [sp["eid"] for sp in error_spans]
        judge_source = entry.get("source_detok") or entry["source"]

        n_try = min(
            len(cands) * (len(cands) - 1) // 2,
            config.PAIR_OVERSAMPLE_FACTOR * config.MAX_PAIRS_PER_SOURCE,
        )
        # pair sampling for a source uses its own RNG instance (safe across tasks)
        local_rng = random.Random(config.RANDOM_SEED + entry["sid"])
        pair_indices = _sample_pairs(len(cands), n_try, local_rng)
        n_accepted_for_source = 0

        for i, j in pair_indices:
            if n_accepted_for_source >= config.MAX_PAIRS_PER_SOURCE:
                break
            if _global_done() or _dataset_done(entry["dataset"]):
                break
            cand_a, cand_b = cands[i], cands[j]
            if cand_a["cid"] > cand_b["cid"]:
                cand_a, cand_b = cand_b, cand_a

            # judge calls are limited by pair_sem
            async with pair_sem:
                # this counter is only for logging, so no lock is needed
                n_pairs_tried += 1
                pair_idx = n_pairs_tried
                log.info("▶ pair#%d sid=%d (%s) cids=(%d,%d) — calling %d judges...",
                         pair_idx, entry["sid"], entry["dataset"],
                         cand_a["cid"], cand_b["cid"], len(config.JUDGES))
                t_start = time.time()
                outputs = await judge_all(
                    entry["sid"], judge_source, error_spans,
                    cand_a, cand_b, cache=cache,
                )

            elapsed = time.time() - t_start
            agg = _aggregate_judges(outputs, eids)
            vote_repr = ",".join(
                (outputs[j.model_id]["overall"] if outputs.get(j.model_id) else "FAIL")
                for j in config.JUDGES
            )

            if agg is None or not agg["unanimous"]:
                log.info("  ✗ rejected pair#%d (%s) votes=[%s] (%.1fs) | accepted=%d/%d",
                         pair_idx, entry["dataset"], vote_repr, elapsed,
                         len(accepted), config.TARGET_PAIRS)
                continue

            winner = agg["overall_votes"][0]
            y_plus = cand_a if winner == "A" else cand_b
            y_minus = cand_b if winner == "A" else cand_a

            new_row = {
                "sid": entry["sid"],
                "dataset": entry["dataset"],
                "source": entry["source"],
                "source_detok": judge_source,
                "gold_reference": entry["gold_reference"],
                "error_spans": error_spans,
                "y_plus": y_plus,
                "y_minus": y_minus,
                "span_labels_y_plus":  agg["span_majority"][winner],
                "span_labels_y_minus": agg["span_majority"]["B" if winner == "A" else "A"],
                "axis_preferences": agg["axis_majority"],
                "axis_votes": agg["axis_votes"],
                "overall_preference": "y_plus",
                "overall_votes": agg["overall_votes"],
                "judge_models": [j.model_id for j in config.JUDGES],
            }
            async with state_lock:
                # another task may have filled the quota in the meantime, so check again
                if _global_done() or _dataset_done(entry["dataset"]):
                    log.info("  ⊘ discarded pair#%d (%s) — quota raced", pair_idx, entry["dataset"])
                    break
                accepted.append(new_row)
                dataset_counts[entry["dataset"]] += 1
                n_pairs_unanimous += 1
                n_accepted_for_source += 1
                accepted_snapshot = len(accepted)
                ds_snapshot = dict(dataset_counts)
            log.info("  ✓ accepted pair#%d (%s) winner=%s votes=[%s] (%.1fs) | accepted=%d/%d %s",
                     pair_idx, entry["dataset"], winner, vote_repr, elapsed,
                     accepted_snapshot, config.TARGET_PAIRS, ds_snapshot)

        # progress statistics every 100 sources
        if n_sources_tried % 100 == 0:
            log.info("[stats] sources=%d pairs_tried=%d unanimous=%d accepted=%d (%.0f%%) by_ds=%s",
                     n_sources_tried, n_pairs_tried, n_pairs_unanimous,
                     len(accepted),
                     100.0 * n_pairs_unanimous / max(n_pairs_tried, 1),
                     dict(dataset_counts))

    await asyncio.gather(*[_process_source(e) for e in pool])

    log.info("DONE: sources=%d, pairs tried=%d, unanimous=%d, accepted=%d",
             n_sources_tried, n_pairs_tried, n_pairs_unanimous, len(accepted))
    return accepted


def main() -> None:
    with timed(log, "pair sampling + 3-judge consensus"):
        pairs = asyncio.run(collect_consensus_pairs())
    dump_json(pairs, config.PATHS["pairs"])
    log.info("accepted preference pairs: %d → %s",
             len(pairs), config.PATHS["pairs"])


if __name__ == "__main__":
    main()
