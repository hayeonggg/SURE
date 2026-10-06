"""Build the candidate pool.

Collects the candidate corrections for each source.

Existing candidates (from the source datasets):
  - gold reference (annotator 0 / human_correction_0)
  - additional human corrections (if any)
  - the other JFLEG references (fluency edits)

LLM candidates (N_LLM_REWRITES per source, GPT-4o):
  - a minimal-edit correction ("Correct the following sentence: ...")
  - a rewrite-oriented correction that may restructure the sentence

A style-diverse pool exposes the preference data to valid corrections beyond minimal edits.
"""
from __future__ import annotations

import asyncio
from typing import Any

import config
from llm_clients import call_model
from utils import (
    append_jsonl, detokenize, dump_json, get_logger, is_source_copy,
    load_json, load_jsonl, normalize_ws, timed,
)

log = get_logger("candidate_pool")


# ---------------- Existing candidates ----------------
def _existing_candidates(item_lookup: dict[str, dict], sid_entry: dict) -> list[dict]:
    """Collect the existing candidates that come with the source in the original dataset."""
    key = f"{sid_entry['dataset']}::{sid_entry['orig_id']}"
    raw = item_lookup.get(key)
    cands: list[dict] = []
    if not raw:
        return cands

    # All existing candidates are detokenized so that the judges see natural text;
    # this avoids a spurious preference for GPT-4o outputs caused by formatting differences.
    def _add(text: str, tag: str, style: str) -> None:
        if not text:
            return
        cands.append({
            "text": detokenize(normalize_ws(text)),
            "source_tag": tag,
            "style_hint": style,
        })

    if sid_entry["dataset"] == "bea2019":
        # annotator 0 is the minimal-edit gold (also registered as gold_reference)
        ann = raw.get("annotators") or {}
        for k in sorted(ann.keys()):
            _add(ann[k].get("corrected"), f"bea_annotator_{k}", "minimal_edit")

    elif sid_entry["dataset"] == "jfleg":
        cdict = raw.get("candidates", {})
        # all four JFLEG human references are fluency edits
        for k, v in cdict.items():
            if k.startswith("human_correction_"):
                _add(v, f"jfleg_{k}", "fluency_edit")

    # Remove duplicate texts (keeping the first occurrence)
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in cands:
        if c["text"] in seen:
            continue
        seen.add(c["text"])
        deduped.append(c)

    # Remove source copies: a candidate that is effectively identical to the source
    # (nothing corrected) is dropped. The comparison uses the detokenized source.
    source_for_cmp = detokenize(sid_entry["source"])
    deduped = [c for c in deduped if not is_source_copy(c["text"], source_for_cmp)]
    return deduped


def _build_item_lookup() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for ds, path in config.SOURCES.items():
        for item in load_json(path):
            lookup[f"{ds}::{item.get('id')}"] = item
    return lookup


# ---------------- LLM rewrite candidates ----------------
REWRITE_SYSTEM = "You are a careful English writing assistant."

# Two prompts are used to obtain a spread of correction styles:
#   index 0: minimal-edit correction
#   index 1: rewrite-oriented correction that preserves the meaning but may restructure the sentence
# Different temperatures add further surface diversity.
REWRITE_VARIANTS: list[dict] = [
    {
        "prompt": (
            "Correct the following sentence:\n\n{source}\n\n"
            "Return ONLY the corrected sentence on one line, with no commentary."
        ),
        "temperature": 0.7,
    },
    {
        "prompt": (
            "Rewrite the following sentence so that it sounds natural and "
            "fluent in English. You may restructure clauses, change word "
            "order, swap synonyms, or split/merge clauses as long as the "
            "original meaning is preserved. Do not add information that "
            "is not in the source. Aim for a noticeably different surface "
            "form from a minimal-edit correction.\n\n"
            "Source: {source}\n\n"
            "Return ONLY the rewritten sentence on one line, with no commentary."
        ),
        "temperature": 1.1,
    },
]


async def _generate_rewrites_for(sid: int, source: str, n: int) -> list[dict]:
    """Call GPT-4o n times for the same source; call i uses the prompt REWRITE_VARIANTS[i]."""
    # if n exceeds the number of variants, cycle through them (default n=2 == number of variants)
    tasks = []
    for i in range(n):
        variant = REWRITE_VARIANTS[i % len(REWRITE_VARIANTS)]
        tasks.append(call_model(
            config.GENERATOR_MODEL,
            system=REWRITE_SYSTEM,
            user=variant["prompt"].format(source=source),
            temperature=variant["temperature"],
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.warning("sid=%d rewrite %d failed: %s", sid, i, r)
            continue
        text = normalize_ws(r)
        if text:
            out.append({
                "text": text,
                "source_tag": f"gpt4o_rewrite_{i}",
                "style_hint": "rewrite",
            })
    return out


# ---------------- Orchestration ----------------
def _load_already_done(path) -> set[int]:
    done = set()
    for row in load_jsonl(path):
        done.add(row["sid"])
    return done


async def build_pool() -> None:
    """Results are appended to a jsonl file, so an interrupted run can be resumed."""
    sources = load_json(config.PATHS["filtered_sources"])
    lookup = _build_item_lookup()

    cache_path = config.OUTPUT_DIR / "candidate_pool.jsonl"
    already = _load_already_done(cache_path)
    log.info("sources=%d, already done=%d", len(sources), len(already))

    # Use a separate (lower) concurrency for GPT-4o to stay within its TPM limit
    sem = asyncio.Semaphore(config.REWRITE_SOURCE_CONCURRENCY)
    log.info("source concurrency = %d (rewrite calls in-flight ≈ %d)",
             config.REWRITE_SOURCE_CONCURRENCY,
             config.REWRITE_SOURCE_CONCURRENCY * config.N_LLM_REWRITES)
    stats = {"kept": 0, "dropped_source_copy_only": 0}

    async def _work(entry: dict) -> None:
        if entry["sid"] in already:
            return
        # The source shown to the judges is kept separately in detokenized form;
        # the tokenized `source` is kept for the ERRANT alignment.
        source_detok = detokenize(entry["source"])

        existing = _existing_candidates(lookup, entry)
        async with sem:
            rewrites = await _generate_rewrites_for(
                entry["sid"], source_detok, config.N_LLM_REWRITES,
            )
        # GPT-4o outputs are also detokenized and kept only if they are not source copies.
        rewrites_kept: list[dict] = []
        for c in rewrites:
            c["text"] = detokenize(c["text"])
            if is_source_copy(c["text"], source_detok):
                log.info("sid=%d rewrite '%s' dropped (source-copy)",
                         entry["sid"], c["source_tag"])
                continue
            rewrites_kept.append(c)

        all_cands = existing + rewrites_kept

        # drop the source if too few candidates remain after removing source copies
        if len(all_cands) < config.MIN_CANDIDATES_PER_SOURCE:
            stats["dropped_source_copy_only"] += 1
            log.info("sid=%d dropped: only %d non-source-copy candidates "
                     "(< MIN_CANDIDATES_PER_SOURCE=%d)",
                     entry["sid"], len(all_cands), config.MIN_CANDIDATES_PER_SOURCE)
            return
        stats["kept"] += 1

        # assign cid: sort by source_tag for a stable order
        all_cands_sorted = sorted(all_cands, key=lambda c: c["source_tag"])
        for i, c in enumerate(all_cands_sorted):
            c["cid"] = i

        record = {
            "sid": entry["sid"],
            "dataset": entry["dataset"],
            "source": entry["source"],                # tokenized (for ERRANT)
            "source_detok": source_detok,             # for the judge prompt
            "gold_reference": entry["gold_reference"],
            "n_errant_edits": entry["n_errant_edits"],
            "candidates": all_cands_sorted,
            "n_candidates": len(all_cands_sorted),
        }
        append_jsonl(record, cache_path)

    await asyncio.gather(*[_work(e) for e in sources])

    log.info("source filtering summary: kept=%d, dropped(too few non-copy candidates)=%d",
             stats["kept"], stats["dropped_source_copy_only"])

    # jsonl -> sorted final json
    rows = sorted(load_jsonl(cache_path), key=lambda r: r["sid"])
    dump_json(rows, config.PATHS["candidate_pool"])
    log.info("candidate pool built: %d sources → %s",
             len(rows), config.PATHS["candidate_pool"])


def main() -> None:
    with timed(log, "candidate pool construction"):
        asyncio.run(build_pool())


if __name__ == "__main__":
    main()
