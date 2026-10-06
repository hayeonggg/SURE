"""LLM judge call (span resolution + three criteria + overall).

A single call judges all of the following:
  1) for each candidate, whether each span e1/e2/... is resolved/missed/partial
  2) Grammaticality: A vs B (A | B | tie)
  3) Faithfulness:   A vs B
  4) Fluency:        A vs B
  5) Overall:        A vs B

Principles:
  - Error identification (done beforehand with ERRANT) is separated from resolution judgment (by the LLM).
  - The three criteria are sentence-level comparisons.
  - The amount of editing is not a criterion: the judges decide which candidate resolves the
    source errors, preserves the meaning, and reads as more natural English.

Responses are parsed as strict JSON. Each judge response is appended to judge_cache.jsonl,
so a re-run skips the calls that are already cached.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Iterable

import config
from llm_clients import call_model, parse_json_loose
from utils import append_jsonl, get_logger, load_jsonl

log = get_logger("llm_judge")


JUDGE_SYSTEM = (
    "You are an expert linguistic annotator for grammatical error correction (GEC). "
    "You compare two candidate corrections of the same source sentence on multiple "
    "axes. You always reply with a single JSON object that follows the requested "
    "schema exactly. Do not include commentary outside the JSON."
)


JUDGE_USER_TEMPLATE = """Source sentence:
{source}

Known errors in the source (extracted by ERRANT against a minimal-edit reference;
treat these as the ground-truth list of errors that should be addressed):
{error_block}

Candidate A:
{cand_a}

Candidate B:
{cand_b}

Your task:
For each candidate, judge whether it RESOLVED each listed error.
  - "resolved": the error is corrected and the result is grammatical.
  - "missed":   the error is still present (or re-introduced).
  - "partial":  attempted but incomplete / introduces a new issue.

Then compare A vs B on three sentence-level axes:
  - Grammaticality: which candidate has fewer grammar errors w.r.t. the source.
  - Faithfulness:   which candidate better preserves the source's meaning and intent.
  - Fluency:        which candidate reads as more natural, idiomatic English.

Finally, give an OVERALL preference for the better correction.

IMPORTANT:
  - The AMOUNT of editing is NOT a criterion. Minimal edits and rewrites are equally
    valid; judge only on the criteria above.
  - For each axis and overall, you MUST choose exactly one of: "A" or "B".
    DO NOT use "tie". If the two candidates seem close on a criterion,
    still pick whichever is even marginally better.

Reply with a single JSON object using this exact schema:
{{
  "span_resolution": {{
    "A": {{"e1": "resolved|missed|partial", ...}},
    "B": {{"e1": "resolved|missed|partial", ...}}
  }},
  "axes": {{
    "grammaticality": "A|B",
    "faithfulness":   "A|B",
    "fluency":        "A|B"
  }},
  "overall": "A|B",
  "rationale": "<one short sentence>"
}}
"""


def _format_error_block(error_spans: list[dict]) -> str:
    if not error_spans:
        return "  (no errors detected by ERRANT)"
    lines = []
    for sp in error_spans:
        span_repr = sp.get("source_span") or "(missing token / insertion site)"
        lines.append(f"  {sp['eid']}: \"{span_repr}\" — {sp['type']}")
    return "\n".join(lines)


def build_judge_prompt(source: str, error_spans: list[dict],
                       cand_a: str, cand_b: str) -> str:
    return JUDGE_USER_TEMPLATE.format(
        source=source,
        error_block=_format_error_block(error_spans),
        cand_a=cand_a,
        cand_b=cand_b,
    )


# ---------------- Cache key ----------------
def judge_cache_key(judge_model_id: str, sid: int,
                    cid_a: int, cid_b: int) -> str:
    """Pairs are ordered: (A, B) and (B, A) calls have different cache keys."""
    raw = f"{judge_model_id}|{sid}|{cid_a}|{cid_b}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# The prompt forbids ties, so only 'A' or 'B' is valid.
# If a model still returns 'tie', the response is invalid and call_judge returns None.
_VALID_VERDICT = {"A", "B"}
_VALID_RES = {"resolved", "missed", "partial"}


def _validate(parsed: dict | None, eids: list[str]) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    try:
        sr = parsed["span_resolution"]
        axes = parsed["axes"]
        overall = parsed["overall"]
        # axes
        for k in ("grammaticality", "faithfulness", "fluency"):
            if axes[k] not in _VALID_VERDICT:
                return None
        if overall not in _VALID_VERDICT:
            return None
        # span
        for side in ("A", "B"):
            side_map = sr.get(side, {})
            for eid in eids:
                v = side_map.get(eid)
                if v not in _VALID_RES:
                    # missing eid → fill with "missed" (conservative)
                    side_map[eid] = "missed"
            sr[side] = {eid: side_map[eid] for eid in eids}  # keep only the listed eids, in order
        parsed["span_resolution"] = sr
        return parsed
    except (KeyError, TypeError):
        return None


# ---------------- Caller ----------------
async def call_judge(
    judge_spec,
    sid: int,
    source: str,
    error_spans: list[dict],
    cand_a: dict,    # {"cid": int, "text": str, ...}
    cand_b: dict,
    *,
    cache: dict[str, dict] | None = None,
) -> dict | None:
    """Returns validated judge response dict, or None if irrecoverable failure."""
    key = judge_cache_key(judge_spec.model_id, sid, cand_a["cid"], cand_b["cid"])
    if cache is not None and key in cache:
        return cache[key]["response"]

    prompt = build_judge_prompt(source, error_spans, cand_a["text"], cand_b["text"])
    eids = [sp["eid"] for sp in error_spans]

    try:
        raw = await call_model(
            judge_spec,
            system=JUDGE_SYSTEM,
            user=prompt,
            temperature=0.0,
            json_mode=True,
        )
    except Exception as e:
        log.warning("judge %s sid=%d failed: %s", judge_spec.model_id, sid, e)
        return None

    parsed = parse_json_loose(raw)
    validated = _validate(parsed, eids)
    if validated is None:
        log.warning("judge %s sid=%d returned invalid JSON: %.200s",
                    judge_spec.model_id, sid, raw)
        return None

    record = {
        "key": key,
        "judge": judge_spec.model_id,
        "sid": sid,
        "cid_a": cand_a["cid"],
        "cid_b": cand_b["cid"],
        "response": validated,
    }
    append_jsonl(record, config.PATHS["judge_cache"])
    if cache is not None:
        cache[key] = record
    return validated


def load_judge_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for row in load_jsonl(config.PATHS["judge_cache"]):
        cache[row["key"]] = row
    return cache


# ---------------- Convenience: call all 3 judges concurrently ----------------
async def judge_all(
    sid: int,
    source: str,
    error_spans: list[dict],
    cand_a: dict,
    cand_b: dict,
    *,
    cache: dict[str, dict] | None = None,
) -> dict[str, dict | None]:
    """Returns {judge_model_id: validated_response_or_None}."""
    tasks = [
        call_judge(j, sid, source, error_spans, cand_a, cand_b, cache=cache)
        for j in config.JUDGES
    ]
    results = await asyncio.gather(*tasks)
    return {j.model_id: r for j, r in zip(config.JUDGES, results)}
