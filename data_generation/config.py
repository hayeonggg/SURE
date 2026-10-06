"""Preference data generation pipeline configuration.

API keys are read from environment variables. Set them before running:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# -------- Paths --------
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "sources"            # written by prepare_sources.py
OUTPUT_DIR = ROOT / "data_generation" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source JSON files (built by prepare_sources.py)
# CoNLL-2014 is not used because its sources overlap with SEEDA.
SOURCES = {
    "bea2019": SOURCE_DIR / "bea2019_dev_gold.json",
    "jfleg": SOURCE_DIR / "jfleg.json",
}

# Intermediate / final outputs
PATHS = {
    "filtered_sources": OUTPUT_DIR / "filtered_sources.json",
    "candidate_pool":   OUTPUT_DIR / "candidate_pool.json",
    "error_spans":      OUTPUT_DIR / "error_spans.json",
    "judge_cache":      OUTPUT_DIR / "judge_cache.jsonl",
    "pairs":            OUTPUT_DIR / "pairs.json",
    "final_dataset":    OUTPUT_DIR / "preference_pairs.json",
}


# -------- Source filters --------
MIN_TOKENS = 10
MIN_ERRANT_ERRORS = 2

# Maximum number of sources kept per dataset
SOURCE_TARGETS = {
    "bea2019":  4000,
    "jfleg":    1500,
}

# Smoke test: `SMOKE_N=10 python run_pipeline.py` shrinks every stage to N
# for a cheap end-to-end check.
_SMOKE_N = int(os.getenv("SMOKE_N", "0"))


# -------- Candidate pool --------
N_LLM_REWRITES = 2                # GPT-4o candidates per source (minimal-edit + rewrite)
# Number of sources processed concurrently when calling GPT-4o. Each source issues
# N_LLM_REWRITES calls in parallel, so in-flight calls ≈ REWRITE_SOURCE_CONCURRENCY × 2.
REWRITE_SOURCE_CONCURRENCY = 3
# A source is dropped if fewer than this many candidates remain after removing
# source copies (candidates that correct nothing).
MIN_CANDIDATES_PER_SOURCE = 3


# -------- Generator / judge models --------
@dataclass(frozen=True)
class ModelSpec:
    role: str            # "generator" | "judge"
    provider: str        # "openai" | "anthropic" | "xai"
    model_id: str


GENERATOR_MODEL = ModelSpec("generator", "openai", "gpt-4o")

JUDGES: list[ModelSpec] = [
    ModelSpec("judge", "openai",    "gpt-4.1-mini"),
    ModelSpec("judge", "anthropic", "claude-haiku-4-5-20251001"),
    ModelSpec("judge", "xai",       "grok-4.3"),
]


# -------- Pair sampling / consensus --------
TARGET_PAIRS = 2400               # final dataset size
MAX_PAIRS_PER_SOURCE = 4          # cap to keep source diversity
PAIR_OVERSAMPLE_FACTOR = 2        # try this many random pairs per source first

# Pair quota per dataset; must sum to TARGET_PAIRS.
# Once a dataset reaches its quota, no more of its pairs are accepted.
DATASET_PAIR_QUOTAS: dict[str, int] = {
    "bea2019":   1500,
    "jfleg":      900,
}
assert sum(DATASET_PAIR_QUOTAS.values()) == TARGET_PAIRS, \
    "DATASET_PAIR_QUOTAS must sum to TARGET_PAIRS"

# Number of pairs judged concurrently. Each pair calls the three judges in parallel,
# so in-flight LLM calls ≈ PAIR_CONCURRENCY * 3 (within each provider's max_concurrency).
PAIR_CONCURRENCY = 8


# -------- Apply SMOKE_N override (must come after all knobs are defined) --------
if _SMOKE_N > 0:
    # keep only N sources, spread evenly over the datasets
    _datasets = tuple(SOURCE_TARGETS)
    _per = max(1, _SMOKE_N // len(_datasets))
    _remain = _SMOKE_N - _per * len(_datasets)
    SOURCE_TARGETS = {
        ds: _per + (1 if i < _remain else 0) for i, ds in enumerate(_datasets)
    }
    TARGET_PAIRS = _SMOKE_N
    MAX_PAIRS_PER_SOURCE = 2
    # keep the BEA/JFLEG ratio (1500:900) so the quota logic is exercised
    _b = round(_SMOKE_N * 0.625)
    DATASET_PAIR_QUOTAS = {"bea2019": _b, "jfleg": _SMOKE_N - _b}
    PAIR_CONCURRENCY = 4


# -------- API call settings --------
@dataclass
class APISettings:
    max_retries: int = 5
    initial_backoff: float = 2.0
    request_timeout: float = 60.0
    max_concurrency: int = 16      # async semaphore for each provider (default)
    # Lower limit for Anthropic to stay within low-tier rate limits (50 RPM).
    anthropic_max_concurrency: int = 2

    openai_api_key:    str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    xai_api_key:       str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))


API = APISettings()


# -------- Reproducibility --------
RANDOM_SEED = 20260521
