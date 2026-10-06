"""Shared helpers for the preference data generation pipeline."""
from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


# -------- Logging --------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# -------- JSON I/O --------
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(record: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# -------- Tokenization (for MIN_TOKENS gate) --------
_TOKEN_RE = re.compile(r"\S+")


def whitespace_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# -------- Detokenization --------
# The sources/references of BEA-2019 and JFLEG are stored in Treebank-style
# tokenization (e.g. "It 's", "do n't", "families '"), whereas GPT-4o outputs
# use natural formatting. Showing them as is would let the judges prefer GPT-4o
# outputs for formatting alone, so the same detokenization is applied to every
# text shown to the judges (source + candidates).
def detokenize(text: str) -> str:
    if not text:
        return text

    # 1. attach English contractions: " 's" / " n't" / " 're" / " 've" / " 'll" / " 'm" / " 'd"
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'m|'d)\b", r"\1", text)

    # 2. possessive apostrophe after a word: "families '" → "families'"
    text = re.sub(r"(\w)\s+'(?=\s|$|[.,!?;:\"])", r"\1'", text)

    # 3. remove the space before punctuation: "word ." → "word."
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # 4. remove the spaces inside paired double quotes: '" text "' → '"text"'
    #    (applied twice to cover nested cases)
    for _ in range(2):
        text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
        text = re.sub(r'"\s+([^"]*?)"',   r'"\1"', text)
        text = re.sub(r'"([^"]*?)\s+"',   r'"\1"', text)

    # 5. remove the spaces inside parentheses: "( text )" → "(text)"
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # 6. collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------- Source-copy detection --------
# Decides whether a candidate is effectively identical to the source (nothing corrected).
# The comparison is an exact match after lowercasing, removing punctuation, and collapsing spaces.
# It is deliberately conservative: a candidate with any real edit is not a source copy.
def _norm_for_compare(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)   # remove all non-word, non-space characters
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_source_copy(candidate: str, source: str) -> bool:
    """True if the candidate is effectively the same text as the source."""
    return _norm_for_compare(candidate) == _norm_for_compare(source)


# -------- Timing --------
@contextmanager
def timed(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info("▶ %s ...", label)
    try:
        yield
    finally:
        logger.info("✔ %s done in %.1fs", label, time.time() - t0)
