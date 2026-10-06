"""SURE reward model — data loading + preprocessing.

Processing:
  1) load the preference pairs (data/sure_preference_pairs.json)
  2) mark error spans in source_detok with the [ERR]/[/ERR] special tokens (training only)
  3) tokenize (source_with_markers, candidate)
  4) encode both y_plus and y_minus this way
  5) map span labels (resolved/missed/partial) to the taxonomy (resolved/unresolved/harmful)
  6) convert criteria preferences 'A/B/tie' to '+1/-1/0' from the viewpoint of y_plus

At inference the source is given as is, without [ERR] markers (reference-free).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from . import config


# ---------------- Loading ----------------
def load_pairs(path: Path | None = None) -> list[dict]:
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [config.DATA_PATH]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"preference data not found in: {[str(p) for p in candidates]}"
    )


# ---------------- Span marker injection ----------------
def inject_span_markers(source_detok: str, error_spans: list[dict]) -> str:
    """Mark the error spans in source_detok with [ERR]...[/ERR].

    Returns source_detok unchanged if error_spans is empty or None,
    so calling it without error_spans gives the marker-free form used at inference.
    """
    if not error_spans:
        return source_detok

    spans = sorted(error_spans, key=lambda s: (s.get("o_start") or 0))
    parts: list[str] = []
    cursor = 0
    for sp in spans:
        span_text = (sp.get("source_span") or "").strip()
        if not span_text:
            continue
        idx = source_detok.find(span_text, cursor)
        if idx == -1:
            continue
        parts.append(source_detok[cursor:idx])
        parts.append(f" {config.SPAN_OPEN_TOKEN} ")
        parts.append(span_text)
        parts.append(f" {config.SPAN_CLOSE_TOKEN} ")
        cursor = idx + len(span_text)
    parts.append(source_detok[cursor:])
    return " ".join("".join(parts).split())


# ---------------- Label utilities ----------------
def remap_span_label(raw_label: str) -> str:
    """Map a span label in the preference data to the 3-way taxonomy."""
    return config.SPAN_LABEL_REMAP.get(raw_label, "unresolved")


def convert_axis_labels(axis_prefs: dict[str, str],
                        overall_votes: list[str]) -> dict[str, int]:
    """A/B/tie -> +1/-1/0 from the viewpoint of y_plus.

    overall_votes[0] is the side of y_plus (always A or B, since only unanimous pairs are kept).
    """
    y_plus_side  = overall_votes[0]
    y_minus_side = "B" if y_plus_side == "A" else "A"
    out: dict[str, int] = {}
    for ax in config.AXES:
        pref = axis_prefs.get(ax, "tie")
        if pref == y_plus_side:
            out[ax] = +1
        elif pref == y_minus_side:
            out[ax] = -1
        else:
            out[ax] = 0
    return out


# ---------------- Dataset ----------------
class RewardModelDataset(Dataset):
    """Training dataset. Always uses the source with [ERR] markers."""

    def __init__(self, pairs: list[dict]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.pairs[idx]
        source_detok = row.get("source_detok") or row["source"]
        error_spans  = row.get("error_spans", [])
        eids         = [sp["eid"] for sp in error_spans]

        # training: source with markers
        source_marked = inject_span_markers(source_detok, error_spans)

        y_plus_text  = row["y_plus"]["text"]
        y_minus_text = row["y_minus"]["text"]

        def _labels_for(side: str) -> list[int]:
            label_map = row.get(f"span_labels_{side}", {})
            arr = []
            for eid in eids:
                raw_cls = label_map.get(eid, "missed")
                new_cls = remap_span_label(raw_cls)
                arr.append(config.SPAN_CLASS_TO_IDX[new_cls])
            return arr

        return {
            "source_marked":   source_marked,
            "y_plus_text":     y_plus_text,
            "y_minus_text":    y_minus_text,
            "span_labels_pos": _labels_for("y_plus"),
            "span_labels_neg": _labels_for("y_minus"),
            "axis_labels":     convert_axis_labels(
                row["axis_preferences"],
                row.get("overall_votes") or row.get("meta", {}).get("overall_votes", []),
            ),
            "n_spans":         len(eids),
            "sid":             row.get("sid", idx),
        }


# ---------------- Collator ----------------
class RewardCollator:
    """Batch tokenization and tensor padding."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self.span_open_id = tokenizer.convert_tokens_to_ids(config.SPAN_OPEN_TOKEN)
        if self.span_open_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"{config.SPAN_OPEN_TOKEN} not registered. "
                "Call ensure_special_tokens(tokenizer) first."
            )

    def _encode(self, sources: list[str], cands: list[str]):
        return self.tokenizer(
            sources, cands,
            padding=True, truncation=True, max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        sources = [b["source_marked"] for b in batch]
        enc_pos = self._encode(sources, [b["y_plus_text"]  for b in batch])
        enc_neg = self._encode(sources, [b["y_minus_text"] for b in batch])

        max_spans = max((b["n_spans"] for b in batch), default=0)
        B = len(batch)
        # -100 = CE ignore_index
        span_labels_pos = torch.full((B, max(max_spans, 1)), -100, dtype=torch.long)
        span_labels_neg = torch.full((B, max(max_spans, 1)), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["n_spans"]
            if n > 0:
                span_labels_pos[i, :n] = torch.tensor(b["span_labels_pos"], dtype=torch.long)
                span_labels_neg[i, :n] = torch.tensor(b["span_labels_neg"], dtype=torch.long)

        axis_labels = {
            ax: torch.tensor([b["axis_labels"][ax] for b in batch], dtype=torch.long)
            for ax in config.AXES
        }

        return {
            "input_ids_pos":     enc_pos["input_ids"],
            "attention_mask_pos": enc_pos["attention_mask"],
            "input_ids_neg":     enc_neg["input_ids"],
            "attention_mask_neg": enc_neg["attention_mask"],
            "span_labels_pos":   span_labels_pos,
            "span_labels_neg":   span_labels_neg,
            "axis_labels":       axis_labels,
            "span_open_id":      self.span_open_id,
        }


# ---------------- Tokenizer setup ----------------
def ensure_special_tokens(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Add [ERR] and [/ERR] as special tokens."""
    tokenizer.add_special_tokens({
        "additional_special_tokens": [config.SPAN_OPEN_TOKEN, config.SPAN_CLOSE_TOKEN]
    })
    return tokenizer


# ---------------- Train/val split ----------------
def split_pairs(pairs: list[dict], val_ratio: float = config.VAL_RATIO,
                seed: int = config.SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_idx = set(indices[:n_val])
    train, val = [], []
    for i, p in enumerate(pairs):
        (val if i in val_idx else train).append(p)
    return train, val
