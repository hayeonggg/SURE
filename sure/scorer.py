"""SURE reward model — inference (reference-free).

Training:  [ERR] markers are inserted into the source and the span head is trained as an auxiliary task.
Inference: only (source, candidate) is given, without markers, and the criteria and overall scores
           are returned, so neither ERRANT annotation nor a reference is needed.

Usage:
  python -m sure.scorer \
      --source "He go to school yesterday." \
      --candidate "He went to school yesterday."
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from . import config
from .data import ensure_special_tokens
from .model import SpanAwareRewardModel


def resolve_checkpoint(ckpt: str | Path | None = None) -> Path:
    """Return a local checkpoint dir as is; otherwise treat it as a Hugging Face Hub repo id and download it."""
    ckpt = str(ckpt or config.HF_CHECKPOINT)
    if Path(ckpt).is_dir():
        return Path(ckpt)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=ckpt))


# The training data is detokenized, so Treebank-style tokenized input
# (e.g. "do n't", "word .") should be normalized the same way before scoring.
def detokenize(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'m|'d)\b", r"\1", text)
    text = re.sub(r"(\w)\s+'(?=\s|$|[.,!?;:\"])", r"\1'", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    for _ in range(2):
        text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
        text = re.sub(r'"\s+([^"]*?)"',   r'"\1"', text)
        text = re.sub(r'"([^"]*?)\s+"',   r'"\1"', text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class RewardScorer:
    """Loads a checkpoint (adapter + heads + tokenizer) and runs inference."""

    def __init__(self, ckpt_dir: str | Path | None = None, device: str | None = None) -> None:
        ckpt_dir = resolve_checkpoint(ckpt_dir)
        meta = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
        base_model = meta["base_model"]

        # 1) Tokenizer (saved with the special tokens)
        tok_dir = ckpt_dir / "tokenizer"
        if tok_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.tokenizer = ensure_special_tokens(self.tokenizer)

        # 2) Backbone + LoRA + custom heads
        # Build the model as in training, then load the trained LoRA adapter separately.
        self.model = SpanAwareRewardModel(
            base_model=base_model,
            vocab_size=len(self.tokenizer),
        )
        # Reload the encoder with PeftModel.from_pretrained to pick up the trained adapter.
        base_encoder = AutoModel.from_pretrained(base_model)
        base_encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.encoder = PeftModel.from_pretrained(base_encoder, ckpt_dir)

        # 3) Custom heads
        heads_state = torch.load(ckpt_dir / "heads.pt", map_location="cpu")
        self.model.span_classifier.load_state_dict(heads_state["span_classifier"])
        for ax, sd in heads_state["axis_heads"].items():
            self.model.axis_heads[ax].load_state_dict(sd)
        self.model.overall_head.load_state_dict(heads_state["overall_head"])

        self.model.eval()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.span_open_id = self.tokenizer.convert_tokens_to_ids(config.SPAN_OPEN_TOKEN)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, source: str, candidate: str) -> dict[str, Any]:
        """Score a single (source, candidate) pair.

        - no error spans needed (reference-free)
        - returns axis_scores and overall_reward (the span head is not used)
        """
        enc = self.tokenizer(
            source, candidate,
            padding=True, truncation=True, max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            span_open_id=self.span_open_id,
            compute_span=False,                  # the span head is skipped at inference
        )

        return {
            "source":         source,
            "candidate":      candidate,
            "axis_scores":    {ax: out["axis_scores"][ax][0].item() for ax in config.AXES},
            "overall_reward": out["overall_reward"][0].item(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compare(self, source: str, cand_a: str, cand_b: str) -> dict[str, Any]:
        """Compare the rewards of candidates A and B."""
        sa = self.score(source, cand_a)
        sb = self.score(source, cand_b)
        return {
            "A": sa,
            "B": sb,
            "winner": "A" if sa["overall_reward"] > sb["overall_reward"] else "B",
            "margin": sa["overall_reward"] - sb["overall_reward"],
        }


# ---------------- CLI ----------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default=None,
                   help="checkpoint dir or Hugging Face repo id (default: config.HF_CHECKPOINT)")
    p.add_argument("--source",    type=str, required=True)
    p.add_argument("--candidate", type=str, required=True)
    args = p.parse_args()

    scorer = RewardScorer(args.ckpt)
    result = scorer.score(args.source, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
