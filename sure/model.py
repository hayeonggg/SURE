"""SURE reward model (span-aware, multi-criteria).

Backbone:
  - DeBERTa-v3-large (pre-trained, AutoModel)
  - LoRA adapters (only the attention projections are trained)
  - Embeddings are resized for the [ERR]/[/ERR] special tokens (before wrapping with LoRA)

Heads (full training):
  - Span head:    h_eᵢ → 3-way logits (resolved/unresolved/harmful)  ← auxiliary
  - Axis heads:   h_cls → 3 scalars (grammaticality, faithfulness, fluency)
  - Overall head: h_cls → 1 scalar R(s, y)

The span head supervises the backbone through L_span during training, but only
axis_scores and overall_reward are used at inference (reference-free).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel

from . import config


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
    )


class SpanAwareRewardModel(nn.Module):
    """LoRA-wrapped DeBERTa + 3 heads (span / axis / overall).

    Parameters
    ----------
    base_model : str
        HF model id (e.g. "microsoft/deberta-v3-large").
    vocab_size : int
        Final tokenizer vocabulary size (including the added special tokens).
    """

    def __init__(self, base_model: str = config.BASE_MODEL,
                 vocab_size: int | None = None) -> None:
        super().__init__()

        # Load in float32 explicitly: newer transformers versions may pick up fp16 safetensors,
        # which would not match the dtype of the custom heads (fp32).
        base = AutoModel.from_pretrained(base_model, torch_dtype=torch.float32)
        # Resize the vocabulary for the special tokens (must happen before wrapping with LoRA)
        if vocab_size is not None and vocab_size != base.config.vocab_size:
            base.resize_token_embeddings(vocab_size)

        # Apply LoRA
        self.encoder = get_peft_model(base, build_lora_config())
        hidden = base.config.hidden_size

        # ---- Heads (full training) ----
        # Span head: auxiliary 3-way classifier
        self.span_classifier = nn.Linear(hidden, len(config.SPAN_CLASSES))

        # Axis heads: 3 separate scalar heads
        self.axis_heads = nn.ModuleDict({
            ax: nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            for ax in config.AXES
        })

        # Overall reward head
        self.overall_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    # ------------------------------------------------------------------
    def print_trainable_parameters(self) -> None:
        """Print the share of trainable parameters (LoRA + heads)."""
        trainable, total = 0, 0
        for n, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        pct = 100.0 * trainable / max(total, 1)
        print(
            f"[params] trainable={trainable:,} / total={total:,} "
            f"({pct:.2f}%)  ← LoRA + custom heads"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_span_repr(
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        span_open_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take the hidden state at each [ERR] open token as the per-span representation."""
        B, T, H = last_hidden.shape
        is_open = input_ids == span_open_id

        # positions of [ERR] in each batch item
        per_batch = []
        max_spans = 0
        for b in range(B):
            positions = is_open[b].nonzero(as_tuple=False).flatten()
            max_spans = max(max_spans, positions.numel())
            per_batch.append(last_hidden[b][positions])

        if max_spans == 0:
            return (
                last_hidden.new_zeros(B, 0, H),
                last_hidden.new_zeros(B, 0),
            )

        span_h   = last_hidden.new_zeros(B, max_spans, H)
        span_msk = last_hidden.new_zeros(B, max_spans)
        for b, h in enumerate(per_batch):
            n = h.size(0)
            if n > 0:
                span_h[b, :n]   = h
                span_msk[b, :n] = 1.0
        return span_h, span_msk

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_open_id: int,
        compute_span: bool = True,
    ) -> dict[str, Any]:
        """Forward pass.

        compute_span: if False, skip the span head (used at inference).
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state                   # [B, T, H]
        # Force fp32 so the dtype matches the heads even if the encoder runs in fp16
        if last_hidden.dtype != torch.float32:
            last_hidden = last_hidden.float()
        cls_h = last_hidden[:, 0, :]                          # [B, H]

        # Axis heads
        axis_scores = {
            ax: head(cls_h).squeeze(-1)                       # [B]
            for ax, head in self.axis_heads.items()
        }

        # Overall reward
        overall_reward = self.overall_head(cls_h).squeeze(-1) # [B]

        result: dict[str, Any] = {
            "cls_h":          cls_h,
            "axis_scores":    axis_scores,
            "overall_reward": overall_reward,
        }

        # Span head (auxiliary; only meaningful during training)
        if compute_span:
            span_h, span_mask = self._extract_span_repr(
                last_hidden, input_ids, span_open_id,
            )
            if span_h.size(1) > 0:
                span_logits = self.span_classifier(span_h)    # [B, S, n_classes]
            else:
                span_logits = span_h.new_zeros(
                    span_h.size(0), 0, len(config.SPAN_CLASSES),
                )
            result["span_h"]      = span_h
            result["span_mask"]   = span_mask
            result["span_logits"] = span_logits

        return result
