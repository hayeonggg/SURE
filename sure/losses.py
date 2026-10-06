"""SURE reward model — loss components.

L = L_pair + α·L_axis + β·L_span

L_pair:   Bradley-Terry pairwise ranking on overall_reward
L_axis:   per-axis pairwise ranking (ties excluded)
L_span:   per-span 3-way cross-entropy (auxiliary; small β)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import config


# ---------------- L_pair ----------------
def pair_loss(R_pos: torch.Tensor, R_neg: torch.Tensor,
              margin: float = config.PAIR_MARGIN) -> torch.Tensor:
    """Bradley-Terry: L = -log σ(R⁺ − R⁻ − margin)."""
    diff = R_pos - R_neg - margin
    return F.softplus(-diff).mean()


# ---------------- L_axis ----------------
def axis_loss(axis_scores_pos: dict[str, torch.Tensor],
              axis_scores_neg: dict[str, torch.Tensor],
              axis_labels: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-axis Bradley-Terry. Tied samples are excluded."""
    losses = []
    for ax in config.AXES:
        labels = axis_labels[ax].float()
        non_tie = labels != 0
        if non_tie.sum() == 0:
            continue
        s_pos = axis_scores_pos[ax][non_tie]
        s_neg = axis_scores_neg[ax][non_tie]
        l = labels[non_tie]                                   # ±1
        diff = l * (s_pos - s_neg)
        losses.append(F.softplus(-diff).mean())
    if not losses:
        any_score = next(iter(axis_scores_pos.values()))
        return any_score.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------- L_span ----------------
def span_loss(span_logits: torch.Tensor,
              span_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over span class.
    span_logits: [B, S_model, n_classes]
      S_model = number of [ERR] tokens the model found in the input (batch max).
              May be smaller than n_spans in the data because of truncation.
    span_labels: [B, S_data] (-100 = ignore)
      S_data  = max n_spans in the data (batch max).
    When the two differ, both are cut to min(S_model, S_data); truncated spans are ignored.
    """
    if span_logits.numel() == 0:
        return span_logits.new_tensor(0.0, requires_grad=True)
    B, S_model, C = span_logits.shape
    S_data = span_labels.shape[1]
    S = min(S_model, S_data)
    if S == 0:
        return span_logits.new_tensor(0.0, requires_grad=True)
    return F.cross_entropy(
        span_logits[:, :S, :].reshape(B * S, C),
        span_labels[:, :S].reshape(B * S),
        ignore_index=-100,
    )


# ---------------- Combined ----------------
def compute_total_loss(
    out_pos: dict[str, torch.Tensor],
    out_neg: dict[str, torch.Tensor],
    span_labels_pos: torch.Tensor,
    span_labels_neg: torch.Tensor,
    axis_labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    L_pair = pair_loss(out_pos["overall_reward"], out_neg["overall_reward"])
    L_axis = axis_loss(out_pos["axis_scores"], out_neg["axis_scores"], axis_labels)

    # The span head is trained only when 'span_logits' is present in out_pos/out_neg
    if "span_logits" in out_pos and "span_logits" in out_neg:
        L_span_pos = span_loss(out_pos["span_logits"], span_labels_pos)
        L_span_neg = span_loss(out_neg["span_logits"], span_labels_neg)
        L_span = 0.5 * (L_span_pos + L_span_neg)
    else:
        L_span = out_pos["overall_reward"].new_tensor(0.0, requires_grad=True)

    total = (
        L_pair
        + config.LAMBDA_AXIS * L_axis
        + config.LAMBDA_SPAN * L_span
    )

    return total, {
        "L_pair": L_pair.item(),
        "L_axis": L_axis.item(),
        "L_span": L_span.item(),
        "total":  total.item(),
    }


# ---------------- Metrics ----------------
@torch.no_grad()
def pair_accuracy(R_pos: torch.Tensor, R_neg: torch.Tensor) -> float:
    return (R_pos > R_neg).float().mean().item()


@torch.no_grad()
def span_accuracy(span_logits: torch.Tensor, span_labels: torch.Tensor) -> float:
    # handle shape mismatch (same logic as span_loss)
    if span_logits.numel() == 0:
        return 0.0
    S_model = span_logits.shape[1]
    S_data = span_labels.shape[1]
    S = min(S_model, S_data)
    if S == 0:
        return 0.0
    logits = span_logits[:, :S, :]
    labels = span_labels[:, :S]
    valid = labels != -100
    if valid.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[valid] == labels[valid]).float().mean().item()


@torch.no_grad()
def axis_accuracy(axis_scores_pos: dict[str, torch.Tensor],
                  axis_scores_neg: dict[str, torch.Tensor],
                  axis_labels: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for ax in config.AXES:
        labels = axis_labels[ax]
        non_tie = labels != 0
        if non_tie.sum() == 0:
            out[ax] = float("nan")
            continue
        s_pos = axis_scores_pos[ax][non_tie]
        s_neg = axis_scores_neg[ax][non_tie]
        l = labels[non_tie].float()
        diff = l * (s_pos - s_neg)
        out[ax] = (diff > 0).float().mean().item()
    return out
