"""SURE reward model — training script.

Usage (run from the repository root):
  python -m sure.train                                   # train with the paper's configuration
  python -m sure.train --data path/to/pairs.json --epochs 3 --run-name my_run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from . import config
from .data import (
    RewardCollator, RewardModelDataset, ensure_special_tokens,
    load_pairs, split_pairs,
)
from .losses import (
    axis_accuracy, compute_total_loss, pair_accuracy, span_accuracy,
)
from .model import SpanAwareRewardModel


# ---------------- Reproducibility ----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    n = 0
    sum_pair_acc = 0.0
    sum_span_acc_pos = 0.0
    sum_span_acc_neg = 0.0
    sum_axis_acc = {ax: 0.0 for ax in config.AXES}
    axis_counts  = {ax: 0   for ax in config.AXES}
    sum_loss_parts = {"L_pair": 0.0, "L_axis": 0.0, "L_span": 0.0, "total": 0.0}

    for batch in loader:
        ids_pos = batch["input_ids_pos"].to(device)
        msk_pos = batch["attention_mask_pos"].to(device)
        ids_neg = batch["input_ids_neg"].to(device)
        msk_neg = batch["attention_mask_neg"].to(device)
        span_labels_pos = batch["span_labels_pos"].to(device)
        span_labels_neg = batch["span_labels_neg"].to(device)
        axis_labels = {ax: t.to(device) for ax, t in batch["axis_labels"].items()}
        span_open_id = batch["span_open_id"]

        out_pos = model(ids_pos, msk_pos, span_open_id, compute_span=True)
        out_neg = model(ids_neg, msk_neg, span_open_id, compute_span=True)

        _, parts = compute_total_loss(
            out_pos, out_neg, span_labels_pos, span_labels_neg, axis_labels,
        )
        for k in sum_loss_parts:
            sum_loss_parts[k] += parts[k]

        sum_pair_acc     += pair_accuracy(out_pos["overall_reward"], out_neg["overall_reward"])
        sum_span_acc_pos += span_accuracy(out_pos["span_logits"], span_labels_pos)
        sum_span_acc_neg += span_accuracy(out_neg["span_logits"], span_labels_neg)
        ax_accs = axis_accuracy(out_pos["axis_scores"], out_neg["axis_scores"], axis_labels)
        for ax in config.AXES:
            if not math.isnan(ax_accs[ax]):
                sum_axis_acc[ax] += ax_accs[ax]
                axis_counts[ax]  += 1
        n += 1

    metrics = {
        "loss_total":   sum_loss_parts["total"]  / max(n, 1),
        "loss_pair":    sum_loss_parts["L_pair"] / max(n, 1),
        "loss_axis":    sum_loss_parts["L_axis"] / max(n, 1),
        "loss_span":    sum_loss_parts["L_span"] / max(n, 1),
        "pair_acc":     sum_pair_acc / max(n, 1),
        "span_acc_pos": sum_span_acc_pos / max(n, 1),
        "span_acc_neg": sum_span_acc_neg / max(n, 1),
    }
    for ax in config.AXES:
        metrics[f"axis_acc_{ax}"] = (
            sum_axis_acc[ax] / axis_counts[ax] if axis_counts[ax] > 0 else float("nan")
        )
    return metrics


# ---------------- Checkpoint helpers ----------------
def save_checkpoint(model, tokenizer, ckpt_dir: Path,
                    epoch: int, val_metrics: dict | None) -> None:
    """Save the LoRA adapter, custom heads, and tokenizer separately.

    Layout:
      ckpt_dir/
        adapter_model.safetensors / adapter_config.json   (peft)
        heads.pt                                          (span/axis/overall)
        tokenizer/                                        (with the special tokens)
        metadata.json                                     (epoch, metrics)
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) LoRA adapter
    model.encoder.save_pretrained(ckpt_dir)

    # 2) Custom heads
    heads_state = {
        "span_classifier": model.span_classifier.state_dict(),
        "axis_heads":      {ax: h.state_dict() for ax, h in model.axis_heads.items()},
        "overall_head":    model.overall_head.state_dict(),
    }
    torch.save(heads_state, ckpt_dir / "heads.pt")

    # 3) Tokenizer (with the special tokens added)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")

    # 4) Metadata
    meta = {
        "base_model":   config.BASE_MODEL,
        "vocab_size":   len(tokenizer),
        "max_length":   config.MAX_LENGTH,
        "span_classes": config.SPAN_CLASSES,
        "axes":         config.AXES,
        "epoch":        epoch,
        "val_metrics":  val_metrics,
    }
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)


# ---------------- Train loop ----------------
def train(args: argparse.Namespace) -> None:
    set_seed(config.SEED)

    device = torch.device("cuda" if (torch.cuda.is_available() and config.PREFER_GPU) else "cpu")
    print(f"[device] {device}")

    # ---- Output directory ----
    # with --run-name: runs/<name>/{checkpoints,logs}
    # otherwise: config defaults (checkpoints/, logs/)
    if args.run_name:
        run_root = config.REPO_ROOT / "runs" / args.run_name
        ckpt_dir = run_root / "checkpoints"
        log_dir  = run_root / "logs"
    else:
        ckpt_dir = config.CHECKPOINT_DIR
        log_dir  = config.LOG_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] ckpt_dir={ckpt_dir}")
    print(f"[output] log_dir ={log_dir}")

    # ---- Data ----
    pairs = load_pairs(Path(args.data) if args.data else None)
    print(f"[data] loaded {len(pairs)} pairs")
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=config.VAL_RATIO, seed=config.SEED)
    print(f"[data] train={len(train_pairs)} val={len(val_pairs)}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    tokenizer = ensure_special_tokens(tokenizer)
    print(f"[tokenizer] {config.BASE_MODEL} vocab={len(tokenizer)} (+special tokens)")

    # ---- Model (with LoRA) ----
    model = SpanAwareRewardModel(config.BASE_MODEL, vocab_size=len(tokenizer)).to(device)
    model.print_trainable_parameters()

    # ---- DataLoaders ----
    collator = RewardCollator(tokenizer)
    train_loader = DataLoader(
        RewardModelDataset(train_pairs), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        RewardModelDataset(val_pairs), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ---- Optimizer + Scheduler ----
    grad_accum = args.grad_accum
    optim_steps = math.ceil(len(train_loader) / grad_accum) * args.epochs
    warmup_steps = int(optim_steps * config.WARMUP_RATIO)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optim_steps)
    print(f"[opt] AdamW lr={args.lr} warmup={warmup_steps}/{optim_steps} "
          f"grad_accum={grad_accum} effective_bs={args.batch_size * grad_accum}")

    # ---- Train ----
    best_val_pair_acc = -1.0
    metrics_log: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"L_pair": 0.0, "L_axis": 0.0, "L_span": 0.0, "total": 0.0}
        microstep = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            ids_pos = batch["input_ids_pos"].to(device)
            msk_pos = batch["attention_mask_pos"].to(device)
            ids_neg = batch["input_ids_neg"].to(device)
            msk_neg = batch["attention_mask_neg"].to(device)
            span_labels_pos = batch["span_labels_pos"].to(device)
            span_labels_neg = batch["span_labels_neg"].to(device)
            axis_labels = {ax: t.to(device) for ax, t in batch["axis_labels"].items()}
            span_open_id = batch["span_open_id"]

            out_pos = model(ids_pos, msk_pos, span_open_id, compute_span=True)
            out_neg = model(ids_neg, msk_neg, span_open_id, compute_span=True)

            loss, parts = compute_total_loss(
                out_pos, out_neg, span_labels_pos, span_labels_neg, axis_labels,
            )
            (loss / grad_accum).backward()

            for k in running:
                running[k] += parts[k]
            microstep += 1

            if microstep % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.GRAD_CLIP,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % config.LOG_EVERY == 0:
                    avg = {k: v / (config.LOG_EVERY * grad_accum) for k, v in running.items()}
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[ep {epoch} step {global_step}] "
                        f"total={avg['total']:.4f} pair={avg['L_pair']:.4f} "
                        f"axis={avg['L_axis']:.4f} span={avg['L_span']:.4f} "
                        f"lr={lr:.2e} elapsed={time.time()-t0:.0f}s"
                    )
                    running = {k: 0.0 for k in running}

        # flush the remaining micro-batches at the end of the epoch
        if microstep % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- Eval ----
        if epoch % config.EVAL_EVERY == 0:
            val_metrics = evaluate(model, val_loader, device)
            tag = f"[ep {epoch} VAL]"
            print(tag + " " + ", ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
                if not (isinstance(v, float) and math.isnan(v))
            ))
            metrics_log.append({"epoch": epoch, **val_metrics})

            if val_metrics["pair_acc"] > best_val_pair_acc:
                best_val_pair_acc = val_metrics["pair_acc"]
                best_dir = ckpt_dir / f"rm_best_ep{epoch}_acc{val_metrics['pair_acc']:.3f}"
                save_checkpoint(model, tokenizer, best_dir, epoch, val_metrics)
                print(f"[ckpt] saved best to {best_dir}")

    # ---- Final save ----
    final_dir = ckpt_dir / "rm_last"
    save_checkpoint(model, tokenizer, final_dir, args.epochs,
                    metrics_log[-1] if metrics_log else None)
    print(f"[ckpt] saved final to {final_dir}")

    log_path = log_dir / "train_metrics.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, ensure_ascii=False, indent=2, default=float)
    print(f"[log] metrics → {log_path}")


# ---------------- Entrypoint ----------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default=None,
                   help="path to the preference pairs JSON (default: data/sure_preference_pairs.json)")
    p.add_argument("--epochs",     type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    p.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    p.add_argument("--alpha",      type=float, default=None,
                   help="weight of L_axis (overrides config.LAMBDA_AXIS)")
    p.add_argument("--beta",       type=float, default=None,
                   help="weight of L_span (overrides config.LAMBDA_SPAN)")
    p.add_argument("--run-name",   type=str, default=None,
                   help="save outputs to runs/<name>/{checkpoints,logs}; "
                        "otherwise checkpoints/ and logs/")
    args = p.parse_args()
    if args.alpha is not None:
        config.LAMBDA_AXIS = args.alpha
    if args.beta is not None:
        config.LAMBDA_SPAN = args.beta
    print(f"[loss] α(LAMBDA_AXIS)={config.LAMBDA_AXIS}  β(LAMBDA_SPAN)={config.LAMBDA_SPAN}")
    train(args)


if __name__ == "__main__":
    main()
