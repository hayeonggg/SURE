"""SURE reward model — configuration.

Backbone: DeBERTa-v3-large (pre-trained) + LoRA fine-tuning.
Span head: auxiliary only (supervises the backbone during training, unused at inference).
"""
from __future__ import annotations

from pathlib import Path
import os


# -------- Paths --------
REPO_ROOT = Path(os.environ.get("SURE_ROOT", Path(__file__).resolve().parents[1]))

# Preference data used in the paper (2,400 pairs)
DATA_PATH = REPO_ROOT / "data" / "sure_preference_pairs.json"

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR        = REPO_ROOT / "logs"

# Released checkpoint (Hugging Face Hub repo id); downloaded when --ckpt is not given.
HF_CHECKPOINT = "hayeonggg/SURE"


# -------- Backbone --------
BASE_MODEL = "microsoft/deberta-v3-large"
MAX_LENGTH = 256

# Special tokens marking error-span positions in the source
SPAN_OPEN_TOKEN  = "[ERR]"
SPAN_CLOSE_TOKEN = "[/ERR]"


# -------- LoRA --------
# Attention projection modules of DeBERTa-v3
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.1
LORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj"]
LORA_BIAS           = "none"


# -------- Span taxonomy (3-way) --------
SPAN_CLASSES      = ["resolved", "unresolved", "harmful"]
SPAN_CLASS_TO_IDX = {c: i for i, c in enumerate(SPAN_CLASSES)}
SPAN_IDX_TO_CLASS = {i: c for c, i in SPAN_CLASS_TO_IDX.items()}

# Mapping from the span labels in the preference data (resolved/missed/partial/new_error)
# to the taxonomy above; partial is merged into unresolved.
SPAN_LABEL_REMAP = {
    "resolved":  "resolved",
    "missed":    "unresolved",
    "partial":   "unresolved",
    "new_error": "harmful",      # not present in the current data
}


# -------- Axes --------
AXES = ["grammaticality", "faithfulness", "fluency"]


# -------- Training --------
SEED          = 42
VAL_RATIO     = 0.10
BATCH_SIZE    = 4
GRAD_ACCUM    = 2              # effective batch size = BATCH_SIZE × GRAD_ACCUM
NUM_EPOCHS    = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
GRAD_CLIP     = 1.0
LOG_EVERY     = 20             # steps
EVAL_EVERY    = 1              # epochs


# -------- Loss weights --------
# L = L_pair + α·L_axis + β·L_span
# β is kept small so that the span head does not dominate the axis/overall heads.
LAMBDA_AXIS = 0.5     # α
LAMBDA_SPAN = 0.2     # β (auxiliary)
PAIR_MARGIN = 0.0


# -------- Device --------
PREFER_GPU = True
