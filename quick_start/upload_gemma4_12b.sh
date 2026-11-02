#!/usr/bin/env bash
# Upload the winning trial of Gemma-4-12B-it abliteration to Hugging Face.
#
# Runs AFTER deploy_gemma4_12b.sh finishes and produces a full Optuna journal at
# $CHECKPOINT_DIR. Selects the Pareto-optimal trial (min refusals subject to
# KL ≤ KL_CEILING), re-applies its steering to a freshly-loaded base model,
# merges weights, and pushes to HF Hub.
#
# Usage (on the pod, AFTER the run completes):
#   bash /workspace/abliterix/quick_start/upload_gemma4_12b.sh
#
# Environment overrides:
#   REPO_ID          target HF repo (default: wangzhang/gemma-4-12B-it-abliterix)
#   CHECKPOINT_DIR   optuna journal dir (default: /workspace/checkpoints_gemma4_12b)
#   KL_CEILING       max KL for trial selection (default: 0.05 — published winner 0.043)
#   TRIAL            bypass auto-selection, upload this specific trial index
#   DRY_RUN=1        pick trial + print plan, don't upload
#
# Typical flow:
#   1. Run deploy_gemma4_12b.sh, wait ~3 h.
#   2. Sanity-check `tail -50 /workspace/run_gemma4_12b.log` — confirm the best
#      trial has refusals dropping (≤ ~15/100) at KL ≤ ~0.05.
#   3. bash quick_start/upload_gemma4_12b.sh
#   4. Verify at https://huggingface.co/wangzhang/gemma-4-12B-it-abliterix
#   5. Smoke-test with 15 batched harmful prompts (feedback_validation_minimal).

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/gemma4_12b.toml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_gemma4_12b}"
REPO_ID="${REPO_ID:-wangzhang/gemma-4-12B-it-abliterix}"
SAVE_DIR="${SAVE_DIR:-/workspace/merged_gemma4_12b}"
MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
KL_CEILING="${KL_CEILING:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DRY_RUN="${DRY_RUN:-0}"
TRIAL="${TRIAL:-}"

cd "$REPO_DIR"

# ─── 1. .env + HF token ──────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Need HF_TOKEN with write access."
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2. Checkpoint dir present ───────────────────────────────────────────────
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "ERROR: $CHECKPOINT_DIR missing. Did the run complete? Run deploy_gemma4_12b.sh first."
  exit 1
fi

# ─── 3. Auto-select best trial (unless user pinned TRIAL=N) ──────────────────
if [ -z "$TRIAL" ]; then
  echo "=== Selecting best trial from $CHECKPOINT_DIR (KL ≤ $KL_CEILING) ==="
  TRIAL=$(AX_CONFIG="$CONFIG" python3 - <<PY
import os, sys
from abliterix.scriptlib import setup_io
from abliterix.util import slugify_model_name

setup_io()

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage

ckpt = "$CHECKPOINT_DIR"
model = "$MODEL_ID"
kl_ceiling = float("$KL_CEILING")

slug = slugify_model_name(model)
journal = os.path.join(ckpt, f"{slug}.jsonl")
if not os.path.exists(journal):
    sys.stderr.write(f"journal not found: {journal}\n")
    sys.exit(2)

study = optuna.load_study(
    study_name="abliterix",
    storage=JournalStorage(JournalFileBackend(journal)),
)

completed = [t for t in study.trials
             if t.user_attrs.get("refusals") is not None
             and t.user_attrs.get("kl_divergence") is not None]
if not completed:
    sys.stderr.write("no completed trials\n"); sys.exit(3)

eligible = [t for t in completed if t.user_attrs["kl_divergence"] <= kl_ceiling]
pool = eligible if eligible else completed
# Primary: min refusals. Tie-break: min KL.
best = min(pool, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))
idx = best.user_attrs.get("index", best.number)

sys.stderr.write(
    f"Selected trial #{idx}: refusals={best.user_attrs['refusals']}, "
    f"KL={best.user_attrs['kl_divergence']:.4f}"
    f"{' (KL > ceiling — ceiling relaxed)' if not eligible else ''}\n"
)
sys.stderr.write(f"Pool size: {len(pool)}/{len(completed)} under KL≤{kl_ceiling}\n")
print(idx)
PY
)
  if [ -z "$TRIAL" ]; then
    echo "ERROR: trial auto-selection failed"
    exit 1
  fi
  echo "Auto-selected TRIAL=$TRIAL"
else
  echo "Using user-pinned TRIAL=$TRIAL"
fi

# ─── 4. Plan summary ─────────────────────────────────────────────────────────
echo
echo "=== Upload plan ==="
echo "Base model      : $MODEL_ID"
echo "Config          : $CONFIG"
echo "Checkpoint dir  : $CHECKPOINT_DIR"
echo "Trial           : $TRIAL"
echo "Save dir        : $SAVE_DIR (~24 GB BF16 shards)"
echo "Target repo     : $REPO_ID"
echo "Batch size      : $BATCH_SIZE"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1 — exiting before upload."
  exit 0
fi

# ─── 5. Disk guard (need ≥40 GB free on $SAVE_DIR volume) ────────────────────
SAVE_VOL_FREE_GB=$(df -BG --output=avail "$(dirname "$SAVE_DIR")" | tail -1 | tr -d 'G ')
if [ "$SAVE_VOL_FREE_GB" -lt 40 ]; then
  echo "ERROR: < 40 GB free on $(dirname "$SAVE_DIR") — merged BF16 model is ~24 GB + scratch."
  exit 1
fi
mkdir -p "$SAVE_DIR"

# ─── 6. Run export + upload ──────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

UPLOAD_LOG="${UPLOAD_LOG:-/workspace/upload_gemma4_12b.log}"

echo "=== Running scripts/upload_model.py ==="
echo "Log: $UPLOAD_LOG"
echo

AX_CONFIG="$CONFIG" python3 scripts/upload_model.py \
  --model "$MODEL_ID" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --trial "$TRIAL" \
  --repo-id "$REPO_ID" \
  --config "$CONFIG" \
  --save-dir "$SAVE_DIR" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$UPLOAD_LOG"

echo
echo "=== Upload complete ==="
echo "Model page: https://huggingface.co/$REPO_ID"
echo
echo "Next steps:"
echo "  1. Smoke-test with 15 batched harmful prompts against the new repo"
echo "     (feedback_validation_minimal — never run full eval just to confirm upload)."
echo "  2. The reproduce/ artifacts + 'reproducible' tag are pushed automatically;"
echo "     verify reproduce.json + SHA256SUMS landed on the repo."
