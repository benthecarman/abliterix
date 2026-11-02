#!/usr/bin/env bash
# Deploy Gemma-4-12B-it abliteration — single GPU (≥40 GB; H100 80 GB ideal).
#
# Corrected "un-over-engineered" recipe. Earlier campaigns wrongly concluded
# this SKU was non-linearly entangled and capped at 27/100 @ KL 0.88. Four
# independent groups have since published WORKING linear abliterations of this
# exact checkpoint (huihui-ai, zaakirio Heretic 23/100 @ KL 0.043, OpenYourMind,
# OBLITERATUS). Root cause of our failure was OVER-ENGINEERING. See
# configs/gemma4_12b.toml for the full rationale. The fix in one line:
#   single mean direction + linear mid-stack tent + asymmetric o_proj/down_proj
#   ranges + weight_normalization="full" + projected OFF. Nothing exotic.
#
# Backend MUST be HF: gemma4_unified ForConditionalGeneration trips vLLM Punica
# on the visual.* modules. Dense model → no expert/router config.
#
# Prereqs on the pod:
#   - 1× GPU ≥ 40 GB (BF16 weights ~24 GB). H100 80 GB autotunes batch → 64.
#   - ≥ 120 GB disk on /workspace (24 GB weights + HF cache + checkpoints + merge).
#   - Repo synced to /workspace/abliterix INCLUDING datasets/ (gitignored).
#   - .env with HF_TOKEN + OPENROUTER_API_KEY.
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_gemma4_12b.sh
#
# Expected runtime:
#   Download : ~4 min @ 100 MB/s (24 GB BF16)
#   Phase 1  : hidden-state extraction over 800 prompts — ~5-8 min
#   Phase 2  : 60 trials × ~2-3 min = ~2.5-3 h on H100 / RTX Pro 6000
#   Total    : ~3 h once weights are local. Cost ~$5 @ RunPod $1.6/h.
#
# FALLBACK if the abliterix run stalls or under-performs: zaakirio's proven
# Heretic trial reproduces directly with `pip install heretic-llm` against the
# same base checkpoint.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/gemma4_12b.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_gemma4_12b.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_gemma4_12b}"
MIN_GPUS="${MIN_GPUS:-1}"
# 24 GB weights need a ≥40 GB card. H100 80 GB → 80000. RTX Pro 6000 96 GB → 90000.
MIN_VRAM_MIB="${MIN_VRAM_MIB:-40000}"
MIN_HF_SPEED="${MIN_HF_SPEED:-30}"
MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--google--gemma-4-12B-it}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"

mkdir -p /workspace
cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: expected >= ${MIN_GPUS} GPUs, found $GPU_COUNT"
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' -v min="$MIN_VRAM_MIB" '$2+0 < min {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "ERROR: GPU has < ${MIN_VRAM_MIB} MiB VRAM. BF16 weights need ~24 GB; use a ≥40 GB card."
  echo "       On a smaller card, lower max_batch_size and max_memory in the TOML."
  exit 1
fi
GPU_NAME=$(echo "$GPU_INFO" | head -1 | awk -F',' '{print $1}' | xargs)
echo "GPU: $GPU_NAME"

# ─── 2. .env check ───────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Needed keys: HF_TOKEN, OPENROUTER_API_KEY"
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env (config requires llm_judge)}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2b. Dataset check ───────────────────────────────────────────────────────
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds — re-sync the repo WITHOUT excluding datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

# ─── 3. Network speed check ──────────────────────────────────────────────────
echo "=== HF download speed test ==="
# gemma-4-12B-it ships ONE ~22 GB model.safetensors (no shard index).
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  -r 0-67108863 \
  "https://huggingface.co/${MODEL_ID}/resolve/main/model.safetensors" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s (single-stream sample — multi-worker is 8-20× faster)"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "WARN: HF single-stream speed ${SPEED_MB} MB/s below MIN_HF_SPEED=${MIN_HF_SPEED}."
  echo "      Real download uses 16 workers; usually fine. Continuing."
fi

# ─── 4. Disk space check ─────────────────────────────────────────────────────
echo "=== Disk check ==="
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$FS_MOUNT" = "/" ]; then
  echo "NOTE: /workspace is on container root (no dedicated volume) — wiped on pod destruction."
  echo "      Push the abliterated model to HF Hub before tearing down."
fi
# 12B end-to-end peak: ~24 GB model cache + ~24 GB merged shards at upload = ~48 GB.
# 50 GB floor leaves headroom on a 64 GB no-volume pod.
MIN_DISK_GB="${MIN_DISK_GB:-50}"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free. Need 24 GB model + cache + merge scratch."
  exit 1
fi

# ─── 5. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# Gemma-4-12B config.json ships transformers_version=5.10.0.dev0 and
# model_type=gemma4_unified — only loadable on transformers 5.10.x.
# kernels: 5.10.1 requires <0.13; the old `~=0.11` pin pulls 0.15 → transformers
# hub_kernels.LayerRepository raises "Either a revision or a version must be
# specified". Pin >=0.12,<0.13 (gets 0.12.3).
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=5.10,<5.11" \
  "peft>=0.18" \
  "huggingface-hub>=1.6" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  datasets \
  bitsandbytes \
  "kernels>=0.12,<0.13" \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  rich

# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, accelerate, peft, optuna; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} peft={peft.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5b. CUDA smoke test ─────────────────────────────────────────────────────
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — driver/toolchain mismatch.")
try:
    x = torch.randn(16, 16, device="cuda:0")
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"ERROR: CUDA kernel smoke-test failed: {e}")
cap = torch.cuda.get_device_capability(0)
print(f"CUDA smoke-test OK on {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
PY

# ─── 6. HF cache + pre-download ──────────────────────────────────────────────
mkdir -p "$HF_CACHE" "$CHECKPOINT_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  echo "=== Pre-downloading ${MODEL_ID} to ${HF_CACHE} (hf_transfer, 16 workers) ==="
  hf download "$MODEL_ID" \
    --repo-type model \
    --max-workers 16 \
    --quiet || {
      echo "ERROR: hf download failed. Check HF_TOKEN (model is gated — accept the license)"
      echo "       and network, then re-run."
      exit 1
    }
  echo "Download complete. Cache size:"
  du -sh "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME" || true
fi

# ─── 7. Env exports for the run ──────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ─── 8. Launch abliterix ─────────────────────────────────────────────────────
echo "=== Launching abliterix (Gemma-4-12B-it) ==="
echo "Config:         $CONFIG"
echo "Log:            $LOG_FILE"
echo "HF cache:       $HF_CACHE"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo

# Single-writer guard: never run two abliterix procs on one Optuna journal.
# Match the actual optimization invocation, NOT the repo path /workspace/abliterix
# (a bare `pgrep -f abliterix` false-positives on this script's own path).
if pgrep -f "abliterix --optimization" >/dev/null 2>&1; then
  echo "ERROR: an abliterix optimization run is already active. pkill -9 -f 'abliterix --optimization'"
  echo "       and verify zero procs before relaunching (multi-writer corrupts the Pareto front)."
  exit 1
fi

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "Ship-candidate signals (published winner: 23/100 @ KL 0.043):"
echo "  - Trial KL ≤ 0.05 with refusals dropping toward ≤ 15/100"
echo "  - mlp.down_proj.max_weight in [1.0, 1.7], nearly-flat tent (min_frac ~0.9)"
echo "  - attn.o_proj.max_weight in [0.4, 1.1], moderate taper"
echo
echo "Hard cutoffs — reconsider if:"
echo "  - Trial 18 (end of warmup) best refusals still ≥ 90/100 at KL > 0.1"
echo "    → the corrected recipe is NOT moving refusals; fall back to heretic-llm"
echo "      (pip install heretic-llm) to reproduce zaakirio's proven trial directly."
