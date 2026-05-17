#!/bin/bash
# Run all 10 VISTA batches sequentially on the least-loaded GPUs.
# Model loads ONCE; each batch's CSV is saved when that batch finishes.
# Safe to re-run — completed batches (CSV exists) are skipped.
#
# Usage:
#   bash run_all_sequential.sh              # auto-pick 4 least-loaded GPUs
#   N_GPUS=2 bash run_all_sequential.sh     # use 2 GPUs (need a smaller model or AWQ)
#   bash run_all_sequential.sh --start 31to40
#   bash run_all_sequential.sh --only 1to10,11to20

set -euo pipefail
cd "$(dirname "$0")"

N_GPUS="${N_GPUS:-4}"

# 1. Pick the least-loaded GPUs and pin to them
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    PICKED="$(bash pick_gpus.sh "$N_GPUS")"
    export CUDA_VISIBLE_DEVICES="$PICKED"
fi
export TP_SIZE="$N_GPUS"

# 2. Local HF cache (shared across runs)
export HF_HOME="$PWD/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. vLLM knobs
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# 4. Activate venv
if [[ -d qwen_venv ]]; then
    # shellcheck disable=SC1091
    source qwen_venv/bin/activate
fi

mkdir -p outputs slurm_logs

LOG="slurm_logs/run_all_$(date +%Y%m%d_%H%M%S).log"

echo "============================================" | tee "$LOG"
echo " VISTA Qwen2.5-32B sequential run"           | tee -a "$LOG"
echo " Started:              $(date)"              | tee -a "$LOG"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"| tee -a "$LOG"
echo " TP_SIZE:              $TP_SIZE"             | tee -a "$LOG"
echo " GPU_MEM_UTIL:         $GPU_MEM_UTIL"        | tee -a "$LOG"
echo " Log file:             $LOG"                 | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv 2>/dev/null | tee -a "$LOG" || true
echo "============================================" | tee -a "$LOG"

python3 -u run_all_batches.py "$@" 2>&1 | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"
echo " Finished: $(date)"                          | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
