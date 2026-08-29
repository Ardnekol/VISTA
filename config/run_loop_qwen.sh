#!/bin/bash
# Bash for-loop runner: one python3 invocation per batch (your previous style).
# Model reloads each iteration (vLLM ~5-10 min/load) — accepted tradeoff.
# Already-completed batches are skipped so you can re-run after a crash.

set -euo pipefail
cd "$(dirname "$0")"

source qwen_venv/bin/activate

export HF_HOME="$PWD/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TP_SIZE="${TP_SIZE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Pick least-loaded GPUs only if not already set (e.g. by SLURM)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(bash pick_gpus.sh "$TP_SIZE")"
fi

mkdir -p outputs slurm_logs

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "TP_SIZE              = $TP_SIZE"

for f in \
    Dataset/profile_description_1to10.json \
    Dataset/profile_description_11to20.json \
    Dataset/profile_description_21to30.json \
    Dataset/profile_description_31to40.json \
    Dataset/profile_description_41to50.json \
    Dataset/profile_description_51to60.json \
    Dataset/profile_description_61to70.json \
    Dataset/profile_description_71to80.json \
    Dataset/profile_description_81to90.json \
    Dataset/profile_description_91to100.json
do
    range="$(basename "$f" .json | sed 's/profile_description_//')"
    out="outputs/llm_decision_analysis_${range}_qwen.csv"

    if [[ -f "$out" ]]; then
        echo "==============================="
        echo "SKIP (already done): $out"
        echo "==============================="
        continue
    fi

    echo "==============================="
    echo "Starting: $f       $(date)"
    echo "==============================="
    python3 -u llm_decision_analysis_qwen.py --profiles "$f" --suffix qwen
done

echo "ALL QWEN BATCHES COMPLETE   $(date)"
