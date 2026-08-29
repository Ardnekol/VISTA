#!/bin/bash
# Bash for-loop runner — Llama 3.1 70B Instruct AWQ-INT4 via vLLM.
# Same shape as run_loop_qwen.sh; only model + script differ.
# Skips any batch whose CSV already exists (re-run safe).

set -euo pipefail
cd "$(dirname "$0")"

source qwen_venv/bin/activate

export HF_HOME="$PWD/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Llama model + quantization (override via env if you want to switch)
export LLAMA_MODEL="${LLAMA_MODEL:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"
export LLAMA_QUANT="${LLAMA_QUANT:-awq}"

export TP_SIZE="${TP_SIZE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Pick least-loaded GPUs only if SLURM hasn't already pinned us
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(bash pick_gpus.sh "$TP_SIZE")"
fi

mkdir -p outputs slurm_logs

echo "Model                = $LLAMA_MODEL"
echo "Quantization         = $LLAMA_QUANT"
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
    out="outputs/llm_decision_analysis_${range}_llama.csv"

    if [[ -f "$out" ]]; then
        echo "==============================="
        echo "SKIP (already done): $out"
        echo "==============================="
        continue
    fi

    echo "==============================="
    echo "Starting: $f       $(date)"
    echo "==============================="
    python3 -u llm_decision_analysis_llama.py --profiles "$f" --suffix llama
done

echo "ALL LLAMA BATCHES COMPLETE   $(date)"
