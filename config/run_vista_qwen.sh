#!/bin/bash
#SBATCH --job-name=vista_qwen
#SBATCH --output=slurm_logs/vista_qwen_%x_%j.out
#SBATCH --error=slurm_logs/vista_qwen_%x_%j.err
#SBATCH --partition=cse-gpu-all
#SBATCH --nodelist=dgx-a100-02
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00

# Usage:
#   sbatch --job-name=vista_qwen_1to10 run_vista_qwen.sh Dataset/profile_description_1to10.json

set -euo pipefail

PROFILES_FILE="${1:-Dataset/profile_description_1to10.json}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs outputs

# Activate venv created by setup_env.sh
source "$SLURM_SUBMIT_DIR/qwen_venv/bin/activate"

# Pin the model cache to a project-local path so all jobs share weights
export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
# vLLM knobs (consumed by llm_decision_analysis_qwen.py)
export TP_SIZE=4
export GPU_MEM_UTIL=0.90
export MAX_MODEL_LEN=4096
# Quieter HF logs
export TRANSFORMERS_VERBOSITY=warning

echo "============================================"
echo " VISTA Qwen2.5-32B-Instruct"
echo " Job:        $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo " Node:       $SLURM_NODELIST"
echo " Profiles:   $PROFILES_FILE"
echo " Started:    $(date)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "============================================"

python3 -u llm_decision_analysis_qwen.py \
    --profiles "$PROFILES_FILE" \
    --suffix qwen

echo "============================================"
echo " Finished:   $(date)"
echo "============================================"
