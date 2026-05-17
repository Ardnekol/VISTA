#!/bin/bash
# Submit one SLURM job per profile batch.
set -euo pipefail

cd "$(dirname "$0")"

BATCHES=(
    "1to10"
    "11to20"
    "21to30"
    "31to40"
    "41to50"
    "51to60"
    "61to70"
    "71to80"
    "81to90"
    "91to100"
)

for b in "${BATCHES[@]}"; do
    profiles="Dataset/profile_description_${b}.json"
    job_name="vista_qwen_${b}"
    out_csv="outputs/llm_decision_analysis_${b}_qwen.csv"

    if [[ -f "$out_csv" ]]; then
        echo "[skip] $out_csv already exists"
        continue
    fi

    echo "[submit] $job_name  ($profiles)"
    sbatch --job-name="$job_name" run_vista_qwen.sh "$profiles"
done

echo
echo "Done. Watch with:  squeue -u $USER"
