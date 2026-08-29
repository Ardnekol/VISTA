#!/bin/bash
#SBATCH --job-name=vista_train
#SBATCH --output=slurm_logs/vista_train_%j.out
#SBATCH --error=slurm_logs/vista_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Create log directory
mkdir -p slurm_logs

# Load modules (adjust these to match your cluster)
# module load python/3.12
# module load cuda/12.1

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Install dependencies (only needed first time)
# pip install -r requirements.txt

echo "============================================"
echo "VISTA Fine-Tuning on SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Run training
python3 -m stage1_value_inference.train

echo "============================================"
echo "Training complete!"
echo "============================================"

# After training, run the simulation with the fine-tuned model
echo "Running simulation with fine-tuned model..."
python3 -m simulation.run_simulation

echo "Done! Check outputs/ for results."
