# VISTA Paper Analysis Steps

Complete, isolated workflow for publishing the VISTA paper. Each step is self-contained with its own scripts and results folder.

## Structure

```
paper_steps/
├── next_todo                    ← Full plan (8 steps)
├── README.md                    ← This file
│
├── step1_scripts/               ← STEP 1: Noise floor (paraphrase robustness)
│   ├── step1_extract_baseline_sample.py
│   ├── step1_paraphrase_generator.py
│   └── step1_rerun_paraphrases.py
├── step1_results/               ← Outputs: baseline_sample_*.json, paraphrases_*.json, noise_results_*.json, noise_summary_*.txt
```

## Quick Start: STEP 1

**Run these three commands in order:**

```bash
cd ~/VISTA/paper_steps/step1_scripts

# 1. Extract 100 baseline rows (fast)
python3 step1_extract_baseline_sample.py --n 100 --models qwen llama

# 2. Generate paraphrases (fast)
python3 step1_paraphrase_generator.py --models qwen llama

# 3. Re-run through LLMs (slow, ~1 hour)
# Inside a GPU tmux session:
tmux new -s step1
srun -p cse-gpu-all --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=2:00:00 --pty bash
cd ~/VISTA/paper_steps/step1_scripts
python3 step1_rerun_paraphrases.py --models qwen llama
```

**Results appear in:** `../step1_results/`
- `noise_results_qwen.json` — per-row flip info
- `noise_results_llama.json`
- `noise_summary_qwen.txt` — **MAIN OUTPUT: noise flip rate %**
- `noise_summary_llama.txt`

## Paper Claim Strategy

For this run, you only need to complete **Step 1: Establish a Noise Floor (Paraphrase/Seed Test)**.

- Defends against the #1 reviewer objection: "flip rates are just noise"
- Quantifies how often LLMs flip decisions due to paraphrasing alone
- If the noise floor is low, observed modifier effects are real; if not, the claim is weakened

See `next_todo` for the full details of Step 1 and the overall plan.

## File Naming Convention

All outputs follow: `{metric}_{model}.{ext}`

Examples:
- `baseline_sample_qwen.json` — STEP 1 input
- `paraphrases_llama.json` — STEP 1 input
- `noise_results_qwen.json` — STEP 1 output
- `noise_summary_qwen.txt` — STEP 1 summary

This keeps outputs isolated per model, easy to scan.

## Data Flow

```
VISTA/outputs/
  ├── master_llm_decisions_qwen.csv  (input)
  └── master_llm_decisions_llama.csv

↓ (via step1_scripts/)

paper_steps/step1_results/
  ├── baseline_sample_qwen.json
  ├── baseline_sample_llama.json
  ├── paraphrases_qwen.json
  ├── paraphrases_llama.json
  ├── noise_results_qwen.json
  └── noise_summary_qwen.txt  ← Paper claim: "modifier flips 5x above noise"
```

## Resumability

All scripts are **resume-safe**. If a run crashes:
- Already-computed rows are skipped (results files checked at start)
- Just re-run the same command; it picks up where it left off

Example: `step1_rerun_paraphrases.py` saves intermediate results every 50 rows.

## Next

Read `next_todo` for the full Step 1 instructions:

```bash
cat next_todo | head -50  # Read STEP 1 instructions
```

---

**Keep focus on Step 1.** Only the step1_scripts/ and step1_results/ folders are relevant for this run. ✅
