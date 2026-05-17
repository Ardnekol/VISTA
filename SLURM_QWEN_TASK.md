# VISTA — SLURM Task Brief for Qwen2.5-32B-Instruct
**For: Claude agent running on SLURM cluster**
**Date: 2026-05-13**

---

## What This Project Does

VISTA is a value-based decision analysis framework. The task you need to complete is:

> Run `llm_decision_analysis.py` for all 10 profile batches using **Qwen2.5-32B-Instruct** on SLURM, producing one CSV output file per batch.

---

## File Structure (what you have)

```
VISTA/
├── llm_decision_analysis.py    ← MAIN SCRIPT — needs Qwen adaptation (see below)
├── merge_and_analyze.py        ← Run this AFTER all batches complete
├── train_slurm.sh              ← Reference SLURM config for this cluster
├── requirements.txt            ← Python dependencies
└── Dataset/
    ├── profile_description_1to10.json    ← 10 profiles
    ├── profile_description_11to20.json   ←  9 profiles
    ├── profile_description_21to30.json   ←  9 profiles
    ├── profile_description_31to40.json   ←  9 profiles
    ├── profile_description_41to50.json   ← 10 profiles
    ├── profile_description_51to60.json   ←  9 profiles
    ├── profile_description_61to70.json   ← 10 profiles
    ├── profile_description_71to80.json   ← 10 profiles
    ├── profile_description_81to90.json   ←  9 profiles
    ├── profile_description_91to100.json  ← 10 profiles
    ├── scenarios_batch1.json             ← 10 scenarios
    └── modifiers_batch1.json             ← 80 modifiers (8 per scenario)
```

---

## What the Script Does

`llm_decision_analysis.py` runs a decision simulation:

1. Loads a profile batch (e.g. `profile_description_1to10.json`)
2. For each profile × each scenario × each condition (baseline + 8 modifiers):
   - Builds a prompt: *person's value profile + scenario + optional modifier*
   - Asks the LLM: *"Which action would this person choose — A0 or A1?"*
   - Gets back: `decision`, `confidence`, `driving_values`, `reasoning` as JSON
3. Also runs dot-product scoring for comparison
4. Saves everything to a CSV in `outputs/`

**Total calls per batch:** 9 profiles × 90 = 810 (or 10 × 90 = 900)
**Total calls all batches:** 8,550

---

## The Problem to Solve

The script currently uses **Ollama** (local HTTP API):
```python
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "gemma4:31b"
```

SLURM does not have Ollama. You need to **replace the `call_llm()` function** to load Qwen2.5-32B-Instruct via **HuggingFace transformers** or **vLLM**.

---

## What You Need to Change

### Option A — HuggingFace Transformers (simpler)
Replace the `call_llm()` function to use:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-32B-Instruct"
```

### Option B — vLLM (faster, recommended for 32B)
Replace the `call_llm()` function to use vLLM's `LLM` class:
```python
from vllm import LLM, SamplingParams
model = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=<n_gpus>)
```

### Keep everything else the same:
- `build_prompt()` — no changes needed
- `parse_profile_weights()` — no changes needed
- `dotproduct_decide()` — no changes needed
- `main()` — no changes needed
- CLI args `--profiles` and `--suffix` — no changes needed

---

## Output Format Required

Each output CSV must have these exact columns:
```
vsw_id, profile_HIGH_values,
scenario_id, theme_id, scenario_brief,
A0_text, A1_text,
condition, modifier_text, pressured_values,
llm_decision, llm_confidence, llm_driving_values, llm_reasoning,
dp_score_A0, dp_score_A1, dp_decision,
llm_dp_agree,
llm_changed_from_baseline, dp_changed_from_baseline,
llm_change_explanation
```

`llm_decision` must be exactly `"A0"`, `"A1"`, or `"ERROR"`.

---

## LLM Prompt Format

The prompt sent to the model looks like this:

```
You are simulating the decision-making of a real person with the following value priorities.

PERSON'S VALUE PROFILE:
  - Self-Direction: LOW. The person does not prioritize personal autonomy.
  - Security: HIGH. The person highly values stability and predictability.
  - Conformity: HIGH. The person prioritizes following rules and norms.
  ...

Situational context (this is the environment right now):
  The department head has recommended following established procedures.

SCENARIO:
A hospital department is piloting a new patient care protocol...

CHOICES:
  A0: implement the new care protocol
  A1: maintain established procedures

Step into this person's perspective completely. Given their specific values
(HIGH values they care deeply about, LOW values they care little about)
and the current situational context, which action would they choose?

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{
  "decision": "A0" or "A1",
  "confidence": "high" or "medium" or "low",
  "driving_values": ["value1", "value2"],
  "reasoning": "one concise sentence explaining which values led to this choice"
}
```

---

## SLURM Job Requirements

Based on `train_slurm.sh` reference config:
- **GPU:** Need enough VRAM for 32B model — at least 2× A100 80GB or 4× A100 40GB
- **Time:** ~25 hours per batch (810-900 calls × ~100s each)
- **Memory:** 64GB+ RAM recommended
- **Partition:** gpu

Suggested SLURM config per batch:
```bash
#SBATCH --job-name=vista_qwen_1to10
#SBATCH --gres=gpu:2          # adjust to cluster
#SBATCH --mem=64G
#SBATCH --time=28:00:00       # 28 hours per batch
#SBATCH --cpus-per-task=8
```

---

## Commands to Run (after script is adapted)

Submit one job per batch:
```bash
python3 llm_decision_analysis.py --profiles Dataset/profile_description_1to10.json  --suffix qwen
python3 llm_decision_analysis.py --profiles Dataset/profile_description_11to20.json --suffix qwen
...
python3 llm_decision_analysis.py --profiles Dataset/profile_description_91to100.json --suffix qwen
```

Output files will be:
```
outputs/llm_decision_analysis_1to10_qwen.csv
outputs/llm_decision_analysis_11to20_qwen.csv
...
outputs/llm_decision_analysis_91to100_qwen.csv
```

After all batches complete, run:
```bash
python3 merge_and_analyze.py
```

---

## Context: Why This Experiment

This is a research project studying whether situational pressure (modifiers)
overrides personal values (profiles) in LLM-simulated decision making.

We have already run:
- Llama 3.1 8B  → 8,550 decisions (complete)
- Gemma4 31B    → in progress on separate GPU server

Qwen2.5-32B-Instruct is the third model for multi-model comparison in a
conference paper targeting EMNLP 2026 / AAAI 2027.

Key finding so far (from Llama): authority signals flip decisions most (3.3%),
strong value profiles resist all pressure (0% flip rate at 9 HIGH values).
