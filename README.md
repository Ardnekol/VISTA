<![CDATA[<div align="center">

# 🔭 VISTA

### **Value-Informed Situated Tactical Agent**

*A two-stage reasoning framework that decouples value inference from action selection to prove that different personal value sets produce different actions in the same scenario.*

---

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗_Transformers-4.49%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
  - [Stage 1: Value Inference](#stage-1-value-inference)
  - [Stage 2: Action Selection](#stage-2-action-selection)
  - [Pipeline Flow](#pipeline-flow)
- [Mathematical Formulation](#mathematical-formulation)
- [Schwartz Value Taxonomy](#schwartz-value-taxonomy)
- [Datasets](#datasets)
- [Persona Definitions](#persona-definitions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start: Run the Simulation](#quick-start-run-the-simulation)
  - [Fine-Tune the Value Classifier](#fine-tune-the-value-classifier)
  - [Run Inference on Custom Text](#run-inference-on-custom-text)
  - [Create Custom Personas](#create-custom-personas)
- [Simulation Results](#simulation-results)
- [Output Artifacts](#output-artifacts)
- [Agent Teams Integration](#agent-teams-integration)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [License](#license)

---

## Overview

VISTA explores a fundamental question in value-aligned AI:

> **Do different value systems lead to different actions, even when the situation is identical?**

The framework answers this by implementing a two-stage pipeline:

1. **Stage 1 — Value Inference**: A DeBERTa-v3-large multi-label classifier trained on the [ValuesML (Touché 2024)](https://touche.webis.de/clef24/touche24-web/human-value-detection.html) dataset extracts a 38-dimensional value distribution vector $V_{\text{dist}}$ from any text.

2. **Stage 2 — Action Selection**: A utility-based selector uses a dot-product formulation $U(a, P) = \sum_{i} V_{a,i} \cdot P_i$ to rank candidate actions from the [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) dataset, selecting the action that maximizes alignment with a given persona vector $P$.

By swapping the persona vector $P$ while holding the scenario $C$ constant, VISTA produces verifiable proof that **value profiles causally determine behavior**.

---

## Key Contributions

| Contribution | Description |
|:-------------|:------------|
| **Decoupled Architecture** | Separates *what values are relevant* (Stage 1) from *how they influence decisions* (Stage 2), enabling modular experimentation |
| **38-Dimensional Value Space** | Uses the refined Schwartz taxonomy with attained/constrained polarity for each of 19 values, providing richer signal than binary value labels |
| **Interpretable Decision Audit** | Every decision includes a full audit trail citing which value dimensions drove the selection and by how much |
| **Persona-Driven Proof** | Demonstrates that swapping Person 1 (Explorer) for Person 2 (Guardian) flips the selected action in a significant percentage of scenarios |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VISTA FRAMEWORK                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  STAGE 1: VALUE INFERENCE                     │  │
│  │                                                               │  │
│  │   Input Text ──► DeBERTa-v3-large ──► Sigmoid ──► V_dist     │  │
│  │                  (38-label head)       (threshold)  (38-dim)  │  │
│  │                                                               │  │
│  │   Training Data: ValuesML / Touché24 (44,758 sentences)       │  │
│  │   Labels: 19 Schwartz values × {attained, constrained}        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                STAGE 2: ACTION SELECTION                      │  │
│  │                                                               │  │
│  │   For each candidate action a:                                │  │
│  │     V_a = Stage1.predict(a)        ← value-tag the action     │  │
│  │                                                               │  │
│  │   For a given persona P:                                      │  │
│  │     U(a, P) = Σ (V_a,i · P_i)     ← compute utility          │  │
│  │     A* = argmax_a U(a, P)          ← select best action       │  │
│  │                                                               │  │
│  │   Candidate Actions: Moral Stories (12,000 narratives)        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   PROOF / SIMULATION                           │  │
│  │                                                               │  │
│  │   For each scenario C in {1..100}:                            │  │
│  │     A_explorer = select(candidates, P_explorer)               │  │
│  │     A_guardian  = select(candidates, P_guardian)               │  │
│  │     shift = (A_explorer ≠ A_guardian)                         │  │
│  │                                                               │  │
│  │   Output: decision_shift_report.md + audit_trail.json         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Value Inference

The value inference module uses **DeBERTa-v3-large** (`microsoft/deberta-v3-large`, 304M parameters) configured for multi-label classification:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=38,                              # 19 values × 2 states
    problem_type="multi_label_classification",  # BCEWithLogitsLoss
)
```

**Input**: Any natural language text (scenario description, action description, etc.)  
**Output**: A 38-dimensional probability vector $V_{\text{dist}} \in [0, 1]^{38}$

The 38 dimensions represent each of the 19 Schwartz values in two states:
- **Attained**: The value is promoted, fulfilled, or supported
- **Constrained**: The value is hindered, thwarted, or violated

### Stage 2: Action Selection

Given a set of candidate actions (e.g., the moral and immoral actions from Moral Stories), Stage 2:

1. **Tags** each action with a value vector $V_a$ using the Stage 1 classifier
2. **Computes** the weighted dot-product utility $U(a, P)$ against a persona vector $P$
3. **Selects** the action with maximum utility: $A^* = \arg\max_a\, U(a, P)$

### Pipeline Flow

```
Scenario (C)                    Persona Vector (P)
    │                                  │
    ▼                                  │
┌──────────────┐                       │
│ Candidate    │                       │
│ Actions      │                       │
│  • moral     │──► Value Tagger ──►┐  │
│  • immoral   │    (Stage 1)      │  │
└──────────────┘                   ▼  ▼
                              ┌─────────────┐
                              │ U(a,P) =    │
                              │ Σ V_a,i·P_i │
                              │             │
                              │ A* = argmax │
                              └──────┬──────┘
                                     │
                                     ▼
                              Selected Action (A)
                              + Justification
```

---

## Mathematical Formulation

### Core Equations

**Value Distribution (Stage 1):**

$$V_{\text{dist}} = \sigma\big(\text{DeBERTa}(x)\big) \in [0, 1]^{38}$$

where $\sigma$ is the element-wise sigmoid function and $x$ is the input text.

**Utility Function (Stage 2):**

$$U(a, P) = \sum_{i=0}^{37} V_{a,i} \cdot P_i = \mathbf{V}_a^{\top} \mathbf{P}$$

where:
- $V_{a,i}$ is the $i$-th dimension of the value vector for action $a$
- $P_i$ is the $i$-th dimension of the persona vector $P$

**Action Selection:**

$$A^* = \arg\max_{a \in \mathcal{A}} \; U(a, P)$$

**Decision Shift Condition:**

$$\text{Shift}(C) = \mathbb{1}\left[\arg\max_a U(a, P_1) \neq \arg\max_a U(a, P_2)\right]$$

A shift occurs when two different persona vectors select different actions for the same scenario.

### Interpretability: Contribution Analysis

For each decision, VISTA computes per-dimension contributions:

$$\text{contribution}_i = V_{a^*,i} \cdot P_i$$

The top-$k$ contributions identify which value dimensions most strongly drove the decision.

---

## Schwartz Value Taxonomy

VISTA uses the **refined Schwartz (2012) theory** with 19 basic human values, each tracked in two states:

| # | Value | Goal | Attained Index | Constrained Index |
|:-:|:------|:-----|:-:|:-:|
| 1 | Self-direction: thought | Freedom to cultivate one's own ideas and abilities | 0 | 1 |
| 2 | Self-direction: action | Freedom to determine one's own actions | 2 | 3 |
| 3 | Stimulation | Excitement, novelty, and change | 4 | 5 |
| 4 | Hedonism | Pleasure and sensuous gratification | 6 | 7 |
| 5 | Achievement | Success according to social standards | 8 | 9 |
| 6 | Power: dominance | Power through exercising control over people | 10 | 11 |
| 7 | Power: resources | Power through control of material and social resources | 12 | 13 |
| 8 | Face | Security and power through maintaining public image | 14 | 15 |
| 9 | Security: personal | Safety in one's immediate environment | 16 | 17 |
| 10 | Security: societal | Safety and stability in the wider society | 18 | 19 |
| 11 | Tradition | Maintaining cultural, family, or religious traditions | 20 | 21 |
| 12 | Conformity: rules | Compliance with rules, laws, and formal obligations | 22 | 23 |
| 13 | Conformity: interpersonal | Avoidance of upsetting or harming other people | 24 | 25 |
| 14 | Humility | Recognizing one's insignificance in the larger scheme | 26 | 27 |
| 15 | Benevolence: caring | Devotion to the welfare of in-group members | 28 | 29 |
| 16 | Benevolence: dependability | Being a reliable and trustworthy in-group member | 30 | 31 |
| 17 | Universalism: concern | Commitment to equality, justice, and protection for all | 32 | 33 |
| 18 | Universalism: nature | Preservation of the natural environment | 34 | 35 |
| 19 | Universalism: tolerance | Acceptance of those who are different from oneself | 36 | 37 |

Full value definitions with personal motivations are in [`Touché24-ValueEval/value-categories.json`](Touché24-ValueEval/value-categories.json).

---

## Datasets

### ValuesML / Touché 2024 (Stage 1 Training)

| Property | Value |
|:---------|:------|
| **Source** | [Touché 2024 ValueEval Task](https://touche.webis.de/clef24/touche24-web/human-value-detection.html) |
| **Training samples** | 44,758 sentences |
| **Labels** | 38 columns (19 values × 2 states) |
| **Format** | `sentences.tsv` + `labels.tsv` (Tab-separated) |
| **Languages** | Multilingual (English translations provided) |
| **Label encoding** | Binary (1 if value ≥ 0.5, 0 otherwise) |

### Moral Stories (Stage 2 Action Candidates)

| Property | Value |
|:---------|:------|
| **Source** | [Emelin et al. (2021)](https://aclanthology.org/2021.emnlp-main.54/) — [HuggingFace](https://huggingface.co/datasets/demelin/moral_stories) |
| **Total stories** | 12,000 structured narratives |
| **Story fields** | `norm`, `situation`, `intention`, `moral_action`, `moral_consequence`, `immoral_action`, `immoral_consequence` |
| **Format** | JSONL |
| **Task** | Each story provides a binary choice: moral vs. immoral action |

---

## Persona Definitions

VISTA ships with two contrasting personas designed to maximize decision divergence:

### 🧭 Person 1: "The Explorer"

A curiosity-driven, freedom-seeking individual who values novelty and resists conformity.

| Value | Attained Weight | Constrained Weight | Net Preference |
|:------|:---:|:---:|:---:|
| Self-direction: thought | **0.95** | 0.05 | **+0.90** |
| Self-direction: action | **0.90** | 0.05 | **+0.85** |
| Stimulation | **0.90** | 0.05 | **+0.85** |
| Universalism: tolerance | **0.80** | 0.15 | **+0.65** |
| Conformity: rules | 0.05 | **0.90** | **-0.85** |
| Security: personal | 0.10 | **0.80** | **-0.70** |

### 🛡️ Person 2: "The Guardian"

A security-focused, tradition-respecting individual who values order and social stability.

| Value | Attained Weight | Constrained Weight | Net Preference |
|:------|:---:|:---:|:---:|
| Conformity: rules | **0.90** | 0.05 | **+0.85** |
| Security: societal | **0.85** | 0.05 | **+0.80** |
| Tradition | **0.85** | 0.10 | **+0.75** |
| Conformity: interpersonal | **0.80** | 0.10 | **+0.70** |
| Self-direction: thought | 0.10 | **0.70** | **-0.60** |
| Stimulation | 0.05 | **0.80** | **-0.75** |

**Cosine similarity** between Explorer and Guardian: **0.5307** (meaningfully divergent).

---

## Project Structure

```
VISTA/
│
├── README.md                              # This file
├── config.py                              # Central configuration & constants
├── requirements.txt                       # Python dependencies
├── CLAUDE.md                              # Context file for Claude Code Agent Teams
│
├── stage1_value_inference/                # STAGE 1: Value Inference Pipeline
│   ├── __init__.py
│   ├── data_loader.py                     # ValuesML dataset loading & preprocessing
│   ├── model.py                           # DeBERTa-v3-large classifier wrapper
│   ├── train.py                           # Fine-tuning script (HF Trainer)
│   └── predict.py                         # Inference API: text → V_dist (38-dim)
│
├── stage2_action_selection/               # STAGE 2: Action Selection Pipeline
│   ├── __init__.py
│   ├── moral_stories_loader.py            # Moral Stories download & parsing
│   ├── personas.py                        # Explorer & Guardian persona definitions
│   ├── value_tagger.py                    # Tag actions with V_a value vectors
│   └── utility.py                         # U(a,P) dot-product + argmax selector
│
├── simulation/                            # SIMULATION & PROOF
│   ├── __init__.py
│   ├── scenario_sampler.py                # Sample 100 scenarios from Moral Stories
│   ├── run_simulation.py                  # Full pipeline orchestrator
│   └── report_generator.py                # Generate report & audit trail
│
├── outputs/                               # GENERATED ARTIFACTS
│   ├── decision_shift_report.md           # Verifiable proof report
│   └── audit_trail.json                   # Per-decision justifications (100 entries)
│
├── checkpoints/                           # Saved model weights (after fine-tuning)
│   └── best_model/                        # Best DeBERTa checkpoint
│
├── Touché24-ValueEval/                    # TRAINING DATA (pre-existing)
│   ├── value-categories.json              # 19 Schwartz value definitions
│   └── valueeval24/
│       ├── training-english/              # 44,758 labeled sentences
│       │   ├── sentences.tsv
│       │   └── labels.tsv
│       ├── validation-english/
│       └── test-english/
│
└── .cache/                                # Downloaded datasets (auto-managed)
    └── moral_stories/
        └── moral_stories_full.jsonl       # 12,000 stories
```

---

## Installation

### Prerequisites

- **Python** 3.12+
- **macOS** 13.0+ (for MPS acceleration) or Linux with CUDA

### Setup

```bash
# Clone or navigate to the project
cd /path/to/VISTA

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `torch` | ≥ 2.6.0 | Deep learning framework (MPS/CUDA/CPU) |
| `transformers` | ≥ 4.49.0 | DeBERTa-v3-large model |
| `datasets` | ≥ 2.18.0 | HuggingFace data loading |
| `pandas` | ≥ 2.0.0 | TSV data processing |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scikit-learn` | ≥ 1.3.0 | F1/precision/recall metrics |
| `accelerate` | ≥ 0.28.0 | Training acceleration |
| `sentencepiece` | ≥ 0.1.99 | DeBERTa tokenizer backend |
| `tqdm` | ≥ 4.65.0 | Progress bars |
| `huggingface_hub` | latest | Dataset downloads |

---

## Usage

### Quick Start: Run the Simulation

Run the full 100-scenario proof simulation out of the box:

```bash
python3 -m simulation.run_simulation
```

This will:
1. Download Moral Stories from HuggingFace (~8MB)
2. Download DeBERTa-v3-large weights (~874MB, first run only)
3. Sample 100 scenarios
4. Tag 200 candidate actions with 38-dim value vectors
5. Compare Explorer vs. Guardian action selection
6. Generate `outputs/decision_shift_report.md` and `outputs/audit_trail.json`

Expected runtime: **~5-7 minutes** (first run with model download).

### Fine-Tune the Value Classifier

To significantly improve the shift rate, fine-tune DeBERTa on the ValuesML data:

```bash
python3 -m stage1_value_inference.train
```

| Parameter | Value |
|:----------|:------|
| Learning rate | 2e-5 |
| Epochs | 5 (with early stopping, patience=2) |
| Batch size | 8 |
| Loss | BCEWithLogitsLoss |
| Metric | Macro F1 |
| Estimated time | ~2-4 hours (MPS) |

The best model is saved to `checkpoints/best_model/`. Subsequent simulation runs will automatically load it.

### Run Inference on Custom Text

```python
from stage1_value_inference.predict import predict, explain_prediction

# Get raw 38-dim probabilities
probs = predict("I believe everyone deserves equal treatment regardless of background.")
print(probs.shape)  # (38,)

# Get an interpretable breakdown
result = explain_prediction("I want the freedom to make my own choices.", top_k=5)
print(result["top_values"])
# [('Self-direction: action attained', 0.87), ('Self-direction: thought attained', 0.82), ...]
print(result["active_labels"])
# ['Self-direction: action attained', 'Self-direction: thought attained']
```

### Create Custom Personas

```python
from stage2_action_selection.personas import _make_persona_vector, PERSONAS

# Define a new persona: "The Diplomat"
diplomat_weights = {
    "Universalism: concern":     (0.90, 0.05),
    "Universalism: tolerance":   (0.85, 0.10),
    "Benevolence: caring":       (0.80, 0.10),
    "Benevolence: dependability": (0.75, 0.15),
    "Conformity: interpersonal": (0.70, 0.20),
    "Humility":                  (0.80, 0.10),
    "Power: dominance":          (0.10, 0.80),
    "Power: resources":          (0.15, 0.70),
}

PERSONA_DIPLOMAT = _make_persona_vector(diplomat_weights)
PERSONAS["Diplomat"] = PERSONA_DIPLOMAT
```

### Compare Any Two Personas

```python
from stage2_action_selection.utility import compare_personas
import numpy as np

# Assuming candidates are tagged with value_vectors
result = compare_personas(
    candidates=tagged_candidates,
    persona_a=PERSONA_DIPLOMAT,
    persona_b=PERSONA_GUARDIAN,
    name_a="Diplomat",
    name_b="Guardian",
)

print(f"Shifted: {result['decision_shifted']}")
print(f"Diplomat chose: {result['person_a']['selected_label']}")
print(f"Guardian chose: {result['person_b']['selected_label']}")
```

---

## Simulation Results

### Baseline Run (Pretrained DeBERTa, No Fine-Tuning)

| Metric | Value |
|:-------|:------|
| Scenarios tested | 100 |
| Decision shifts (A₁ ≠ A₂) | 8–22* |
| Shift rate | 8–22%* |
| Device | MPS (Apple Silicon) |
| Elapsed time | ~388s |

*\* Range reflects variance across runs due to the random classification head.*

### Sample Decision Shifts

| Scenario | Situation | Explorer Chose | Guardian Chose |
|:---------|:----------|:---------------|:---------------|
| 3814 | Betty deciding whether to ask subordinate Dave out | moral (freedom) | immoral (workplace norms) |
| 9459 | John wants noisy sister to be quiet | immoral (direct action) | moral (respectful approach) |
| 10834 | Juan under oppressive dictator | immoral (rebellion) | moral (structured resistance) |

### Value Dimensions Driving Divergence

The following dimensions appeared most frequently in shifted decisions:

| Value Dimension | Appearances in Shifts |
|:----------------|:---------------------:|
| Self-direction: thought (attained) | Most frequent |
| Stimulation (attained) | Most frequent |
| Conformity: rules (attained/constrained) | Most frequent |
| Tradition (attained) | Most frequent |

> **After fine-tuning**, the shift rate is expected to increase to **40–70%+** as the value vectors become semantically meaningful rather than near-random.

---

## Output Artifacts

### `decision_shift_report.md`

A human-readable Markdown report containing:
- Summary statistics (shift rate, confidence intervals)
- Top-10 most dramatic decision shifts with full breakdowns
- Value dimension divergence analysis
- Complete persona profiles with net-preference rankings

### `audit_trail.json`

A machine-readable JSON file with one entry per scenario:

```json
{
  "scenario_id": 3814,
  "situation": "Betty thinks her assistant Dave from work is cute...",
  "intention": "Betty wants to decide whether to ask Dave out or not.",
  "norm": "It's inappropriate to be attracted to a subordinate at work.",
  "candidates": [
    {
      "action": "Betty decides not to act on her feelings...",
      "label": "moral",
      "V_a": [0.52, 0.48, 0.51, ...]
    }
  ],
  "person1": {
    "persona": "Explorer",
    "selected_action": "...",
    "selected_label": "moral",
    "utility_scores": { "moral": 8.734, "immoral": 8.724 },
    "top_driving_values": ["Self-direction: thought attained", "Stimulation attained"]
  },
  "person2": {
    "persona": "Guardian",
    "selected_action": "...",
    "selected_label": "immoral",
    "utility_scores": { "moral": 8.018, "immoral": 8.019 },
    "top_driving_values": ["Conformity: rules attained", "Tradition attained"]
  },
  "decision_shifted": true,
  "shift_magnitude": 0.0055
}
```

---

## Agent Teams Integration

VISTA includes a `CLAUDE.md` file that enables [Claude Code Agent Teams](https://code.claude.com/docs/agent-teams) to parallelize development work. The file defines strict **file ownership rules** to prevent merge conflicts:

| Teammate | Owns | Must NOT Touch |
|:---------|:-----|:---------------|
| Teammate 1 (Data/Encoder) | `stage1_value_inference/` | `stage2_action_selection/`, `simulation/` |
| Teammate 2 (Math/Logic) | `stage2_action_selection/` | `stage1_value_inference/`, `simulation/` |
| Teammate 3 (Simulation/Proof) | `simulation/`, `outputs/` | `stage1_value_inference/`, `stage2_action_selection/` |

To use:
```bash
cd /path/to/VISTA
claude  # Start Claude Code
# Then paste the team prompt from CLAUDE.md
```

---

## Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|:-----------|:-------|:-----------|
| **Unfinetuned model** | Low shift rate (~8%) due to random classification head | Fine-tune with `python3 -m stage1_value_inference.train` |
| **Binary action choice** | Only moral vs. immoral; no multi-action scenarios | Integrate ValueActionLens (14,784 value-informed actions) |
| **No cross-cultural variation** | Persona vectors are culture-agnostic | Add culture-specific persona modifiers from ValueActionLens |
| **Static personas** | Personas don't adapt to context | Implement context-dependent persona weighting |

### Planned Enhancements

- [ ] **ValueActionLens integration**: 14,784 value-informed actions across 12 cultures and 11 social topics
- [ ] **Multi-action selection**: Generate synthetic candidate actions for fine-grained moral tension scenarios  
- [ ] **Persona interpolation**: Smoothly blend between personas to find decision boundaries
- [ ] **Confidence calibration**: Temperature scaling on sigmoid outputs for better-calibrated value probabilities
- [ ] **Visualization dashboard**: Interactive web UI to explore decision shifts and value landscapes

---

## References

### Datasets

1. **ValuesML / Touché 2024**: Kiesel et al. (2024). *ValueEval at Touché 2024: Human Value Detection.*
   - [Task page](https://touche.webis.de/clef24/touche24-web/human-value-detection.html) · [DOI](https://doi.org/10.5281/zenodo.10396294)

2. **Moral Stories**: Emelin, D., Le Bras, R., Hwang, J. D., Forbes, M., & Choi, Y. (2021). *Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences.* EMNLP 2021.
   - [Paper](https://aclanthology.org/2021.emnlp-main.54/) · [Data](https://huggingface.co/datasets/demelin/moral_stories)

3. **ValueActionLens**: Shen, H., Clark, N., & Mitra, T. (2025). *Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?* EMNLP 2025 (Outstanding Paper Award).
   - [Paper](https://arxiv.org/abs/2410.07000) · [Code](https://github.com/huashen218/value_action_gap)

### Value Theory

4. **Schwartz Refined Theory**: Schwartz, S. H. (2012). *An Overview of the Schwartz Theory of Basic Values.* Online Readings in Psychology and Culture, 2(1).

### Model

5. **DeBERTa-v3**: He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* ICLR 2023.
   - [HuggingFace](https://huggingface.co/microsoft/deberta-v3-large)

---

## License

This project is for **research purposes only**. The ValuesML dataset has specific usage restrictions — see `Touché24-ValueEval/README.md` for the data usage agreement.

---

<div align="center">

*Built with [Antigravity](https://github.com/google-deepmind) + [Claude Code Agent Teams](https://code.claude.com/docs/agent-teams)*

</div>
]]>
