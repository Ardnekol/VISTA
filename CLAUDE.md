# VISTA — Value-Informed Situated Tactical Agent

## Project Overview
VISTA is a two-stage reasoning framework that decouples **value inference** from **action selection** to prove that different personal value sets (P) generate different actions (A) in the same scenario (C).

- **Stage 1**: DeBERTa-v3-large multi-label classifier on ValuesML Touché24 data → 38-dim value vector
- **Stage 2**: Utility-based action selector: U(a,P) = Σ(V_a,i · P_i), A = argmax_a U(a,P)

## File Ownership Rules (for Agent Teams)
| Teammate | Owns | Must NOT Touch |
|---|---|---|
| **Teammate 1 (Data/Encoder)** | `stage1_value_inference/` | `stage2_action_selection/`, `simulation/` |
| **Teammate 2 (Math/Logic)** | `stage2_action_selection/` | `stage1_value_inference/`, `simulation/` |
| **Teammate 3 (Simulation/Proof)** | `simulation/`, `outputs/` | `stage1_value_inference/`, `stage2_action_selection/` |

Shared read-only files: `config.py`, `Touché24-ValueEval/`

## Code Style
- Python 3.12+, type hints on all function signatures
- Imports: stdlib → third-party → local (separated by blank lines)
- All constants come from `config.py`
- Docstrings in Google style

## Key Data Files
- Training data: `Touché24-ValueEval/valueeval24/training-english/sentences.tsv` + `labels.tsv`
- Value taxonomy: `Touché24-ValueEval/value-categories.json`
- Moral Stories: load via `datasets.load_dataset("demelin/moral_stories")`

## Testing
- Each module must have a `if __name__ == "__main__":` block with a quick smoke test
- Utility function must be tested with known persona vectors
