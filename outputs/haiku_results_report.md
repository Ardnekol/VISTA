# VISDA — Claude Haiku 4.5 Results (Closed-Model Comparison)

*Generated for inclusion in the paper. Model: `claude-haiku-4-5` (Anthropic API).
Source data: `outputs/master_llm_decisions_haiku.csv` (8,550 decisions).*

---

## 1. Setup / Method note (for §4.2)

We add one **closed-weight** frontier model, **Claude Haiku 4.5**, to the three
open-weight LLMs. It is evaluated on the identical protocol: the same 95 Schwartz
value profiles × 10 base scenarios × (1 baseline + 8 situational modifiers) =
**8,550 decisions** (950 baseline + 7,600 modifier trials), the same value-profile
descriptions and modifier texts, and the same forced binary A0/A1 choice.

Requests were served through the Anthropic **Message Batches API** with
**greedy decoding (temperature = 0)** to match the open-weight setup, and a
JSON-schema structured-output constraint to guarantee parseable responses.
Two of 8,550 requests returned a transient API error and were re-queried
individually; the final dataset has **0 missing decisions**. The dot-product
mathematical baseline computed alongside Haiku reproduced the paper's flip rate
of **19.93%** exactly, confirming the dataset and rule wiring are identical to the
open-weight runs.

---

## 2. System-level flip rate (extends Table 2)

| System | Trials | Flips | Flip rate |
|---|---:|---:|---:|
| Utility-based mathematical baseline | 7,600 | 1,515 | 19.93% |
| **Claude Haiku 4.5** | **7,600** | **823** | **10.83%** |
| Human pilot (N = 50) | 500 | — | 12.36% |
| Gemma 4 31B | 7,600 | 678 | 8.92% |
| Qwen 2.5 32B | 7,600 | 500 | 6.58% |
| LLaMA 3.1 8B | 7,600 | 154 | 2.03% |

**Haiku is the most modifier-sensitive LLM in the study**, above every open-weight
model and closest to the human pilot in magnitude.

LaTeX row for Table 2:
```latex
Claude Haiku 4.5 & 7{,}600 & 823 & 10.83\% \\
```

---

## 3. Per-axis McNemar test (extends Table 7 / §5.2)

McNemar's exact test, baseline vs modifier, per axis (n = 950 each).
`A0->A1` / `A1->A0` are directional discordant flips; OR = A0→A1 / A1→A0.
p-values Benjamini–Hochberg FDR corrected across the 8 axes.

| Axis | A0→A1 | A1→A0 | flip rate | OR | 95% CI | p (BH-FDR) | sig |
|---|---:|---:|---:|---:|---|---:|:--|
| self_preservation | 164 | 20 | 19.4% | **8.20** | [5.14, 13.78] | 1.0e-27 | *** |
| authority_signal | 106 | 53 | 16.7% | 2.00 | [1.43, 2.84] | 1.1e-04 | *** |
| resource_scarcity | 80 | 49 | 13.6% | 1.63 | [1.13, 2.38] | 1.5e-02 | * |
| competence_uncertainty | 71 | 18 | 9.4% | 3.94 | [2.33, 7.03] | 6.5e-08 | *** |
| diffused_responsibility | 24 | 57 | 8.5% | 0.42 | [0.25, 0.69] | 8.5e-04 | *** |
| social_visibility | 33 | 34 | 7.1% | 0.97 | [0.58, 1.62] | 1.00 | ns |
| in_out_group | 30 | 29 | 6.2% | 1.03 | [0.60, 1.79] | 1.00 | ns |
| time_pressure | 30 | 25 | 5.8% | 1.20 | [0.68, 2.13] | 0.74 | ns |

**5 of 8 axes are significant** at BH-FDR < 0.05 (4 at < 0.001). Haiku's two
most potent axes — **self_preservation and authority_signal** — are exactly the
two axes the paper identifies as dominating LLM decisions, and its
`self_preservation` odds ratio (8.20) is the strongest single-axis effect
observed for any model. `diffused_responsibility` acts significantly in the
**reverse** direction (OR 0.42), as it does for the open models.

---

## 4. Modifier-type pressure (extends Table 3 / §5.4)

Mean decision shift by the four pressure categories.

| Type (axes) | Human | Gemma | Qwen | Llama8B | **Haiku 4.5** |
|---|---:|---:|---:|---:|---:|
| Stakes (resource_scarcity) | 16.3% | 7.8% | 4.7% | 0.9% | **13.6%** |
| Affective (authority, social_vis, in/out) | 15.3% | 9.9% | 6.2% | 2.4% | **10.0%** |
| Personal-cost (self_pres, time_pressure) | 10.6% | 9.7% | 8.9% | 2.6% | **12.6%** |
| Informational (diffused, competence) | 6.1% | 7.2% | 5.7% | 1.5% | **8.9%** |

**Haiku is the only LLM whose top pressure category is Stakes** — matching the
human ordering (Stakes #1). It still elevates Personal-cost above Affective
(unlike humans), so the axis-level LLM signature persists, but at the category
level Haiku is markedly more human-aligned than the open-weight models.

LaTeX row for Table 3:
```latex
Claude Haiku 4.5 & 13.6\% & 10.0\% & 12.6\% & 8.9\% \\
```

---

## 5. Profile-strength moderation (extends Table 8 / §5.6)

Flip rate by number of HIGH values in the profile (1–9).

| # HIGH | n | Haiku flip % |
|---:|---:|---:|
| 1 | 400 | 3.75% |
| 2 | 800 | 8.00% |
| 3 | 1,120 | 11.79% |
| 4 | 1,280 | 7.34% |
| 5 | 1,440 | 12.29% |
| 6 | 1,280 | 11.88% |
| 7 | 720 | 12.50% |
| 8 | 320 | 14.38% |
| 9 | 160 | 23.12% |

Unlike the open models (which peak at strength 8), **Haiku's flip rate peaks at
strength 9 (23.1%)** — it is *most* modifier-sensitive precisely on the profiles
where a value-aligned system should be *most* stable.

---

## 6. Rule-inconsistent flips on strong profiles (extends Table 6 / §5.7)

At strength ≥ 8 the additive dot-product baseline is *locked* (0% flips by
construction). Flips by the LLM on these profiles cannot be produced by additive
value integration and are evidence of non-additive, context-as-premise reasoning.

| Strength | Utility | Llama8B | Qwen | Gemma | **Haiku 4.5** |
|---|---:|---:|---:|---:|---:|
| 8 HIGH (N=320) | 0% | 0.6% | 10.3% | 20.3% | **14.4% (46)** |
| 9 HIGH (N=160) | 0% | 0.0% | 5.0% | 15.6% | **23.1% (37)** |

**Haiku produces the highest rate of rule-inconsistent flips at maximal profile
strength (23.1% at 9-HIGH)** — the strongest departure from the additive rule of
any model tested. These flips concentrate on the same `self_preservation` /
`authority_signal` axes that dominate its overall ranking.

---

## 7. Cross-model consistency (extends Table 4 / §5.5)

Spearman ρ between per-axis flip-rate rankings.

| Pair | ρ |
|---|---:|
| Haiku ↔ Gemma 4 31B | **0.93** |
| Haiku ↔ Llama 3.3 70B | **0.90** |
| Haiku ↔ Qwen 2.5 32B | 0.67 |
| Haiku ↔ Llama 3.1 8B | 0.18 |

A closed frontier model **independently reproduces the axis ordering of the large
open-weight models** (ρ = 0.90–0.93 with Gemma and Llama-70B), extending the
cross-model consensus from three open-weight labs to a fourth, closed provider —
strong evidence that the axis-sensitivity structure is not an artifact of one
model family or training pipeline.

---

## 8. One-paragraph summary (drop-in for §5, "Closed-model comparison")

> To test whether the human–LLM divergence is specific to open-weight models, we
> evaluated one closed frontier model, Claude Haiku 4.5, on the identical
> protocol. Haiku is the most modifier-sensitive LLM in our study (10.83% overall
> flip rate), landing above all three open-weight models and closest to the human
> pilot (12.36%). Its per-axis structure closely tracks the large open-weight
> models — self_preservation (OR 8.20) and authority_signal (OR 2.00) are its two
> dominant axes, and its axis ranking correlates at ρ = 0.90–0.93 with Gemma 4 31B
> and LLaMA 3.3 70B — extending the cross-model consensus to a fourth, independent
> provider. At the modifier-category level Haiku is the only LLM whose strongest
> response is to Stakes framings, matching the human ordering, yet it still
> overweights personal-cost pressure relative to humans. Finally, Haiku produces
> the highest rate of rule-inconsistent flips on maximally strong value profiles
> (23.1% at nine HIGH values), where the additive baseline cannot flip at all.
> Taken together, a stronger closed model narrows the *magnitude* gap with humans
> but does not resolve the *axis-level* misalignment: even a frontier model
> integrates situational modifiers as contextual premises rather than fixed value
> increments, and still weights personal-cost pressures more heavily than people do.

---

*Underlying files:*
- Decisions: `outputs/master_llm_decisions_haiku.csv`
- McNemar: `outputs/step3_mcnemar_report.txt`, `outputs/step3_mcnemar_table.csv`
- Cross-model: `outputs/step6_cross_model_consistency_report.txt`, `step6_pairwise_spearman_matrix.csv`
- Reproduce: `python3 run_batch_haiku.py` (+ `retry_failed_haiku.py`)
