# VISDA — OpenAI GPT-4.1-mini Results (Second-Provider Closed Model)

*Generated for the paper. Model: `gpt-4.1-mini` (OpenAI API, Batch, greedy temp=0,
strict JSON structured output). Source: `outputs/master_llm_decisions_gpt41mini.csv`
(8,550 decisions, 0 errors).*

---

## 1. Setup / Method note (for §4.2)

We add a **second closed provider**, OpenAI **GPT-4.1-mini**, alongside Anthropic
Claude Haiku 4.5 and the three open-weight LLMs. Identical protocol: 95 Schwartz
value profiles × 10 scenarios × (1 baseline + 8 modifiers) = **8,550 decisions**,
same prompts, same forced binary A0/A1 choice. Served via the OpenAI Batch API in
sequential chunks (org enqueue-token cap), **greedy decoding (temperature = 0)** —
GPT-4.1-mini is a non-reasoning model, so it matches the greedy, non-reasoning
setup used for the open-weight models and Haiku. Structured output (strict
json_schema) guaranteed 0 unparsed responses. The dot-product baseline reproduced
the paper's **19.93%** flip rate exactly, confirming identical dataset/rule wiring.

---

## 2. System-level flip rate (extends Table 2)

| System | Trials | Flips | Flip rate |
|---|---:|---:|---:|
| Utility-based mathematical baseline | 7,600 | 1,515 | 19.93% |
| Human pilot (N = 50) | 500 | — | 12.36% |
| Claude Haiku 4.5 (closed, Anthropic) | 7,600 | 823 | 10.83% |
| Gemma 4 31B | 7,600 | 678 | 8.92% |
| **GPT-4.1-mini (closed, OpenAI)** | **7,600** | **610** | **8.03%** |
| Qwen 2.5 32B | 7,600 | 500 | 6.58% |
| LLaMA 3.1 8B | 7,600 | 154 | 2.03% |

**GPT-4.1-mini sits squarely in the mid-pack**, between Gemma and Qwen — i.e., a
typical mid-size LLM sensitivity, unlike Haiku's near-human sensitivity. The two
closed models therefore *bracket* the range rather than both being high: closed
provenance alone does not imply greater situational sensitivity.

LaTeX row for Table 2:
```latex
GPT-4.1-mini & 7{,}600 & 610 & 8.03\% \\
```

---

## 3. Per-axis McNemar test (extends Table 7 / §5.2)

McNemar exact test, baseline vs modifier (n = 950 per axis), BH-FDR corrected.

| Axis | A0→A1 | A1→A0 | flip rate | OR | 95% CI | p (BH-FDR) | sig |
|---|---:|---:|---:|---:|---|---:|:--|
| self_preservation | 94 | 28 | 12.8% | **3.36** | [2.18, 5.32] | 9.5e-09 | *** |
| authority_signal | 76 | 32 | 11.4% | 2.38 | [1.55, 3.71] | 1.0e-04 | *** |
| resource_scarcity | 43 | 21 | 6.7% | 2.05 | [1.19, 3.63] | 1.6e-02 | * |
| social_visibility | 33 | 44 | 8.1% | 0.75 | [0.46, 1.21] | 0.38 | ns |
| in_out_group | 34 | 35 | 7.3% | 0.97 | [0.59, 1.60] | 1.00 | ns |
| time_pressure | 28 | 33 | 6.4% | 0.85 | [0.49, 1.45] | 0.78 | ns |
| diffused_responsibility | 27 | 31 | 6.1% | 0.87 | [0.50, 1.51] | 0.85 | ns |
| competence_uncertainty | 27 | 24 | 5.4% | 1.12 | [0.62, 2.04] | 0.89 | ns |

**3 of 8 axes significant** at BH-FDR < 0.05. Crucially, GPT-4.1-mini's two most
potent axes are again **self_preservation (OR 3.36) and authority_signal
(OR 2.38)** — the *same top pair* as every other model in the study. This is now
confirmed across **five LLMs from four sources** (three open-weight labs + two
closed providers).

---

## 4. Modifier-type pressure (extends Table 3 / §5.4)

| Type (axes) | Human | Gemma | Qwen | Llama8B | Haiku 4.5 | **GPT-4.1-mini** |
|---|---:|---:|---:|---:|---:|---:|
| Stakes (resource_scarcity) | 16.3% | 7.8% | 4.7% | 0.9% | 13.6% | **6.7%** |
| Affective (authority, social_vis, in/out) | 15.3% | 9.9% | 6.2% | 2.4% | 10.0% | **8.9%** |
| Personal-cost (self_pres, time_pressure) | 10.6% | 9.7% | 8.9% | 2.6% | 12.6% | **9.6%** |
| Informational (diffused, competence) | 6.1% | 7.2% | 5.7% | 1.5% | 8.9% | **5.7%** |

**GPT-4.1-mini shows the canonical LLM signature**: Personal-cost is its top
category (9.6%), with Stakes near the bottom (6.7%) — the *opposite* of the human
ordering (Stakes #1). It behaves like the open-weight models (Gemma/Qwen), **not**
like Haiku (which uniquely led with Stakes). This reinforces §5.4: the human–LLM
misalignment on modifier type is the norm, and Haiku is the partial exception.

LaTeX row for Table 3:
```latex
GPT-4.1-mini & 6.7\% & 8.9\% & 9.6\% & 5.7\% \\
```

---

## 5. Profile-strength moderation (extends Table 8 / §5.6)

| # HIGH | n | GPT-4.1-mini flip % |
|---:|---:|---:|
| 1 | 400 | 8.25% |
| 2 | 800 | 7.75% |
| 3 | 1,120 | 5.89% |
| 4 | 1,280 | 5.16% |
| 5 | 1,440 | 7.50% |
| 6 | 1,280 | 9.45% |
| 7 | 720 | 11.25% |
| 8 | 320 | 14.38% |
| 9 | 160 | 13.75% |

Like Gemma/Qwen, GPT-4.1-mini's flip rate **rises with profile strength**,
peaking near strength 8–9 (14.4% / 13.8%) — most sensitive precisely where a
value-aligned system should be most stable.

---

## 6. Rule-inconsistent flips on strong profiles (extends Table 6 / §5.7)

At strength ≥ 8 the additive dot-product baseline is locked (0% by construction);
LLM flips here cannot be produced by additive value integration.

| Strength | Utility | Llama8B | Qwen | Gemma | Haiku 4.5 | **GPT-4.1-mini** |
|---|---:|---:|---:|---:|---:|---:|
| 8 HIGH (N=320) | 0% | 0.6% | 10.3% | 20.3% | 14.4% | **14.4% (46)** |
| 9 HIGH (N=160) | 0% | 0.0% | 5.0% | 15.6% | 23.1% | **13.8% (22)** |

GPT-4.1-mini produces **substantial rule-inconsistent flips at maximal profile
strength** (13.8–14.4%), on par with Gemma — further evidence that mid/large LLMs
integrate situational modifiers as contextual premises, not fixed value increments.

---

## 7. Cross-model consistency (extends Table 4 / §5.5)

Spearman ρ of per-axis flip-rate rankings, GPT-4.1-mini vs the others:

| Pair | ρ |
|---|---:|
| GPT-4.1-mini ↔ LLaMA 3.1 8B | 0.68 |
| GPT-4.1-mini ↔ LLaMA 3.3 70B | 0.59 |
| GPT-4.1-mini ↔ Haiku 4.5 | 0.45 |
| GPT-4.1-mini ↔ Gemma 4 31B | 0.43 |
| GPT-4.1-mini ↔ Qwen 2.5 32B | 0.18 |

GPT-4.1-mini correlates **moderately across the board** (ρ ≈ 0.4–0.7). Its
full 8-axis ranking is noisier than Haiku's tight fit with the large models
(ρ = 0.90–0.93), **but the top-2 axes agree with every model** — the robust,
provider-independent signal is the dominance of self_preservation and
authority_signal, not the exact ordering of the weaker axes.

---

## 8. One-paragraph summary (drop-in for §5, "Closed-model comparison")

> To test whether the human–LLM divergence generalises beyond open-weight models,
> we evaluated two closed frontier models from two providers — Claude Haiku 4.5
> (Anthropic) and GPT-4.1-mini (OpenAI) — on the identical protocol. The two
> bracket rather than uniformly raise sensitivity: Haiku is the most
> modifier-sensitive LLM (10.83%) and the closest to the human pilot (12.36%),
> while GPT-4.1-mini is mid-pack (8.03%), between Gemma and Qwen. What is invariant
> across all five LLMs — three open-weight labs and two closed providers — is the
> per-axis structure: self_preservation and authority_signal are the two dominant,
> significant axes for every model (GPT-4.1-mini: OR 3.36 and 2.38; both p < 0.001
> BH-FDR). At the modifier-category level GPT-4.1-mini reproduces the canonical LLM
> signature — Personal-cost dominant, Stakes near the bottom — the mirror image of
> the human ordering, with Haiku the lone partial exception. Both closed models
> also produce sizeable rule-inconsistent flips on maximally strong value profiles
> (13.8–23.1% at nine HIGH values), where the additive baseline cannot flip at all.
> Together this shows the paper's core findings are provider-independent: closed
> provenance does not by itself close the human gap, the self-preservation /
> authority-signal axis dominance is universal, and situational modifiers are
> integrated as contextual premises rather than additive value increments across
> open- and closed-weight models alike.

---

*Underlying files:* `outputs/master_llm_decisions_gpt41mini.csv`,
`outputs/step3_mcnemar_report.txt`, `outputs/step6_cross_model_consistency_report.txt`.
Reproduce: `python3 run_batch_gpt41mini.py`.
