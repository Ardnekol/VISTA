# VISDA — OpenAI GPT-5-mini Results (Current-Generation Closed Model)

*Generated for the paper. Model: `gpt-5-mini` (OpenAI API, Batch, chunked,
`reasoning_effort="minimal"`, strict JSON structured output). Source:
`outputs/master_llm_decisions_gpt5mini.csv` (8,550 decisions, 0 errors).*

---

## 1. Setup / Method note (for §4.2)

GPT-5-mini is the **current-generation (GPT-5 family)** closed model, added
alongside Claude Haiku 4.5 and GPT-4.1-mini. Identical protocol: 95 profiles × 10
scenarios × (1 baseline + 8 modifiers) = 8,550 decisions, same prompts, forced
binary A0/A1. Served via the OpenAI Batch API in sequential chunks (org
enqueue-token cap). GPT-5-mini is a **reasoning model**, run at
`reasoning_effort="minimal"`; unlike the non-reasoning models it does not take a
custom temperature, so it is not strictly greedy — this is the one methodological
difference from the other models and should be noted in Limitations. Structured
output guaranteed 0 unparsed rows; the dot-product baseline again reproduced
**19.93%**, confirming identical dataset/rule wiring.

---

## 2. System-level flip rate (extends Table 2)

| System | Trials | Flips | Flip rate |
|---|---:|---:|---:|
| Utility-based mathematical baseline | 7,600 | 1,515 | 19.93% |
| Human pilot (N = 50) | 500 | — | 12.36% |
| Claude Haiku 4.5 (closed, Anthropic) | 7,600 | 823 | 10.83% |
| **GPT-5-mini (closed, OpenAI)** | **7,600** | **729** | **9.59%** |
| Gemma 4 31B | 7,600 | 678 | 8.92% |
| GPT-4.1-mini (closed, OpenAI) | 7,600 | 610 | 8.03% |
| Qwen 2.5 32B | 7,600 | 500 | 6.58% |
| LLaMA 3.1 8B | 7,600 | 154 | 2.03% |

**GPT-5-mini is the 2nd most modifier-sensitive LLM**, behind only Haiku 4.5.
The two **current-generation** closed models (Haiku 4.5, GPT-5-mini) are the two
most sensitive LLMs overall, both above every open-weight model **and** above the
older GPT-4.1-mini — suggesting a generational trend: newer/stronger closed models
move *toward* the human sensitivity level.

LaTeX row for Table 2:
```latex
GPT-5-mini & 7{,}600 & 729 & 9.59\% \\
```

---

## 3. Per-axis McNemar test (extends Table 7 / §5.2)

McNemar exact test, baseline vs modifier (n = 950/axis), BH-FDR corrected.

| Axis | A0→A1 | A1→A0 | flip rate | OR | 95% CI | p (BH-FDR) | sig |
|---|---:|---:|---:|---:|---|---:|:--|
| self_preservation | 102 | 31 | 14.0% | **3.29** | [2.18, 5.09] | 3.9e-09 | *** |
| authority_signal | 78 | 42 | 12.6% | 1.86 | [1.26, 2.77] | 3.6e-03 | ** |
| diffused_responsibility | 25 | 54 | 8.3% | 0.46 | [0.28, 0.76] | 3.9e-03 | ** |
| in_out_group | 34 | 57 | 9.6% | 0.60 | [0.38, 0.93] | 3.6e-02 | * |
| social_visibility | 26 | 45 | 7.5% | 0.58 | [0.34, 0.96] | 0.054 | ns |
| resource_scarcity | 49 | 36 | 8.9% | 1.36 | [0.87, 2.15] | 0.29 | ns |
| competence_uncertainty | 33 | 42 | 7.9% | 0.79 | [0.48, 1.27] | 0.50 | ns |
| time_pressure | 42 | 33 | 7.9% | 1.27 | [0.79, 2.07] | 0.50 | ns |

**4 of 8 axes significant** at BH-FDR < 0.05. Its two strongest positive-direction
axes are once more **self_preservation (OR 3.29) and authority_signal (OR 1.86)** —
the universal top pair. GPT-5-mini also shows significant **reversed** effects on
`diffused_responsibility` (OR 0.46) and `in_out_group` (OR 0.60): those modifiers
consistently push it toward A0.

---

## 4. Modifier-type pressure (extends Table 3 / §5.4)

| Type (axes) | Human | Gemma | Qwen | Llama8B | Haiku 4.5 | GPT-4.1-mini | **GPT-5-mini** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stakes (resource_scarcity) | 16.3% | 7.8% | 4.7% | 0.9% | 13.6% | 6.7% | **8.9%** |
| Affective (authority, social_vis, in/out) | 15.3% | 9.9% | 6.2% | 2.4% | 10.0% | 8.9% | **9.9%** |
| Personal-cost (self_pres, time_pressure) | 10.6% | 9.7% | 8.9% | 2.6% | 12.6% | 9.6% | **10.9%** |
| Informational (diffused, competence) | 6.1% | 7.2% | 5.7% | 1.5% | 8.9% | 5.7% | **8.1%** |

**GPT-5-mini shows the canonical LLM signature**: Personal-cost is its top
category (10.9%), with Stakes near the bottom — the mirror image of the human
ordering (Stakes #1). Like GPT-4.1-mini and the open-weight models, and unlike
Haiku (the lone Stakes-first LLM), GPT-5-mini does not reproduce the human
category weighting despite its high overall sensitivity.

LaTeX row for Table 3:
```latex
GPT-5-mini & 8.9\% & 9.9\% & 10.9\% & 8.1\% \\
```

---

## 5. Profile-strength moderation (extends Table 8 / §5.6)

| # HIGH | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5-mini flip % | 4.8 | 8.8 | 9.6 | 8.4 | 8.9 | 9.7 | 11.8 | 13.4 | **21.2** |

Flip rate rises with profile strength and **spikes at strength 9 (21.2%)** — like
Haiku, GPT-5-mini is *most* sensitive exactly on the profiles where a value-aligned
system should be *most* stable.

---

## 6. Rule-inconsistent flips on strong profiles (extends Table 6 / §5.7)

At strength ≥ 8 the additive baseline is locked (0% by construction); LLM flips
here cannot come from additive value integration.

| Strength | Utility | Llama8B | Qwen | Gemma | Haiku 4.5 | GPT-4.1-mini | **GPT-5-mini** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 HIGH (N=320) | 0% | 0.6% | 10.3% | 20.3% | 14.4% | 14.4% | **13.4% (43)** |
| 9 HIGH (N=160) | 0% | 0.0% | 5.0% | 15.6% | 23.1% | 13.8% | **21.3% (34)** |

GPT-5-mini produces **21.3% rule-inconsistent flips at nine HIGH values** — second
only to Haiku, and far above the additive baseline's 0% — strong evidence of
non-additive, context-as-premise reasoning even in a current-generation model.

---

## 7. Cross-model consistency (extends Table 4 / §5.5)

Spearman ρ of per-axis flip-rate rankings, GPT-5-mini vs the others:

| Pair | ρ |
|---|---:|
| GPT-5-mini ↔ Gemma 4 31B | 0.78 |
| GPT-5-mini ↔ LLaMA 3.3 70B | 0.73 |
| GPT-5-mini ↔ Haiku 4.5 | 0.65 |
| GPT-5-mini ↔ GPT-4.1-mini | 0.60 |
| GPT-5-mini ↔ Qwen 2.5 32B | 0.59 |
| GPT-5-mini ↔ LLaMA 3.1 8B | 0.41 |

GPT-5-mini **clusters with the large models** (Gemma 0.78, LLaMA-70B 0.73) — a
tighter fit than GPT-4.1-mini — reinforcing that the self_preservation /
authority_signal axis dominance is shared by the more capable models across
providers.

---

## 8. Combined three-model, two-provider picture (drop-in for §5)

> We evaluated three closed models spanning two providers — Claude Haiku 4.5
> (Anthropic), GPT-4.1-mini and GPT-5-mini (OpenAI) — on the identical protocol.
> The two current-generation models are the two most modifier-sensitive LLMs in
> the study (Haiku 4.5 10.83%, GPT-5-mini 9.59%), both above every open-weight
> model and above the older GPT-4.1-mini (8.03%), indicating that stronger,
> newer closed models move toward — but do not reach — the human sensitivity level
> (12.36%). Across all six LLMs, from three open-weight labs and two closed
> providers, the per-axis structure is invariant: self_preservation and
> authority_signal are the dominant, significant axes for every model (GPT-5-mini
> OR 3.29 / 1.86). At the modifier-category level, five of six LLMs — including
> both OpenAI models — show the canonical signature of overweighting Personal-cost
> and underweighting Stakes, the mirror image of the human ordering, with Haiku the
> sole partial exception. Finally, both current-generation closed models produce
> large rule-inconsistent flips on maximally strong value profiles (21–23% at nine
> HIGH values), where the additive baseline cannot flip at all. The paper's core
> claims are therefore provider- and generation-independent: situational modifiers
> reshape decisions through the same two axes across every model, closed models
> narrow but do not close the human gap, and modifiers are integrated as contextual
> premises rather than additive value increments throughout.

---

*Underlying files:* `outputs/master_llm_decisions_gpt5mini.csv`,
`outputs/step3_mcnemar_report.txt`, `outputs/step6_cross_model_consistency_report.txt`.
Reproduce: `python3 run_batch_gpt5mini.py`.
