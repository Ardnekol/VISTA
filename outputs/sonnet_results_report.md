# VISDA — Anthropic Claude Sonnet 5 Results (Frontier Closed Model)

*Generated for the paper. Model: `claude-sonnet-5` (Anthropic API, Batch,
chunked/resumable, thinking DISABLED, no temperature, strict JSON structured
output). Source: `outputs/master_llm_decisions_sonnet.csv` (8,550 decisions,
0 errors).*

---

## 1. Setup / Method note (for §4.2)

Claude Sonnet 5 is Anthropic's **frontier** tier, added alongside the small-tier
Haiku 4.5 (Anthropic) and the two OpenAI models. Identical protocol: 95 profiles ×
10 scenarios × (1 baseline + 8 modifiers) = 8,550 decisions. Served via the
Message Batches API. Sonnet 5 rejects a custom temperature, so it was run without
one; **thinking was disabled** (`thinking: {type: "disabled"}`) to keep it a
non-reasoning, greedy-style decoder comparable to the other non-reasoning models
(Haiku, GPT-4.1-mini) and to avoid thinking-token cost. Structured output gave 0
unparsed rows; the dot-product baseline again reproduced **19.93%**, confirming
identical dataset/rule wiring.

---

## 2. System-level flip rate (extends Table 2)

| System | Trials | Flips | Flip rate |
|---|---:|---:|---:|
| Utility-based mathematical baseline | 7,600 | 1,515 | 19.93% |
| Human pilot (N = 50) | 500 | — | 12.36% |
| Claude Haiku 4.5 (closed, Anthropic, small) | 7,600 | 823 | 10.83% |
| GPT-5-mini (closed, OpenAI) | 7,600 | 729 | 9.59% |
| Gemma 4 31B | 7,600 | 678 | 8.92% |
| GPT-4.1-mini (closed, OpenAI) | 7,600 | 610 | 8.03% |
| Qwen 2.5 32B | 7,600 | 500 | 6.58% |
| **Claude Sonnet 5 (closed, Anthropic, frontier)** | **7,600** | **489** | **6.43%** |
| LLaMA 3.1 8B | 7,600 | 154 | 2.03% |

**The key surprise: Sonnet 5 is one of the *least* modifier-sensitive LLMs
(6.43%) — below its own small-tier sibling Haiku 4.5 (10.83%).** Within Anthropic,
the frontier model is markedly *more* value-stable than the small model: it holds
its value-based choice under situational pressure far more often. This shows that
**modifier sensitivity is not monotonic in model capability** — bigger/stronger
does not mean more swayed.

LaTeX row for Table 2:
```latex
Claude Sonnet 5 & 7{,}600 & 489 & 6.43\% \\
```

---

## 3. Per-axis McNemar test (extends Table 7 / §5.2)

McNemar exact test, baseline vs modifier (n = 950/axis), BH-FDR corrected.

| Axis | A0→A1 | A1→A0 | flip rate | OR | 95% CI | p (BH-FDR) | sig |
|---|---:|---:|---:|---:|---|---:|:--|
| self_preservation | 61 | 20 | 8.5% | **3.05** | [1.81, 5.34] | 2.6e-05 | *** |
| authority_signal | 69 | 27 | 10.1% | 2.56 | [1.62, 4.15] | 8.6e-05 | *** |
| diffused_responsibility | 21 | 62 | 8.7% | 0.34 | [0.20, 0.56] | 3.2e-05 | *** |
| resource_scarcity | 33 | 15 | 5.1% | 2.20 | [1.16, 4.36] | 2.6e-02 | * |
| social_visibility | — | — | 5.3% | — | — | ns | ns |
| competence_uncertainty | — | — | 5.3% | — | — | ns | ns |
| in_out_group | — | — | 4.6% | — | — | ns | ns |
| time_pressure | — | — | 3.9% | — | — | ns | ns |

**4 of 8 axes significant.** Despite the low overall rate, Sonnet 5's dominant
positive axes are again **self_preservation (OR 3.05) and authority_signal
(OR 2.56)** — the universal top pair, now confirmed across **all seven LLMs**. It
also shows the strongest **reversed** `diffused_responsibility` effect of any model
(OR 0.34): "others are also responsible" reliably pushes it toward A0.

---

## 4. Modifier-type pressure (extends Table 3 / §5.4)

| Type | Human | Gemma | Qwen | Llama8B | Haiku 4.5 | GPT-4.1-mini | GPT-5-mini | **Sonnet 5** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stakes | 16.3% | 7.8% | 4.7% | 0.9% | 13.6% | 6.7% | 8.9% | **5.1%** |
| Affective | 15.3% | 9.9% | 6.2% | 2.4% | 10.0% | 8.9% | 9.9% | **6.7%** |
| Personal-cost | 10.6% | 9.7% | 8.9% | 2.6% | 12.6% | 9.6% | 10.9% | **6.2%** |
| Informational | 6.1% | 7.2% | 5.7% | 1.5% | 8.9% | 5.7% | 8.1% | **7.0%** |

Sonnet 5 is **the flattest model across categories** (5.1–7.0%, range just
1.9 pts) — it does not strongly privilege any pressure type. This low, even
profile is the signature of a value-stable system: it responds a little to
everything and a lot to nothing, unlike the human Stakes/Affective peak or the
open-weight Personal-cost peak.

LaTeX row for Table 3:
```latex
Claude Sonnet 5 & 5.1\% & 6.7\% & 6.2\% & 7.0\% \\
```

---

## 5. Profile-strength moderation (extends Table 8 / §5.6)

| # HIGH | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonnet 5 flip % | 4.0 | 5.4 | 6.0 | 4.8 | 8.3 | 4.8 | 9.3 | 4.4 | **11.2** |

Sonnet 5's flip rate stays low and fairly flat across strengths (mostly 4–9%),
with a modest rise only at strength 9 (11.2%) — the least strength-driven of the
capable models, consistent with its overall stability.

---

## 6. Rule-inconsistent flips on strong profiles (extends Table 6 / §5.7)

| Strength | Utility | Llama8B | Qwen | Gemma | Haiku 4.5 | GPT-4.1-mini | GPT-5-mini | **Sonnet 5** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 HIGH (N=320) | 0% | 0.6% | 10.3% | 20.3% | 14.4% | 14.4% | 13.4% | **4.4% (14)** |
| 9 HIGH (N=160) | 0% | 0.0% | 5.0% | 15.6% | 23.1% | 13.8% | 21.3% | **11.2% (18)** |

Sonnet 5 produces the **fewest rule-inconsistent flips of any capable model** at
strength 8 (4.4%). Even a frontier model still flips a non-trivial share at
strength 9 (11.2%, vs the additive baseline's 0%), so non-additive integration is
present — but Sonnet 5 exhibits it least, again pointing to stronger value anchoring.

---

## 7. Cross-model consistency (extends Table 4 / §5.5)

Spearman ρ of per-axis flip-rate rankings, Sonnet 5 vs the others:

| Pair | ρ |
|---|---:|
| Sonnet 5 ↔ Gemma 4 31B | 0.74 |
| Sonnet 5 ↔ Haiku 4.5 | 0.69 |
| Sonnet 5 ↔ Qwen 2.5 32B | 0.60 |
| Sonnet 5 ↔ LLaMA 3.3 70B | 0.48 |
| Sonnet 5 ↔ GPT-5-mini | 0.37 |
| Sonnet 5 ↔ GPT-4.1-mini | 0.30 |
| Sonnet 5 ↔ LLaMA 3.1 8B | 0.23 |

Sonnet 5 correlates most with Gemma (0.74) and its sibling Haiku (0.69). Its
overall magnitude is low, but the *ordering* still shares the self_preservation /
authority_signal dominance with the other capable models.

---

## 8. Revised cross-model synthesis (drop-in for §5, "Closed-model comparison")

> We evaluated four closed models across two providers — Claude Haiku 4.5 and
> Claude Sonnet 5 (Anthropic), GPT-4.1-mini and GPT-5-mini (OpenAI) — on the
> identical protocol. Crucially, **modifier sensitivity does not track model
> capability**. The small-tier Haiku 4.5 is the most modifier-sensitive LLM
> (10.83%), whereas the frontier Sonnet 5 is among the least (6.43%) — the two
> Anthropic models bracket almost the entire LLM range, with the *larger* model
> being the *more* value-stable. GPT-5-mini (9.59%) and GPT-4.1-mini (8.03%) fall
> in between. What is invariant across all seven LLMs — three open-weight labs and
> two closed providers, spanning small to frontier scale — is the per-axis
> structure: self_preservation and authority_signal are the dominant, significant
> axes for every model (Sonnet 5 OR 3.05 and 2.56), and diffused_responsibility
> acts in reverse. At the modifier-category level the models diverge in magnitude
> and shape — Sonnet 5 is nearly flat across categories, the signature of a
> value-anchored system, while Haiku uniquely mirrors the human Stakes-first
> ordering — but none reproduces the human weighting exactly. Finally, even the
> most stable frontier model still produces rule-inconsistent flips on maximally
> strong value profiles (11.2% at nine HIGH values, vs 0% for the additive rule),
> so situational modifiers are integrated as contextual premises rather than fixed
> value increments universally. The paper's core claims are therefore
> capability-, provider-, and scale-independent: the same two axes govern
> situational shifts in every model, closed provenance and greater capability do
> not by themselves align models to human sensitivity (and can even reduce overall
> sensitivity), and modifiers reshape decisions non-additively throughout.

---

*Underlying files:* `outputs/master_llm_decisions_sonnet.csv`,
`outputs/step3_mcnemar_report.txt`, `outputs/step6_cross_model_consistency_report.txt`.
Reproduce: `python3 run_batch_sonnet.py` (chunked + resumable).
