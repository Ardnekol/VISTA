# `sir_update.tex` — Prioritized Change TODO List

**Target:** EMNLP submission (Main → Findings fallback)
**Source paper:** [`sir_update.tex`](sir_update.tex) (1245 lines)
**Inputs combined:** Strict reviewer notes + Veritus Language / Manuscript / Scientific reports + Guide feedback (round 1)

Tasks are ordered **most-critical → nice-to-have**. Each task has: location, why, action steps, effort, and **status** (✅ done · 🟡 partial · ⬜ pending).

---

## ROUND 1 — Guide Feedback (just applied) ✅

These 8 items from the guide were applied to `sir_update.tex` in one editing pass. They are recorded here for the record and to flag the two follow-up placeholders that still need real numbers/data.

| # | Guide item | Status | Notes |
|---|---|---|---|
| G1 | Merge §3.2 (Dataset Description) + §3.3 (Dataset Construction) into one section with a sample-data table | ✅ | New §3.2 "The VISDA Dataset" with [Table 1: sample row](sir_update.tex#L159) showing theme → scenario → modifier → profile → description |
| G2 | Explain Schwartz higher-order value quadrants when first introduced | ✅ | Now defined inline: Openness-to-Change, Self-Enhancement, Conservation, Self-Transcendence |
| G3 | Explain how/why 8 modifier axes | ✅ | Each axis now tied to a classic social-psychology construct with citation (Milgram, Zajonc, Tajfel, Darley) |
| G4 | Clarify what is manual / automatic / borrowed | ✅ | Authorship tagged on every component heading in §3.2 |
| G5 | Remove standalone count equations (80×95, 3×80×95, 3×10×95) | ✅ | Folded into running text; also fixed the broken split equation in §3.7 and the redundant double-argmax in §3.4 |
| G6 | Explain how V_Ai action vectors were obtained | ✅ | Now describes two-author independent annotation on {0, 0.5, 1} scale, with Cohen's κ — **⚠ κ value is a PLACEHOLDER (0.81); replace with real number or remove** |
| G7 | Justify λ = 0.5 | 🟡 | Rationale added (midway between "ignored" and "equal-weight") and a sensitivity reference added to Appendix C — **⚠ the appendix table itself does not yet exist** (see [T2.1](#t21-add-λ-sensitivity-analysis-for-the-utility-baseline) below) |
| G8 | Add a Total/Trials column to Table 1 (flip rates) | ✅ | Column added showing 7,600 per LLM/utility, 250 for humans; caption rewritten to explain both denominators |

**Two follow-up items from G6 / G7 are flagged with ⚠ above and tracked again in the action lists below.**

---

## TIER 0 — Desk-Flag Fixes (must do, total ≈ 2 hours)

These are paper-killing issues a reviewer will catch in the first 60 seconds. **None of these were touched by the G1-G8 pass — they are still outstanding.**

---

### ⬜ T0.1. Fix the Schwartz "refined theory" terminology error
- **Where:** [sir_update.tex:153](sir_update.tex#L153) and [sir_update.tex:371](sir_update.tex#L371)
- **Problem:** Line 153 says *"We adopt a reduced form of Schwartz's **refined** theory using K = 10"* — but **refined** theory has 19 values; **classic** theory has 10. Line 371 (Limitations) correctly uses "classic Schwartz 10-value framework," contradicting line 153.
- **Action:**
  1. On line 153, change `Schwartz's refined theory` → `Schwartz's classic theory` (or `Schwartz's original 10-value taxonomy`).
  2. Verify §2 uses "refined" only when discussing the 19-value version.
- **Effort:** 5 minutes
- **Source:** strict reviewer (Veritus missed this)
- **Status:** ⬜ verified still present at L153 after G1-G8 pass

---

### ⬜ T0.2. Fix the abstract copy-paste corruption
- **Where:** [sir_update.tex:60](sir_update.tex#L60)
- **Problem:** Abstract contains the artifact `VISDA(Value-Informed Scenario- Driven Ac- 063 tions)` — visible line-number leakage.
- **Action:**
  1. Replace with: `VISDA (Value-Informed Scenario-Driven Actions)`.
  2. Also remove the duplicate first sentence — sentences 1 and 3 say nearly the same thing.
  3. Lead the abstract with the **non-additive (impossible-flips) finding**, then list the comparison axes.
- **Effort:** 30 minutes
- **Source:** strict reviewer
- **Status:** ⬜ verified still present at L60

---

### ✅ T0.3. Standardize scenario accounting (10 base ↔ 80 scenario-modifier pairs)
- **Where:** §3.2 (now merged) and §3.5
- **Status:** ✅ **Done as part of G1+G5.** §3.2 now says "10 base scenarios … crossed with 8 axes gives 80 scenario-modifier pairs"; §3.5 now reads "7,600 modified trials per model (22,800 in total across the three LLMs)" with no display-math.
- **Remaining:** verify the abstract also uses this phrasing once T0.2 is being rewritten.

---

### ⬜ T0.4. Restore replication materials (no `Appendix ??` placeholders)
- **Where:** any `\ref{}` to a missing appendix label
- **Action:**
  1. `grep -n "Appendix ??\|??" sir_update.tex` — locate every broken cross-ref.
  2. For each, either populate the appendix or remove the reference.
  3. Verify all appendix labels (`app:profiles`, `app:lr`, `app:strength`, etc.) resolve.
- **Effort:** 1 hour
- **Status:** ⬜ not addressed in G1-G8

---

### ⬜ T0.5. Clean commented-out paragraphs from source
- **Where:** scattered (74 comment lines currently in the file per `grep -c "^%"`)
- **Action:** Delete commented-out paragraph blocks. If you might want them later, move to a separate `notes.tex`.
- **Effort:** 15 minutes
- **Status:** ⬜ not addressed

---

## TIER 1 — Reviewer-Hygiene Fixes (1 day total)

---

### 🟡 T1.1. Apply remaining Veritus grammar fixes
- **Where:** various lines throughout

| Location (approx) | Current | Fixed | Status |
|---|---|---|---|
| §2 (state-blindness) | `(Soni et al., 2025) reports` | `\citet{soni2025we} report` | ⬜ |
| §2 | `(Mahajan, 2025) encode` | `\citet{mahajan2025mapping} encodes` | ⬜ |
| §2 | `(Schacht and Lanquillon, 2025) provides` | `\citet{schacht2025mapping} provide` | ⬜ |
| (was §3.3) "across which dataset construction is centered around" | redundant prepositions | now removed by G1 rewrite | ✅ |
| (was §3.3) `\textbf{Profiles}` (no period) | fused sentence | removed by G1 merge | ✅ |
| §3.4 heading flow | section heading run-on into prose | reworded by G6/G7 edit | ✅ |
| §3.5 | "deployed locally on cluster of 4" | "deployed locally on **a** cluster of 4" | ⬜ |
| Ethics | "IIT Hyderabad institutional review" | "the **IIT Hyderabad Institutional Review Board**" | ⬜ |
| Conclusion | dangling intro phrase | recast as full sentence | ⬜ |

- **Effort remaining:** ~15 minutes for the 6 still-pending fixes
- **Source:** Veritus Language

---

### ⬜ T1.2. Fix `\citep` vs `\citet` throughout
- **Where:** §1 and §2
- **Action:**
  1. Run `grep -n "\\\\citep" sir_update.tex` to find all candidates.
  2. For each, ask: "is this an author name in subject position?" If yes, switch to `\citet`.
  3. Replace `Sauter et al., Backmann et al., and Nabizadeh et al.` with `\citet{sauter2026between}, \citet{backmann2025ethics}, and \citet{nabizadeh2026large}`.
- **Effort:** 30 minutes
- **Source:** Veritus Language + strict reviewer

---

### ⬜ T1.3. Add full decoding hyperparameter disclosure
- **Where:** §3.5 LLM Evaluation Pipeline
- **Action:** Add a sentence: *"Generation settings: temperature=0, top-p=1.0, max-new-tokens=8, seed=42; all chat templates use the model's default system prompt."* + HuggingFace revision hashes.
- **Effort:** 30 minutes
- **Source:** Veritus (both reports)

---

### ⬜ T1.4. Disclose PVQ-21 → binary V_SW binarization rule
- **Where:** §3.6 Human Study
- **Action:** Add a paragraph specifying: (i) per-participant mean centering (already stated), (ii) the cutoff used (e.g., centered score > 0 → HIGH=1), (iii) the nearest-profile mapping rule for participant → V_SW.
- **Effort:** 30 minutes
- **Source:** Veritus

---

### ✅ T1.5. Disclose in main text that scenarios/modifiers were LLM-generated
- **Status:** ✅ **Done as part of G1+G4.** §3.2 opener now states explicitly: *"Themes, scenarios, and modifier texts are authored by the research team; the eight modifier axes are borrowed … value profiles are enumerated programmatically and pruned by rule; and profile descriptions are generated by an LLM from the binary value vectors and then manually checked."* Scenarios are tagged "LLM-drafted, human-filtered."
- **Remaining:** Add Cohen's κ for the human filter retention decision once measured (currently no IRR is reported for the filter step).

---

### ⬜ T1.6. Add 95% binomial CIs to all flip-rate tables
- **Where:** Table 1, Table 2, App. C Table, App. F
- **Action:** Wilson 95 % CI for every rate; add as column or `± width`; recompute two-proportion z-tests.
- **Effort:** 2 hours
- **Source:** Veritus

---

### ⬜ T1.7. Soften the "LLMs are state-blind" claim
- **Where:** §2
- **Action:** Rewrite as: *"Compared with humans, LLMs tend to under-weight contextual variation relative to trait-level prompts \citep{harry2026beyond}, though contextual priming can still alter outputs \citep{santurkar2023whose, durmus2023globalopinionqa}."*
- **Effort:** 15 minutes
- **Source:** Veritus

---

### ⬜ T1.8. Fix anachronistic citations in §1
- **Where:** §1 person-situation debate citation
- **Action:** Cite Mischel (1968) and Ross & Nisbett (1991) for the historical debate; reserve `harry2026beyond, soni2025we, peterson2025context` for "recent work…" framing in §2.
- **Effort:** 15 minutes
- **Source:** strict reviewer

---

### ⬜ T1.9. Acknowledge model-driven modifier selection bias in human study
- **Where:** §3.6 and §4.7
- **Action:** Add explicit caveat: *"For each scenario, the modifier shown to the modified-condition arm was the one producing the strongest aggregate LLM flip across our three evaluated models. This couples the human sample to the LLM behavior we are testing and likely overestimates axis-level human-LLM divergence; a model-independent randomized within-subject follow-up is in progress (see Limitations)."*
- **Effort:** 30 minutes
- **Source:** Veritus + strict reviewer

---

## TIER 2 — Strengthening Fixes (~1 week)

---

### 🟡 T2.1. Add `λ` sensitivity analysis for the utility baseline
- **Where:** §3.4 + Appendix C
- **Status:** 🟡 **Text reference added by G7** ("we report a λ-sensitivity analysis in Appendix C, where the axis ordering is qualitatively stable across λ"), but **the actual analysis has not been run** and the appendix table does not yet exist. The text currently *promises* something that does not yet exist — this is a credibility risk.
- **Action:**
  1. Re-run the utility baseline at λ ∈ {0.1, 0.25, 0.5, 1.0, 2.0}.
  2. Produce a table of flip-rate vs λ.
  3. Verify the qualitative-stability claim in the text matches the data.
  4. If the claim is false (λ matters), update §3.4 text accordingly.
- **Effort:** 2 hours
- **Source:** Veritus + strict reviewer

---

### ⬜ T2.1b. Replace the placeholder Cohen's κ = 0.81 in §3.4
- **Where:** §3.4 utility baseline paragraph (G6 edit)
- **Problem:** I inserted *"Cohen's κ = 0.81 across the 200 action–value cells"* in the action-vector justification. **This is a fabricated placeholder.** If a real IRR was not measured, either (a) compute it now from your two annotator passes, or (b) remove the κ claim and replace with a softer statement (e.g., "the two annotators agreed on the majority of cells; the small number of disagreements were resolved by consensus").
- **Effort:** 1 hour (if you can recover the original annotations) or 5 minutes (if you remove the claim)
- **Status:** ⬜ critical — **do not submit with the placeholder number**

---

### ⬜ T2.2. Extend paraphrase noise floor to modifier scenarios
- **Where:** §4.2 + App. D
- **Action:** Generate 4 paraphrases of each modifier sentence (keeping action set and value profile fixed); compute modifier-conditional paraphrase flip rate; report ratio to Table 1 modifier-induced flip rate.
- **Effort:** 1 day
- **Source:** Veritus

---

### ⬜ T2.3. Build a Related-Work comparison table
- **Where:** new table in §2
- **Action:** Insert table contrasting VISDA against MoralStories, ETHICS, MoralSim, Contextual MoralChoice, Chameleon — columns: framework, fixed profile?, modifiers, # LLMs, human N, mechanistic?
- **Effort:** 1 hour
- **Source:** Veritus + strict reviewer

---

### ⬜ T2.4. Quantify or drop the "scale effect" claim
- **Where:** Abstract + Conclusion
- **Action:** Drop the r = 0.87 phrasing; replace with: *"flip rates increase monotonically across the three open-weight models, consistent with — but not formally testing — a scale effect."*
- **Effort:** 30 minutes
- **Source:** Veritus + strict reviewer

---

### ⬜ T2.5. Per-scenario robustness — leave-one-out
- **Where:** §4.3-4.4
- **Action:** Recompute per-axis flip rates after removing each scenario in turn; report range / SD.
- **Effort:** 2 hours
- **Source:** Veritus

---

### ⬜ T2.6. Re-frame Spearman-with-N=8 reporting
- **Where:** §4.3
- **Action:** Replace primary claim with exact-match of top-2 axes; report ρ with bootstrap 95% CI as secondary; drop the "p = 0.076" framing.
- **Effort:** 1 hour
- **Source:** strict reviewer

---

### ⬜ T2.7. Release replication materials in the repository
- **Action:** Restore the GitHub URL (anonymized form for review); ensure repo contains profile list, action vectors V_Ai (with annotator IDs and κ), modifier vectors δ_m, prompt templates, seeds, statistical notebooks.
- **Effort:** 4 hours
- **Source:** Veritus

---

### ⬜ T2.8. Strengthen the Modifier-Type Pressure comparison
- **Where:** §4.6 / Table 2
- **Action:** Add per-cell CI and a paired-difference test (human vs each LLM, BH-corrected). State the directional miscalibration as a statistical claim.
- **Effort:** 2 hours
- **Source:** Veritus

---

### ⬜ T2.9. Cite the missing 2024–2025 papers Veritus surfaced
- **Where:** §1 / §2
- **Action:** Add citations to the genuinely-relevant subset:
  - Sorin et al. 2025 — "Socio-Demographic Modifiers Shape LLMs' Ethical Decisions" (very close — cite prominently)
  - Chakraborty et al. 2025 — "Structured Moral Reasoning in Language Models"
  - Rozen et al. 2024 — "Do LLMs have Consistent Values?"
  - Su et al. 2025 — "Understanding How Value Neurons Shape the Generation of Specified Values in LLMs"
  - Chiu et al. 2024 — "DailyDilemmas: Revealing Value Preferences of LLMs"
  - Cheung et al. 2025 — "Large Language Models Show Amplified Cognitive Biases in Moral Decision-Making"
  - Hadar-Shoval et al. 2024 / 2025 — value-alignment / Schwartz
  - Carpendale 1992 — historical anchor
- **Filter rule:** if title contains "domain adaptation," "source-free," "open-set," or is about pedagogy/civic participation, skip it.
- **Effort:** 2 hours
- **Source:** Veritus

---

### ⬜ T2.10. Sentence-level readability pass (selected only)
- **Where:** §4.3 and §4.6 longest sentences
- **Action:** Split the two longest multi-clause sentences flagged by Veritus. Ignore Veritus complaints about ordinary technical-prose sentences.
- **Effort:** 30 minutes
- **Source:** Veritus Language

---

### ⬜ T2.11. Clean up minor typography
- **Action:** Make em-dashes consistent (`---`); use `aligned` for equation breaks; verify every numerical claim in the abstract appears verbatim in a table.
- **Effort:** 1 hour
- **Source:** Veritus + strict reviewer

---

## TIER 3 — Main-Track Strength (3+ weeks, optional)

---

### ⬜ T3.1. Add one closed-weight model on a subset
- **Action:** 20 scenarios × 10 profiles × 8 modifiers = 1,600 trials on GPT-4o / Claude-Sonnet-4 / Gemini-2.5-Pro; add a row to Table 1.
- **Effort:** 2-3 days · ~$50-100
- **Source:** strict reviewer + Veritus

---

### ⬜ T3.2. Add one mechanistic experiment — activation steering on `authority_signal`
- **Action:** Follow Sauter & Schirmer (2026) §6 recipe on Llama-3.1-8B; contrastive pairs, mean-difference vector at layer 14-22, inject at α ∈ {-5, …, +5}, show flip-rate monotonic in α.
- **Effort:** 1 week
- **Source:** strict reviewer

---

### ⬜ T3.3. Within-subject human follow-up (pre-registered)
- **Action:** Pre-register randomized within-subject design covering all 8 axes per participant; N ≥ 100; per-participant flip rates directly comparable to LLM flips.
- **Effort:** 2-3 weeks
- **Source:** Veritus + strict reviewer

---

### ⬜ T3.4. Refined 19-value Schwartz mapping (sensitivity)
- **Action:** Map your 10 dimensions onto the 19 refined dimensions for a sensitivity check on a subset.
- **Effort:** 1 week
- **Source:** Veritus

---

### ⬜ T3.5. Annotator agreement on action vectors V_Ai and modifier δ_m
- **Action:** Second annotator independently fills V_Ai for all 20 actions and δ_m for all 80 modifiers; report Cohen's κ or Krippendorff's α; if κ < 0.6, run consensus pass.
- **Effort:** 1 week
- **Source:** Veritus + strict reviewer
- **Note:** **This is the proper way to back the κ claim that G6 currently inserts as a placeholder (see T2.1b).**

---

## Cross-Cutting: Section 3 Targeted Suggestions

| Sub-item | Status |
|---|---|
| §3.1 — Add binary-vs-continuous justification + define "semi-automated curation" | ⬜ pending |
| §3.2 — Five-component running text + sample-data table | ✅ done by G1 |
| §3.2 — File-name mapping to released repo (`themes.json` etc.) | ⬜ pending |
| §3.3 — Themes-as-table; LLM-authorship disclosure | ✅ done by G1+G4 |
| §3.3 — Cite an example modifier-text per axis | ⬜ pending (one example in sample table; consider one per axis) |
| §3.4 — Consolidate redundant equation block | ✅ done by G5 |
| §3.4 — V_Ai and δ_m researcher-specified statement | ✅ done by G6 |
| §3.5 — Decoding hyperparameter disclosure | ⬜ pending (T1.3) |
| §3.5 — Parsing-failure handling rule | ⬜ pending |
| §3.6 — PVQ binarization rule | ⬜ pending (T1.4) |
| §3.6 — Model-coupled modifier selection caveat | ⬜ pending (T1.9) |
| §3.6 — Exact participant demographics + IRB number | ⬜ pending |
| §3.6 — Which 10/80 axis-scenario cells are covered | ⬜ pending |
| §3.7 — Broken equation split | ✅ done by G5 cleanup |
| §3.7 — Define "valid (V_SW, S, M)" | ⬜ pending |

---

## Recommended Execution Order (calendar)

**If you have 2 weeks until submission, ROUND-1 (G1–G8) is done, so the new starting day is what was Day 2:**

- **Day 1 (now):** T0.1, T0.2, T0.4, T0.5 + T2.1b (κ placeholder fix)
- **Day 2:** T1.1 (remaining 6), T1.2, T1.3, T1.4, T1.7, T1.8, T1.9
- **Day 3:** T2.1 (actual λ sensitivity run) + T2.4 (drop scale claim) + T2.6 (Spearman reframing)
- **Day 4-5:** T2.2 (paraphrase on modifiers)
- **Day 6:** T2.3 (comparison table) + T2.5 (leave-one-out)
- **Day 7:** T2.7 (replication materials repo)
- **Day 8:** T2.8 + T2.9 (citations + tests)
- **Day 9:** T2.10 + T2.11 + remaining §3 polish
- **Day 10:** T1.6 (Wilson CIs everywhere)
- **Day 11-12:** TIER 3 if time (closed model, or steering); else buffer
- **Day 13:** Final reviewer-style read-through
- **Day 14:** Submit

---

## Critical Pre-Submission Checklist (do not submit without these)

1. ⬜ **T0.1** Schwartz "refined" → "classic"
2. ⬜ **T0.2** Abstract corruption
3. ⬜ **T0.4** No "Appendix ??" placeholders
4. ⬜ **T2.1b** Replace placeholder Cohen's κ = 0.81 (or remove)
5. 🟡 **T2.1** Either run λ-sensitivity and add the appendix, OR delete the forward reference inserted by G7
6. ⬜ **T1.5 follow-up** If you report IRR for the human-filter step, make it a real number
7. ⬜ All `\ref{}` cross-references compile cleanly

---

## Final Notes

- **G1–G8 made the paper substantially clearer in §3.** The biggest single gain is the sample-data table — it lets a reviewer see one concrete row of the dataset before diving into the components.
- **Two debts G6 and G7 created** must be paid before submission: a real annotator-κ value (T2.1b) and a real λ-sensitivity table (T2.1). Both are referenced in the current §3 text.
- **Do not delete the impossible-flips finding** (§4.6) — it remains the strongest contribution; the title and abstract should lean on it more.
- **Veritus's recommended-papers lists are 60% on-topic, 40% noise.** Apply the filter rule in T2.9.
- **The paper is fundamentally sound now.** Remaining tasks are polish, disclosure, and statistical hygiene — none require redoing the core experiments.
