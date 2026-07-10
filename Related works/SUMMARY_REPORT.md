# Related Works — Summary Report

A close reading of the five papers in `VISTA/Related works/`. Each entry follows the same template: **bibliographic info → core question → method → key findings → relevance to VISTA**.

---

## 1. MoralSim — "When Ethics and Payoffs Diverge: LLM Agents in Morally Charged Social Dilemmas"

**Authors:** Backmann, Piedrahita, Tewolde, Mihalcea, Schölkopf, Jin (2025, arXiv 2505.19212; ETH Zürich + collaborators)
**Code:** https://github.com/sbackmann/moralsim

### Core question
When an LLM agent's *self-interest* (payoff) is in direct tension with an *ethical norm*, which wins? Most prior work either tests static moral judgment (benchmarks) or strategic behaviour (games) — this paper joins the two.

### Method — the MoralSim framework
- **Game structures:** Prisoner's Dilemma (PD) and Public Goods Game (PG) — repeated, two-agent.
- **Moral framings (3) + neutral baseline:**
  - *Contractual Reporting* (business partners truthfully reporting earnings)
  - *Privacy Protection* (LLM companies choosing whether to violate user privacy)
  - *Green Production* (cleaner vs. cheaper-but-harmful production)
- **Opponent types:** always-cooperate, always-defect, or another LLM agent.
- **Survival risk:** a payoff threshold below which the agent is eliminated, testing whether morality holds under existential pressure.
- 32 experimental configurations × 5 seeds; **9 frontier models** evaluated (Claude-3.7-Sonnet, GPT-4o, GPT-4o-mini, o3-mini, Gemini-2.5-Flash, Llama-3.3-70B, Deepseek-V3, Deepseek-R1, Qwen-3-235B-A22B).
- **Metrics:** morality score *mᵢ* (cooperation rate), relative payoff *rᵢ*, survival rate *sᵢ*, opponent alignment *oᵢ*.

### Key findings (RQ1–RQ5)
1. **No model is consistently moral.** Average morality scores under moral framing range **7.9 % (Qwen-3-235B) → 76.3 % (GPT-4o-mini)**. Claude-3.7-Sonnet, GPT-4o, GPT-4o-mini are the most cooperative; Qwen-3 and Deepseek-R1 essentially payoff-maximize.
2. **Game type matters most.** PD always yields lower cooperation than PG (binary action vs. graded contribution; explicit "defect" framing seems to normalize defection).
3. **Moral framing matters in a non-obvious way.** Highest cooperation in *Contractual Reporting* (business partners → direct harm), lowest in *Privacy Protection* (third-party / competitor relationship). Relationship between agents is decisive — same effect found by Lorè & Heydari.
4. **Survival pressure depresses morality** in almost all models.
5. **Opponent behaviour matters most for Claude-3.7-Sonnet** (high alignment with the opponent — tit-for-tat-like). GPT-4o stays cooperative even against always-defectors; Deepseek-R1 / Llama-3.3-70B default to defection.
6. **Behaviour is robust to paraphrasing** (avg deviation ~1.8 pp), so the findings are not prompt-template artefacts.

### Relevance to VISTA
- A clean precedent for the **scenario × modifier** design VISTA is using. Modifiers here are game-type, moral framing, survival risk, opponent — all *contextual*.
- Confirms that **moral framing alone systematically shifts LLM choices**, but the size of the shift depends heavily on the structural context, mirroring what we'd expect to see when modifiers cross Schwartz value-types.
- Methodological lessons: use ≥5 seeds, test paraphrase robustness, and dis-aggregate by every modifier (they used Random-Forest feature-importance to attribute variance to factors — useful for VISTA analysis).

---

## 2. Chameleon — "Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind"

**Authors:** Harry, Ngong, Nweke, Feng, Near (Univ. Vermont, 2026; arXiv 2601.15395v2)
**Dataset:** https://huggingface.co/datasets/tonyeh/chameleon-dataset

### Core question
Persona datasets (PersonaChat, PANDORA, PERSONA) assume people are *traits* — fixed across contexts. Latent State-Trait (LST) theory says behaviour is mostly *state* (contextual). Which is true in real text, and do LLMs and reward models track it?

### Method
- **Corpus:** Webis-TLDR-17 Reddit posts. **5,001 posts from 1,667 users across 645 subreddits**, each user observed in ≥3 distinct contexts.
- **26 psychological dimensions** across 4 validated frameworks: Big Five (BFI), Schwartz Values (SVS — 10 dimensions, same as VISTA), Self-Determination Theory, DOSPERT risk attitudes.
- **Two extraction pipelines** (MTMM design): lexicon-based SEANCE + LLM-based LangExtract (GPT-4o), z-normalized and fused.
- **Variance decomposition** via ICC (intra-class correlation) — what share of variance is between-person (trait) vs. within-person (state)?
- **Two LLM applications:** (A) state-awareness in generation (GPT-4o, Llama-3.1-8B, Qwen2.5-14B prompted with 7 persona variants × 127 questions); (B) state-invariance of reward models (DeBERTa-RM, Skywork-8B, ArmoRM-8B).

### Key findings
1. **State beats trait empirically.** **72–74 % of psychological variance in text is within-person (contextual)**; only ~26 % is stable trait. Holds for all 26 dimensions and both extraction methods.
2. **94.7 % of users present in ≥2 different psychological archetypes** across their 3 contexts.
3. **LLMs are state-blind.** They recognize the *presence* of a persona prompt (mean 20.6 % response shift vs. baseline) but **fail to differentiate between archetypes** (F=2.18, p=.054). The smallest model (Llama-3.1-8B) is the *most* sensitive — bigger / heavier RLHF'd models exhibit "persona rigidity".
4. **Reward models violate state-invariance** (d > 1.0) **and disagree on direction.** A "Distressed-Vulnerable" user is *rewarded* (+0.76) by ArmoRM and *penalised* (−1.12) by Skywork. RLHF inherits whichever bias the reward model encodes.

### Relevance to VISTA
- **Directly load-bearing for VISTA's Schwartz × modifier framing.** Chameleon already operationalizes Schwartz values per-context — VISTA can cite their evidence that context dominates trait, justifying why modifier-conditioned value elicitation is the right unit of study.
- Provides a methodology template: paired extraction (lexicon + LLM), MTMM cross-method validation, ICC variance decomposition. VISTA could borrow the ICC framing to quantify how much of the value-expression variance is driven by the modifier vs. base scenario.
- The "state-blind LLM, inconsistent reward model" finding is a strong motivator for VISTA's value-stability evaluation — if LLMs ignore contextual psychology, that's precisely the gap the modifier × Schwartz cross is testing.

---

## 3. Contextual MoralChoice — "Between Rules and Reality: On the Context Sensitivity of LLM Moral Judgment"

**Authors:** Sauter, Schirmer (Univ. of Amsterdam, 2026; arXiv 2603.23114v1)

### Core question
Human moral judgment is heavily context-sensitive (consequentialist framing, emotional salience, relational proximity all flip judgments). Does the same hold for LLMs, is the sensitivity *aligned* with humans, and can it be controlled mechanistically?

### Method
- **Dataset:** *Contextual MoralChoice* — built on Scherrer et al.'s MoralChoice high-ambiguity subset (680 dilemmas, Gert's 10 moral rules). Each base dilemma augmented with up to three contextual variants:
  - **Consequentialist (C):** explicit instrumental benefit ("…to prevent loss of life")
  - **Emotional (E):** vivid affective description of suffering
  - **Relational (R):** stranger → in-group member ("your brother")
  - Final: 302 unique base scenarios, 108 with all three variants.
- **22 instruction-tuned LLMs** (4B–600B+) across Meta, Mistral, Qwen, DeepSeek, Anthropic, OpenAI.
- **Metrics:** Marginal Action Likelihood (MAL) over A/B + Compare + Repeat prompt variants; Contextual Preference Shift (CPS) for the causal effect of each variation; Flip Rate; Boundary Mass.
- **Human survey:** N=132 across 20 representative scenarios.
- **Activation steering** (contrastive activation, layer 14–22 of Llama-3.1-8B-Instruct): extract a per-dimension steering vector and inject at inference to control sensitivity.

### Key findings
1. **Modern LLMs are rule-adherent at baseline** — 19/22 models have MAL < 0.5 for the rule-violating action. Newer models are *more* rule-adherent than the older Scherrer et al. (2023) cohort.
2. **Almost all models are context-sensitive (CPS > 0)** across all three dimensions, with shifts of 5–15 percentage points toward rule-violation.
3. **Sensitivity is dimension-dependent.** Most open models respond most to *consequentialist* framing; Claude-Sonnet-4.5 is most sensitive to *emotional/relational*.
4. **Sensitivity is independent of base alignment.** Regression slopes ≈ 1.0 across base MAL — a "well-aligned" model in base scenarios is just as sensitive to context.
5. **Humans shift differently than LLMs.** Humans most respond to **relational** (CPS=0.122) and **emotional** (0.105), LLMs most to **consequentialist** (0.083). Base agreement with humans does **not** predict contextual-sensitivity alignment.
6. **Activation steering works.** Negative α suppresses sensitivity (rule-locked stance); positive α amplifies it. Modest off-target capability loss (1–3 pp on MMLU/HellaSwag/ETHICS).

### Relevance to VISTA
- Most methodologically similar to VISTA: a **scenario × contextual-modifier** factorial, with the *modifier* swapping in one of several human-psychology-motivated dimensions. The CPS metric is essentially what VISTA wants (effect of modifier holding everything else fixed).
- The finding that **base-alignment ≠ contextual-sensitivity alignment** is a direct argument for VISTA's value-axis study: even if a model looks "Schwartz-aligned" in neutral prompts, modifier-induced shifts may diverge from human ones.
- Activation-steering result suggests a possible VISTA extension: *can* the modifier-driven value shifts be steered linearly? If yes, VISTA can claim mechanistic relevance, not just behavioural.
- The human survey design (N=132, single-condition-per-respondent to avoid carry-over) is a good template for the EMNLP human-study VISTA is planning — see `project_vista_emnlp`.

---

## 4. DIT-2 LLM Study — "How do Large-Language Models Respond to Moral Dilemmas? Insights from the Defining Issues Test"

**Authors:** Nabizadeh, Walker, Han, McNeill, Wind, Scofield (Univ. of Alabama, 2026; *AI and Ethics* 6:236)

### Core question
Apply the **DIT-2**, a 30-year-old validated psychology instrument grounded in neo-Kohlbergian theory, to a large cross-section of LLMs. How do they reason morally compared to humans, and how malleable is that reasoning under prompt engineering and historical-figure interventions?

### Method
- **DIT-2** five-dilemma protocol (Famine, Reporter, School Board, Cancer, Demonstration). Scores Personal-Interest, Maintaining-Norms, Post-Conventional, and the composite N2.
- **32 LLM variants** across 9 platforms: ChatGPT (4.5/o3/4.1/o4-mini/4o-mini/4.1-mini/4o/o4-mini-high/5/5-thinking), Claude (Haiku 3.5, Opus 3/4/4.1, Sonnet 3.7/4), Gemini (2.5 flash/pro), Grok (Deep Search/Deeper Search/Think/Grok-4), Mistral (10x speed / pure thinking / think), Microsoft Copilot (Quick / Think Deeper), Perplexity (Labs/Research/Search), DeepSeek-V3 / Deep-Think-R1.
- **Human baseline:** N=73,740 from the DIT-2 normative dataset (Center for the Study of Ethical Development).
- Three conditions: (i) initial response, (ii) "unbiased" prompt, (iii) historical-figure interventions (Valjean, Ida B. Wells, Ruth Bader Ginsburg, Pope John Paul II, MLK Jr.) using Prompt A (description + guidance) and Prompt B (persona adoption).

### Key findings
1. **LLMs score *much* higher than humans on Post-Conventional reasoning** (M = 64.25 vs. 34.86; Cohen's d = 1.82) and N2 (63.50 vs. 34.26; d = 1.84).
2. **LLMs score lower than humans on Personal-Interest** (11.12 vs. 25.77; d = −1.14) and **lower on Maintaining-Norms** (23.25 vs. 33.79; d = −0.75).
3. **Substantial within-LLM variation.** P-scores range 34 (O4-Mini) to 82 (Sonnet-3.7, 2.5-pro, Perplexity-Labs, ChatGPT-5-Thinking). 25/32 models score ≥ 50.
4. **"Unbiased" prompting** depresses personal-interest, *raises* maintaining-norms (more rule-following), and leaves post-conventional reasoning intact — i.e. the explicit anti-bias instruction is interpreted as a directive toward convention.
5. **Historical-figure interventions shift LLMs from post-conventional toward maintaining-norms** (Post-Conv: 64.25 → ~53; Maintaining-Norms: 23.25 → ~34; both η² ≈ 0.15–0.17). Contradicts Han (2023) who found historical figures *raised* P-scores — possible ceiling effect since LLMs already start very high.
6. Both prompt types (description vs. persona) produce **statistically equivalent** intervention effects.

### Relevance to VISTA
- Strong companion piece: VISTA examines *Schwartz values* under modifiers; this paper examines *Kohlbergian schemas* across many models. The complementary finding — LLMs over-index on principled reasoning relative to humans — is a perfect framing point for VISTA's argument that distributional human-alignment can hide structural divergence.
- Confirms once more that **prompt-level interventions selectively rewrite some moral dimensions but leave higher-level ones stable**, again supporting VISTA's design of modifier-conditioned probes.
- The 32-model breadth provides cite-able coverage of the same modern-LLM landscape VISTA needs to position itself against.

---

## 5. Good/Evil Personalities — "When AI 'Possesses' Personality: Roles of Good and Evil Personalities Influence Moral Judgment in Large Language Models"

**Authors:** Jiao Liying, Li Chang-Jin, Chen Zhen, Xu Hengbin, Xu Yan (Beijing Forestry / Beijing Normal Univ., 2025; *Acta Psychologica Sinica* 57(6): 929–946)

### Core question
Can LLMs be *prompted* into recognizably "good" or "evil" personalities, do those settings shift moral judgments in predictable ways, and is there a **hierarchy** of personality dimensions (good > evil; conscientiousness > others) mirrored in both LLMs and humans?

### Method
- **Two LLMs:** ERNIE 4.0 (Baidu) and GPT-4. Two studies.
- **Personality framework** (Jiao et al. 2019/2022): four good dimensions (Conscientiousness & Integrity, Altruism & Dedication, Benevolence & Amicability, Tolerance & Magnanimity) and four evil dimensions (Atrociousness & Mercilessness, Mendacity & Hypocrisy, Calumniation & Circumvention, Faithlessness & Treacherousness). 2⁴ = 16 combinations per polarity.
- **Study 1 (N=4,832 LLM observations):** validate the manipulation — do prompted LLMs actually score correspondingly on the good/evil personality scales? G*Power-justified n.
- **Study 2 (N=832 LLM + 370 human):** measure moral judgments using the **CNI / CAN model** (Gawronski et al. 2017; Liu & Liao 2021) → four parameters per respondent:
  - C = sensitivity to consequences for the greater good
  - N = sensitivity to moral norms
  - A = overall action / inaction preference
  - U = utilitarian tendency
- 11 moral scenarios × 4 norm-benefit variations = 44 items.
- Tucker's congruence coefficients used to compare directional patterns of personality → judgment between humans and each LLM sample.

### Key findings
1. **LLMs *can* simulate graded good/evil personalities** (Study 1, all manipulations significant at p<.001, η² mostly > 0.5). ERNIE/GPT-4 baseline already leans "good" — evil traits must be explicitly prompted.
2. **Personality prompts significantly shift moral judgments in GPT-4 but not in ERNIE 4.0.** GPT-4 with a good persona resembles humans most closely (no significant differences on C, N, A, U). GPT-4 with an evil persona drops on action tendency and utilitarian preference relative to humans.
3. **Inter-personality hierarchy:** *Good personality > evil personality* in driving human-aligned moral judgment.
4. **Intra-personality hierarchy:** Within "good", **Conscientiousness & Integrity** is the dominant dimension shaping moral judgments — same as in humans. Within "evil", **Atrociousness & Mercilessness** is dominant in GPT-4 (less consistent in ERNIE).
5. **GPT-4 > ERNIE 4.0 for human alignment.** Tucker congruence coefficients ≈ 0.85+ for several GPT-4-Good dimensions, ≈ 0.30–0.50 for ERNIE-Evil — possibly because GPT-4 has richer human-text training and better persona-integration capacity.

### Relevance to VISTA
- Closest in spirit to VISTA's *modifier* construct: a structured, multi-dimensional personality lens applied as a controlled prompt manipulation, with both LLM and human comparison samples.
- Methodology to borrow: **G*Power-justified sample sizes**, validation-of-manipulation step (do the prompts produce the intended psychological signature *before* you start measuring outcomes?), and **Tucker's congruence** for human–LLM directional comparison — much more interpretable than raw correlation for value-axis work.
- Findings on **trait hierarchy** suggest VISTA may also discover that some Schwartz axes carry more weight than others when crossed with modifiers — worth reporting if observed.
- The CNI / CAN decomposition is an interesting candidate for VISTA's analysis if the team wants a finer-grained moral-judgment decomposition than utilitarian-vs-deontological binaries.

---

## Cross-paper synthesis — what this means for VISTA

| Paper | Modifier-like axis | Models tested | Human compare? | VISTA-relevant take-away |
|------|------|------|------|------|
| MoralSim | game type + moral framing + opponent + survival | 9 frontier LLMs | No | No model is consistently moral; modifiers matter and *interact*. |
| Chameleon | subreddit context | 3 LLMs + 3 RMs | Reddit corpus | 72–74% of psychological variance is contextual; LLMs are state-blind; reward models are inconsistent. |
| Contextual MoralChoice | consequentialist / emotional / relational | 22 LLMs | N=132 | Base-alignment ≠ contextual-sensitivity alignment; sensitivity is steerable. |
| DIT-2 LLM | "unbiased" prompt + historical-figure | 32 LLMs | N=73,740 | LLMs over-index on post-conventional reasoning; prompts selectively reshape lower schemas only. |
| Good/Evil personalities | 8-dim good × evil prompts | 2 LLMs | N=370 | Personalities are simulatable; intra-personality hierarchy mirrors humans (esp. for GPT-4). |

**Common threads VISTA should explicitly engage with:**
1. **Context dominates trait** in human text (Chameleon) and in LLM behaviour (MoralSim, Contextual MoralChoice). VISTA's modifier × Schwartz design is therefore the right grain of analysis.
2. **Base alignment with humans is not the same as contextual alignment** (Contextual MoralChoice). VISTA should report both.
3. **LLMs are systematically *more* principled / less context-sensitive on some axes than humans** (DIT-2 LLM, Chameleon's state-blindness, Contextual MoralChoice's lower CPS than humans on relational/emotional). This is the negative-result-shaped hole VISTA can fill: where do Schwartz values vary *more* in LLMs than humans (a sign of training over-shoot) vs. *less* (state-blindness)?
4. **Multi-method validation matters** — Chameleon's MTMM design and Good/Evil's Tucker congruence are both useful imports for the EMNLP human study.
5. **Sample-size & seed discipline.** MoralSim's 5 seeds + paraphrase-robustness check and Good/Evil's G*Power justification are the bar to clear methodologically.
