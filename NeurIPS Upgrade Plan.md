# VISTA Phase 2: NeurIPS Upgrade Plan

This plan outlines the steps to upgrade VISTA from a static, linear proof-of-concept to a dynamic, learning-based framework capable of top-tier academic publication. We will focus on the highest-impact improvements identified in the review.

## User Review Required

> [!IMPORTANT]
> Please review the phases below. Phases 1 (Learned Utility) and 2 (Causal Interventions) are the most critical immediate steps. Phase 3 (LLM Integration) requires choosing an LLM backend (e.g., a local open-source model like Llama-3/Mistral, or an API). Let me know which phases you want to prioritize first!

## Proposed Changes

---

### Phase 1: Learned Utility Function (Replacing the Dot Product)

Currently, VISTA uses a static dot product: U(a, P) = Σ V_{a, i} * P_i. We will upgrade this to a learned neural network that captures non-linear value trade-offs.

#### [MODIFY] [utility.py](file:///home/manu/VISTA/stage2_action_selection/utility.py)
- Replace `compute_utility` with an inference call to a PyTorch Multi-Layer Perceptron (MLP) or Attention module.
- The model will take concatenated inputs `[V_a, P]` (76 dimensions) and output a continuous utility score.

#### [NEW] [train_utility.py](file:///home/manu/VISTA/stage2_action_selection/train_utility.py)
- Create a script to train this new utility network.
- **Training Strategy**: We will use contrastive learning. Given a scenario where a persona strongly aligns with action A over action B, the network learns to maximize the margin between U(A, P) and U(B, P).

---

### Phase 2: Causal Value Interventions (The "Interpretability" Proof)

Instead of just hardcoding "Explorer" and "Guardian", we will treat the persona vector P as a continuous control panel to find exact decision boundaries.

#### [NEW] [causal_intervention.py](file:///home/manu/VISTA/simulation/causal_intervention.py)
- Pick a scenario that is highly contested.
- Sweep a single value dimension (e.g., *Security: societal*) from 0.0 to 1.0 in increments of 0.05, holding all other values constant.
- Record the exact threshold at which the decision flips from "Immoral" to "Moral".
- Generate visualizations (e.g., sensitivity curves) for the paper.

---

### Phase 3: Generative LLM Integration (Control & Alignment)

We will upgrade VISTA from merely *selecting* pre-written actions to *steering* the generation of new actions.

#### [NEW] [llm_controller.py](file:///home/manu/VISTA/stage3_generative_alignment/llm_controller.py)
- **Prompting**: Send a scenario and a text-based version of the Persona to an LLM (e.g., Llama-3).
- **Generation**: The LLM generates a custom action.
- **VISTA as the Judge**: VISTA's Stage 1 RoBERTa model tags the generated action. We then compute the utility of the generated response against the desired persona to see if the LLM successfully aligned with the requested values.

---

### Phase 4: Rigorous Evaluation & Baselines

To satisfy reviewers, we need to prove VISTA beats standard approaches.

#### [MODIFY] [run_simulation.py](file:///home/manu/VISTA/simulation/run_simulation.py)
- **Random Baseline**: Does VISTA beat a coin flip?
- **LLM Baseline**: Ask an LLM to pick the action without VISTA's utility function. How often does the LLM hallucinate or fail to adhere to the persona? Is VISTA strictly better at enforcing value adherence?
- Add statistical tests (already prototyped in Phase 1) directly into the simulation auto-reporting.

## Open Questions

1. **LLM Access**: For Phase 3, do you have access to API keys (OpenAI/Anthropic) or should we set up a local open-weights model using HuggingFace/Ollama on your GPU?
2. **Phase Priority**: Do you want to start by building the Learned Utility function (Phase 1), or the Causal Interventions tool (Phase 2), which requires no training and is pure experimental science?

## Verification Plan

- **Phase 1**: The learned MLP must maintain or exceed the 85.4% shift rate achieved by the linear baseline, while showing evidence that it can resolve complex value conflicts (e.g., when a persona cares deeply about two values that dictate opposite actions).
- **Phase 2**: Generate a causal sensitivity plot showing a clean sigmoid-like transition as a single value is dialed up or down.
- **Phase 3**: Prove that VISTA can accurately detect when an LLM strays from the assigned persona constraint.
