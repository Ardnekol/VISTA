# VISTA Human Evaluation — Instrument Package (simple English)

Three blocks the participant fills before seeing any scenarios:
1. **PVQ-21** — Schwartz values, third-person ("this person") framing
2. **Forced-choice trade-offs** — pick one from each pair
3. **SDS-10** — True/False statements to catch faking-good

All wording below is adapted for simpler English so non-native speakers can answer easily. In your paper, cite the original instruments and say *"items adapted for simpler English."*

---

## Block A — PVQ-21 (the value questionnaire)

**What to tell the participant:**
> "Below are short descriptions of different people. For each one, please tell us **how much this person is like you**. There are no right or wrong answers — just pick what feels true."

**Answer scale (6 options for every item):**
1. Not like me at all
2. Not like me
3. A little like me
4. Somewhat like me
5. Like me
6. Very much like me

**Items:**

| # | Item text (simple English) | Value measured |
|---|---|---|
| A1  | "This person enjoys coming up with new ideas. They like doing things in their own way." | Self-Direction |
| A2  | "This person wants to be rich. Having money and nice things matters to them." | Power |
| A3  | "This person believes everyone in the world should be treated equally. Equal chances for all matter to them." | Universalism |
| A4  | "This person likes to show what they can do. They want others to admire their work." | Achievement |
| A5  | "This person wants to feel safe in their daily life. They avoid anything risky." | Security |
| A6  | "This person likes surprises. They want to do many different things in life." | Stimulation |
| A7  | "This person believes people should follow the rules. Even when no one is watching." | Conformity |
| A8  | "This person likes to listen to people who are different from them. Even if they disagree, they want to understand others." | Universalism |
| A9  | "This person likes to be humble. They don't try to stand out." | Tradition |
| A10 | "Having a good time matters to this person. They like to enjoy themselves." | Hedonism |
| A11 | "This person likes to make their own choices. They want to be free to plan their own life." | Self-Direction |
| A12 | "This person loves helping the people around them. They care about others' well-being." | Benevolence |
| A13 | "Being very successful matters to this person. They hope others will notice their achievements." | Achievement |
| A14 | "This person wants the government to keep them safe. A strong country matters to them." | Security |
| A15 | "This person looks for adventure. They enjoy taking risks and having an exciting life." | Stimulation |
| A16 | "This person always tries to behave properly. They don't want to do anything that others would call wrong." | Conformity |
| A17 | "This person wants to be respected by others. They like it when people do what they say." | Power |
| A18 | "Being loyal to friends matters a lot to this person. They give time and care to the people close to them." | Benevolence |
| A19 | "This person cares about nature. Protecting the environment matters to them." | Universalism |
| A20 | "Tradition matters to this person. They follow customs from their religion or family." | Tradition |
| A21 | "This person looks for chances to have fun. They like doing things that feel good." | Hedonism |

### Scoring PVQ-21 (for the researcher)

1. For each value, take the **average** of its 2 or 3 items.
2. For each participant, also compute their **overall mean** across all 21 items.
3. **Subtract** the participant's overall mean from each value score → centered score. This removes the bias from people who rate everything high or everything low. Schwartz strongly recommends this step.
4. Use the centered scores in the regression.

Item-to-value map:
- Self-Direction: A1, A11
- Power: A2, A17
- **Universalism: A3, A8, A19**
- Achievement: A4, A13
- Security: A5, A14
- Stimulation: A6, A15
- Conformity: A7, A16
- Tradition: A9, A20 *(A9 is humility — some papers drop it, document your choice)*
- Benevolence: A12, A18
- Hedonism: A10, A21

---

## Block B — Forced-choice trade-offs

**What to tell the participant:**
> "For each pair below, pick the one that is **more important to you**. Both may feel important — but please choose only one."

| # | Option A | Option B | Tension measured |
|---|---|---|---|
| B1 | Following the traditions I grew up with | Choosing my own path, even if it goes against tradition | Tradition vs. Self-Direction |
| B2 | Feeling safe in my daily life | Having new experiences and adventures | Security vs. Stimulation |
| B3 | Being known as successful in my field | Helping people close to me, even if no one notices | Achievement vs. Benevolence |
| B4 | Having power and influence over others | Working for fairness and equal rights for everyone | Power vs. Universalism |
| B5 | Behaving properly and meeting what others expect | Enjoying life and having fun | Conformity vs. Hedonism |
| B6 | Being humble and not seeking attention | Showing my skills so others recognize me | Tradition vs. Achievement |

### Scoring Block B

- The chosen value gets **+1**, the other gets **0**.
- Use as a **backup check** on PVQ-21. If a person rates everything high in Block A but their forced-choice answers contradict that, they may be faking.

---

## Block C — A few statements about yourself (SDS-10)

**Important:** Do NOT call this block "honesty check" or "lie detector" in the form. Use the label *"A few statements about yourself"*.

**What to tell the participant:**
> "Please tell us if each statement is **True** or **False** about you."

| # | Statement (simple English) | "Faking-good" answer |
|---|---|---|
| C1  | "I always admit it when I make a mistake." | True |
| C2  | "I always do what I tell others to do." | True |
| C3  | "I never feel upset when someone asks me to return a favour." | True |
| C4  | "I have never felt annoyed when someone disagreed with me." | True |
| C5  | "I have never said anything to hurt another person's feelings." | True |
| C6  | "Sometimes I enjoy gossiping." | False |
| C7  | "I have sometimes taken advantage of someone." | False |
| C8  | "Sometimes I want to get back at someone instead of forgiving them." | False |
| C9  | "Sometimes I really insist on having things my own way." | False |
| C10 | "Sometimes I feel like breaking something." | False |

### Scoring SDS-10

- Give **+1** every time the participant gives the "faking-good" answer in the right column.
- Score range: 0–10. Higher = more likely faking.
- **Suggested thresholds:**
  - 0–4: low → treat answers as genuine
  - 5–7: moderate → include SDS score as a control variable in the regression
  - 8–10: high → consider excluding, OR report results with and without these participants
- Pre-register your threshold before collecting data.

---

## Attention checks (hide them inside other blocks)

Don't put these in their own section — that defeats the purpose. Drop them inside Block A and the decision block:

> **AC1** *(place between A10 and A11):* "To show you are reading carefully, please pick 'A little like me' for this item."
>
> **AC2** *(place in the decision block):* "For this question, please pick the second option to show you are reading carefully."

**Exclusion rule (pre-register this!):** drop any participant who fails an attention check OR finishes the whole form in less than 4 minutes.

---

## Google Forms setup notes

- **One section per block.** Make every item *required*.
- **Shuffle questions** inside Block A and the decision block (Form settings → Shuffle question order).
- **Don't add an "Other" box** on Likert or forced-choice items.
- **Group A vs. Group B** (for sibling counterbalancing): Google Forms can't randomize between conditions. Easiest fix: make **two forms** (Form-A, Form-B) and split your shared link 50/50, or share each link with half your contacts.
- **Timing:** Google Forms only logs submission timestamp, not per-question time. If you need per-question timing, use the *Form Timer* add-on or move to Qualtrics / LimeSurvey.
- **Estimated time for the participant:** ~12 minutes (Blocks A+B+C ≈ 7 min, decision block ≈ 5 min).

---

## What to cite in the paper

- **Schwartz, S. H. (2003).** *A proposal for measuring value orientations across nations.* ESS Questionnaire Development Report. *(PVQ-21 source — note your simpler-English adaptation in the methods section.)*
- **Schwartz, S. H., et al. (2012).** *Refining the theory of basic individual values.* JPSP 103(4).
- **Strahan, R., & Gerbasi, K. C. (1972).** *Short, homogeneous versions of the Marlowe-Crowne Social Desirability Scale.* Journal of Clinical Psychology 28(2).
