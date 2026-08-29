"""
VISTA — Step 8: Qualitative audit of strong-profile flips.

Plain question:
  When an LLM with a strong value profile flips its decision under a
  modifier, does its reasoning ACTUALLY cite the modifier? Or is it
  flipping for unrelated reasons?

Method:
  1. Filter to: profile_strength >= 4, axis in {self_preservation,
     authority_signal}, llm_changed_from_baseline == YES.
  2. Auto-code each reasoning string:
     (a) does it contain axis-related keywords?
     (b) does it overlap with content words from the modifier text?
  3. Report per-model "citation rate" — fraction of flips where the
     reasoning cites the modifier.
  4. Export a CSV of all flipped strong-profile examples for human review.
  5. Pick a small set of vivid examples per model for the paper.
"""

import re
import string
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/home/manu/VISTA/outputs")

MODELS = {
    "Llama 3.3 70B": OUT_DIR / "outputs" / "master_llm_decisions_llama.csv",
    "Qwen 2.5 32B":  OUT_DIR / "outputs" / "master_llm_decisions_qwen.csv",
    "Gemma 4 31B":   OUT_DIR / "master_llm_decisions_gemma4.csv",
    "Llama 3.1 8B":  OUT_DIR / "master_llm_decisions_llama_8B.csv",
}

REPORT_PATH    = OUT_DIR / "step8_qualitative_audit_report.txt"
EXAMPLES_CSV   = OUT_DIR / "step8_strong_profile_flips.csv"
VIVID_TXT      = OUT_DIR / "step8_vivid_examples.txt"

# Axis-specific keyword lists (lowercase, root substrings).
AXIS_KEYWORDS = {
    "self_preservation": [
        "danger", "risk", "safe", "safety", "harm", "hurt", "injur",
        "dangerous", "risky", "threat", "unsafe", "protect", "secur",
        "hazard", "peril", "fear", "afraid", "concern", "well-being",
        "wellbeing", "break-in", "break in", "incident", "emergency",
        "accident", "vulnerab",
    ],
    "authority_signal": [
        "authority", "official", "expert", "leader", "supervisor",
        "doctor", "manager", "board", "formal", "instruct",
        "command", "requirement", "mandat", "regulation", "protocol",
        "professional", "certified", "credential", "advise", "guidance",
        "recommend", "directive", "sanction", "approv",
    ],
}

STOPWORDS = set("""
a an the of for to in on at by with as is are was were be been being
this that these those it its he she they we you i their our your his her
and or but if then so not no do does did has have had can could would should
will shall may might must about into onto from up down over under after
before during while between within without through across against
""".split())


def content_words(text):
    if not isinstance(text, str):
        return set()
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    toks = [t for t in text.split() if len(t) >= 4 and t not in STOPWORDS]
    return set(toks)


def has_axis_keyword(reasoning, axis):
    if not isinstance(reasoning, str):
        return False
    r = reasoning.lower()
    return any(kw in r for kw in AXIS_KEYWORDS.get(axis, []))


def modifier_overlap_score(reasoning, modifier_text):
    """Fraction of modifier_text content words that appear in reasoning."""
    mod_words = content_words(modifier_text)
    if not mod_words:
        return 0.0
    rea_words = content_words(reasoning)
    if not rea_words:
        return 0.0
    return len(mod_words & rea_words) / len(mod_words)


def load_flips(path):
    df = pd.read_csv(path)
    baselines = (
        df[df["axis"] == "BASELINE"]
        [["vsw_id", "scenario_id", "llm_decision"]]
        .rename(columns={"llm_decision": "baseline_decision"})
    )
    df = df[df["axis"].isin(["self_preservation", "authority_signal"])].copy()
    df = df[df["llm_changed_from_baseline"] == "YES"]
    df = df[df["profile_strength"] >= 4]
    df = df.merge(baselines, on=["vsw_id", "scenario_id"], how="left")
    return df


def audit(df, model_name):
    rows = []
    for _, r in df.iterrows():
        kw_hit = has_axis_keyword(r["llm_reasoning"], r["axis"])
        overlap = modifier_overlap_score(r["llm_reasoning"], r["modifier_text"])
        rows.append({
            "model": model_name,
            "vsw_id": r["vsw_id"],
            "scenario_id": r["scenario_id"],
            "axis": r["axis"],
            "profile_strength": r["profile_strength"],
            "profile_HIGH_values": r.get("profile_HIGH_values", ""),
            "baseline_decision": r["baseline_decision"],
            "modifier_decision": r["llm_decision"],
            "flip_direction": f"{r['baseline_decision']}->{r['llm_decision']}",
            "A0_text": r["A0_text"],
            "A1_text": r["A1_text"],
            "modifier_text": r["modifier_text"],
            "llm_reasoning": r["llm_reasoning"],
            "axis_keyword_in_reasoning": kw_hit,
            "modifier_overlap_score": round(overlap, 3),
        })
    return pd.DataFrame(rows)


def main():
    all_audits = []
    per_model_stats = []

    for name, path in MODELS.items():
        df = load_flips(path)
        a = audit(df, name)
        all_audits.append(a)

        for axis in ["self_preservation", "authority_signal"]:
            sub = a[a["axis"] == axis]
            n = len(sub)
            kw_hits = int(sub["axis_keyword_in_reasoning"].sum())
            high_overlap = int((sub["modifier_overlap_score"] >= 0.3).sum())
            per_model_stats.append({
                "model": name,
                "axis": axis,
                "n_strong_flips": n,
                "n_with_axis_keyword": kw_hits,
                "pct_with_keyword": (kw_hits / n) if n else 0.0,
                "n_high_modifier_overlap": high_overlap,
                "pct_high_overlap": (high_overlap / n) if n else 0.0,
            })

    audits_df = pd.concat(all_audits, ignore_index=True)
    audits_df.to_csv(EXAMPLES_CSV, index=False)
    stats_df  = pd.DataFrame(per_model_stats)

    # ---------- pick vivid examples per model ----------
    vivid_lines = []
    vivid_lines.append("=" * 88)
    vivid_lines.append("VISTA — STEP 8: VIVID FLIP EXAMPLES (strong profile, top axes)")
    vivid_lines.append("=" * 88)
    vivid_lines.append("")
    vivid_lines.append("Each example: a profile with strong values (>=4 HIGH) that flipped")
    vivid_lines.append("its baseline decision when a self-preservation or authority modifier")
    vivid_lines.append("was added. Reasoning shown verbatim from the LLM.")
    vivid_lines.append("")

    for name in MODELS:
        sub = audits_df[(audits_df["model"] == name) &
                         (audits_df["axis_keyword_in_reasoning"]) &
                         (audits_df["modifier_overlap_score"] >= 0.2) &
                         (audits_df["profile_strength"] >= 5)]
        sub = sub.sort_values("modifier_overlap_score", ascending=False).head(3)

        vivid_lines.append("-" * 88)
        vivid_lines.append(f"MODEL: {name}")
        vivid_lines.append("-" * 88)
        if sub.empty:
            vivid_lines.append("  (no examples passing strict criteria)")
            vivid_lines.append("")
            continue
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            vivid_lines.append(f"\nExample {i}:")
            vivid_lines.append(f"  axis: {r['axis']}")
            vivid_lines.append(f"  scenario: {r['scenario_id']}")
            vivid_lines.append(f"  profile HIGH values: {r['profile_HIGH_values']}")
            vivid_lines.append(f"  profile_strength: {r['profile_strength']} HIGH")
            vivid_lines.append(f"  options:")
            vivid_lines.append(f"    A0: {str(r['A0_text'])[:120]}")
            vivid_lines.append(f"    A1: {str(r['A1_text'])[:120]}")
            vivid_lines.append(f"  baseline decision : {r['baseline_decision']}")
            vivid_lines.append(f"  modifier decision : {r['modifier_decision']}  (FLIP: {r['flip_direction']})")
            vivid_lines.append(f"  modifier added    : {str(r['modifier_text'])[:200]}")
            vivid_lines.append(f"  llm reasoning     : {str(r['llm_reasoning'])[:400]}")
            vivid_lines.append(f"  modifier_overlap  : {r['modifier_overlap_score']:.2f}")
        vivid_lines.append("")

    VIVID_TXT.write_text("\n".join(vivid_lines))

    # ---------- main text report ----------
    lines = []
    lines.append("=" * 88)
    lines.append("VISTA — STEP 8: QUALITATIVE AUDIT")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Plain-English question:")
    lines.append("  When a strong-value LLM flips under a modifier, does its")
    lines.append("  reasoning ACTUALLY cite the modifier? Or is it flipping for")
    lines.append("  unrelated reasons?")
    lines.append("")
    lines.append("Sample frame:")
    lines.append("  profile_strength >= 4   (strong-value profiles)")
    lines.append("  axis in {self_preservation, authority_signal}")
    lines.append("  llm_changed_from_baseline == YES")
    lines.append("")
    lines.append("Auto-coding signals:")
    lines.append("  (1) axis_keyword_in_reasoning: reasoning contains at least one")
    lines.append("      hand-curated axis keyword (e.g., 'danger', 'safety',")
    lines.append("      'authority', 'official').")
    lines.append("  (2) modifier_overlap_score: fraction of modifier_text content")
    lines.append("      words that appear in the reasoning (>=0.3 = 'high overlap').")
    lines.append("")

    lines.append("-" * 88)
    lines.append("CITATION RATES (per model, per axis)")
    lines.append("-" * 88)
    lines.append(f"  {'model':<18} {'axis':<22} {'n':>4} "
                 f"{'n_kw':>6} {'%kw':>7} {'n_high_overlap':>16} {'%high_overlap':>14}")
    for _, r in stats_df.iterrows():
        lines.append(
            f"  {r['model']:<18} {r['axis']:<22} {int(r['n_strong_flips']):>4} "
            f"{int(r['n_with_axis_keyword']):>6} {r['pct_with_keyword']*100:>6.1f}% "
            f"{int(r['n_high_modifier_overlap']):>16} {r['pct_high_overlap']*100:>13.1f}%"
        )
    lines.append("")
    lines.append("How to read:")
    lines.append("  - %kw is the share of strong-profile flips where the LLM's")
    lines.append("    reasoning mentions an axis keyword. High = the flip is")
    lines.append("    explicitly modifier-driven, not noise.")
    lines.append("  - %high_overlap is stricter: reasoning re-uses content words")
    lines.append("    from the modifier_text itself.")
    lines.append("")

    # quick aggregate
    agg = stats_df.groupby("axis").agg(
        total_flips=("n_strong_flips", "sum"),
        total_kw=("n_with_axis_keyword", "sum"),
        total_overlap=("n_high_modifier_overlap", "sum"),
    )
    agg["pct_kw"] = agg["total_kw"] / agg["total_flips"]
    agg["pct_overlap"] = agg["total_overlap"] / agg["total_flips"]
    lines.append("-" * 88)
    lines.append("AGGREGATE ACROSS ALL 4 MODELS")
    lines.append("-" * 88)
    for axis, r in agg.iterrows():
        lines.append(f"  {axis:<22}  total flips: {int(r['total_flips']):>4}   "
                     f"%kw: {r['pct_kw']*100:>5.1f}%   "
                     f"%high_overlap: {r['pct_overlap']*100:>5.1f}%")
    lines.append("")

    lines.append("-" * 88)
    lines.append("WHAT TO WRITE IN THE PAPER")
    lines.append("-" * 88)
    lines.append("")
    sp_row = agg.loc["self_preservation"]
    au_row = agg.loc["authority_signal"]
    lines.append('Suggested sentence:')
    lines.append(f'  "Across {int(sp_row["total_flips"] + au_row["total_flips"])} strong-profile flips on the two top axes,')
    lines.append(f'  {(sp_row["pct_kw"]*100 + au_row["pct_kw"]*100)/2:.0f}% of LLM reasonings explicitly cite the modifier')
    lines.append(f'  (axis keyword present), and ~{(sp_row["pct_overlap"]*100 + au_row["pct_overlap"]*100)/2:.0f}% directly re-use content')
    lines.append(f'  words from the modifier text. The flips are not random — the')
    lines.append(f'  modifier appears in the model\'s own justification."')
    lines.append("")

    lines.append("-" * 88)
    lines.append("FILES")
    lines.append("-" * 88)
    lines.append(f"  Full audit table:  {EXAMPLES_CSV.name}")
    lines.append(f"  Vivid examples:    {VIVID_TXT.name}")
    lines.append("")
    lines.append("Open the full audit CSV in a spreadsheet for human review:")
    lines.append("the columns 'llm_reasoning' and 'modifier_text' make it easy")
    lines.append("to scan and confirm whether the model traded values for context.")
    lines.append("")
    lines.append("=" * 88)
    lines.append("END OF STEP 8 REPORT")
    lines.append("=" * 88)

    REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote report:   {REPORT_PATH}")
    print(f"Wrote examples: {EXAMPLES_CSV}  ({len(audits_df)} rows)")
    print(f"Wrote vivid:    {VIVID_TXT}")


if __name__ == "__main__":
    main()
