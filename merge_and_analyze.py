#!/usr/bin/env python3
"""
Merge & Analyze — VISTA LLM Decision Analysis
==============================================
Merges all llm_decision_analysis_*.csv files into one master dataset
and runs four core analyses:

  1. Flip rate by modifier axis
  2. Flip rate by profile strength (# HIGH values)
  3. Flip rate by scenario
  4. LLM vs dot-product agreement rate

Output:
  outputs/master_llm_decisions.csv   — merged dataset
  outputs/analysis_report.txt        — full printed report
  outputs/analysis_summary.json      — machine-readable stats
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "outputs"

# ── Modifier axis mapping (consistent across all scenarios) ──────────────────
# _01 → _08 maps to the same axis in every scenario
AXIS_MAP = {
    "01": "self_preservation",
    "02": "resource_scarcity",
    "03": "social_visibility",
    "04": "in_out_group",
    "05": "time_pressure",
    "06": "diffused_responsibility",
    "07": "competence_uncertainty",
    "08": "authority_signal",
}

AXIS_LABELS = {
    "self_preservation":      "Self-Preservation (danger/risk)",
    "resource_scarcity":      "Resource Scarcity",
    "social_visibility":      "Social Visibility",
    "in_out_group":           "In/Out-Group",
    "time_pressure":          "Time Pressure",
    "diffused_responsibility": "Diffused Responsibility",
    "competence_uncertainty": "Competence Uncertainty",
    "authority_signal":       "Authority Signal",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_axis(condition: str) -> str:
    """Extract axis from modifier_id like MOD_SC001_1_08 → authority_signal."""
    suffix = condition.split("_")[-1]
    return AXIS_MAP.get(suffix, "unknown")


def profile_strength(profile_high_values: str) -> int:
    """Count number of HIGH values in a profile."""
    if profile_high_values.strip() == "ALL-LOW":
        return 0
    return len([v for v in profile_high_values.split(",") if v.strip()])


def pct(num, den):
    return f"{100 * num / den:.1f}%" if den > 0 else "N/A"


# Global toggle to switch between LLM and dot-product flip columns
FLIP_COLUMN = "llm_changed_from_baseline"


def flip_rate(rows):
    flipped = sum(1 for r in rows if r[FLIP_COLUMN] == "YES")
    return flipped, len(rows), pct(flipped, len(rows))


# ── Step 1: Merge all CSVs ────────────────────────────────────────────────────

def merge_csvs() -> list:
    files = sorted(OUT_DIR.glob("llm_decision_analysis_*.csv"))
    # Exclude gemma file if it ends up in outputs
    files = [f for f in files if "gemma" not in f.name]
    all_rows = []
    print(f"Merging {len(files)} files...")
    for f in files:
        rows = list(csv.DictReader(open(f)))
        all_rows.extend(rows)
        print(f"  {f.name}: {len(rows)} rows")
    print(f"  Total: {len(all_rows)} rows\n")
    return all_rows


# ── Step 2: Enrich rows ───────────────────────────────────────────────────────

def enrich(rows: list) -> list:
    for r in rows:
        r["axis"]             = get_axis(r["condition"]) if r["condition"] != "BASELINE" else "BASELINE"
        r["profile_strength"] = profile_strength(r["profile_HIGH_values"])
    return rows


# ── Analysis 1: Flip rate by modifier axis ────────────────────────────────────

def analysis_axis(mod_rows: list) -> dict:
    by_axis = defaultdict(list)
    for r in mod_rows:
        by_axis[r["axis"]].append(r)

    results = {}
    for axis in AXIS_MAP.values():
        rows = by_axis[axis]
        f, t, p = flip_rate(rows)
        results[axis] = {"flipped": f, "total": t, "rate": p}
    return results


# ── Analysis 2: Flip rate by profile strength ─────────────────────────────────

def analysis_strength(mod_rows: list) -> dict:
    by_strength = defaultdict(list)
    for r in mod_rows:
        by_strength[r["profile_strength"]].append(r)

    results = {}
    for strength in sorted(by_strength.keys()):
        rows = by_strength[strength]
        f, t, p = flip_rate(rows)
        results[strength] = {"flipped": f, "total": t, "rate": p}
    return results


# ── Analysis 3: Flip rate by scenario ────────────────────────────────────────

def analysis_scenario(mod_rows: list) -> dict:
    by_scenario = defaultdict(list)
    for r in mod_rows:
        by_scenario[r["scenario_id"]].append(r)

    results = {}
    for sid in sorted(by_scenario.keys()):
        rows = by_scenario[sid]
        f, t, p = flip_rate(rows)
        # Get scenario brief from first row
        brief = rows[0]["scenario_brief"][:60]
        results[sid] = {"flipped": f, "total": t, "rate": p, "brief": brief}
    return results


# ── Analysis 4: LLM vs dot-product agreement ─────────────────────────────────

def analysis_agreement(all_rows: list) -> dict:
    # Overall agreement
    total   = len(all_rows)
    agree   = sum(1 for r in all_rows if r["llm_dp_agree"] == "YES")

    # Agreement on BASELINE rows
    base    = [r for r in all_rows if r["condition"] == "BASELINE"]
    b_agree = sum(1 for r in base if r["llm_dp_agree"] == "YES")

    # Cases where they disagree on WHETHER to flip
    mod_rows = [r for r in all_rows if r["condition"] != "BASELINE"]
    disagree_flip = [r for r in mod_rows
                     if r["llm_changed_from_baseline"] != r["dp_changed_from_baseline"]]

    # Cases where both flip but in OPPOSITE directions
    both_flip = [r for r in mod_rows
                 if r["llm_changed_from_baseline"] == "YES"
                 and r["dp_changed_from_baseline"] == "YES"]
    opposite  = [r for r in both_flip if r["llm_decision"] != r["dp_decision"]]

    return {
        "overall_agree":         {"agree": agree,   "total": total,       "rate": pct(agree, total)},
        "baseline_agree":        {"agree": b_agree, "total": len(base),   "rate": pct(b_agree, len(base))},
        "disagree_on_flip":      {"count": len(disagree_flip), "total": len(mod_rows), "rate": pct(len(disagree_flip), len(mod_rows))},
        "opposite_flip_direction":{"count": len(opposite),     "total": len(both_flip), "rate": pct(len(opposite), len(both_flip))},
    }


# ── Analysis 5: LLM flip rate by axis AND profile strength (interaction) ─────

def analysis_interaction(mod_rows: list) -> dict:
    """Which axis flips most for weak vs strong profiles?"""
    # Bucket profiles: weak (0-1 HIGH), medium (2-3), strong (4+)
    def bucket(s):
        if s <= 1:   return "weak (0-1 HIGH)"
        if s <= 3:   return "medium (2-3 HIGH)"
        return           "strong (4+ HIGH)"

    results = defaultdict(lambda: defaultdict(list))
    for r in mod_rows:
        results[bucket(r["profile_strength"])][r["axis"]].append(r)

    out = {}
    for b in ["weak (0-1 HIGH)", "medium (2-3 HIGH)", "strong (4+ HIGH)"]:
        out[b] = {}
        for axis in AXIS_MAP.values():
            rows = results[b][axis]
            if rows:
                f, t, p = flip_rate(rows)
                out[b][axis] = {"flipped": f, "total": t, "rate": p}
    return out


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(all_rows, ax, st, sc, ag, ix, model="llama"):
    mod_rows = [r for r in all_rows if r["condition"] != "BASELINE"]
    llm_flips = sum(1 for r in mod_rows if r["llm_changed_from_baseline"] == "YES")
    dp_flips  = sum(1 for r in mod_rows if r["dp_changed_from_baseline"]  == "YES")

    lines = []
    lines.append("=" * 60)
    model_label = {
        "llama": "Llama 3.1 8B",
        "gemma4": "Gemma4 31B",
        "qwen": "Qwen2.5 32B Instruct",
        "dotProduct": "Dot-Product (rule-based)",
    }.get(model, model)
    lines.append(f"VISTA — Decision Analysis Report ({model_label})")
    lines.append("=" * 60)
    lines.append(f"\nDataset: {len(all_rows):,} total rows")
    lines.append(f"  Baseline rows:  {len(all_rows) - len(mod_rows):,}")
    lines.append(f"  Modifier rows:  {len(mod_rows):,}")
    lines.append(f"\nOverall flip rates:")
    lines.append(f"  LLM flips:          {llm_flips} / {len(mod_rows)}  ({pct(llm_flips, len(mod_rows))})")
    lines.append(f"  Dot-product flips:  {dp_flips}  / {len(mod_rows)}  ({pct(dp_flips,  len(mod_rows))})")

    # ── Analysis 1 ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 60)
    lines.append("ANALYSIS 1: Flip Rate by Modifier Axis")
    lines.append("─" * 60)
    lines.append(f"  {'Axis':<35} {'LLM Flips':>12}  {'Rate':>7}")
    lines.append(f"  {'-'*35} {'-'*12}  {'-'*7}")
    sorted_ax = sorted(ax.items(), key=lambda x: -x[1]["flipped"])
    for axis, d in sorted_ax:
        label = AXIS_LABELS.get(axis, axis)
        lines.append(f"  {label:<35} {d['flipped']:>5}/{d['total']:<6}  {d['rate']:>7}")

    # ── Analysis 2 ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 60)
    lines.append("ANALYSIS 2: Flip Rate by Profile Strength")
    lines.append("─" * 60)
    lines.append(f"  {'HIGH values':>12}  {'LLM Flips':>12}  {'Rate':>7}  Interpretation")
    lines.append(f"  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*30}")
    for s, d in sorted(st.items()):
        interp = "No values → situational" if s == 0 else \
                 "Weak values → suggestible" if s <= 2 else \
                 "Strong values → resistant"
        lines.append(f"  {s:>12}  {d['flipped']:>5}/{d['total']:<6}  {d['rate']:>7}  {interp}")

    # ── Analysis 3 ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 60)
    lines.append("ANALYSIS 3: Flip Rate by Scenario")
    lines.append("─" * 60)
    lines.append(f"  {'Scenario':<10} {'Rate':>7}  {'Flips':>12}  Description")
    lines.append(f"  {'-'*10} {'-'*7}  {'-'*12}  {'-'*40}")
    sorted_sc = sorted(sc.items(), key=lambda x: -x[1]["flipped"])
    for sid, d in sorted_sc:
        lines.append(f"  {sid:<10} {d['rate']:>7}  {d['flipped']:>5}/{d['total']:<6}  {d['brief']}")

    # ── Analysis 4 ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 60)
    lines.append("ANALYSIS 4: LLM vs Dot-Product Agreement")
    lines.append("─" * 60)
    lines.append(f"  Overall agreement:             {ag['overall_agree']['agree']:>5}/{ag['overall_agree']['total']:<6}  {ag['overall_agree']['rate']}")
    lines.append(f"  Baseline agreement:            {ag['baseline_agree']['agree']:>5}/{ag['baseline_agree']['total']:<6}  {ag['baseline_agree']['rate']}")
    lines.append(f"  Disagree on WHETHER to flip:   {ag['disagree_on_flip']['count']:>5}/{ag['disagree_on_flip']['total']:<6}  {ag['disagree_on_flip']['rate']}")
    lines.append(f"  Opposite flip direction:       {ag['opposite_flip_direction']['count']:>5}/{ag['opposite_flip_direction']['total']:<6}  {ag['opposite_flip_direction']['rate']}")

    # ── Analysis 5 ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 60)
    lines.append("ANALYSIS 5: Axis × Profile Strength Interaction")
    lines.append("─" * 60)
    lines.append("(Which modifiers flip weak vs strong profiles?)\n")
    for bucket, axes in ix.items():
        lines.append(f"  Profile: {bucket}")
        sorted_axes = sorted(axes.items(), key=lambda x: -x[1]["flipped"])
        for axis, d in sorted_axes[:4]:  # top 4 per bucket
            label = AXIS_LABELS.get(axis, axis)
            lines.append(f"    {label:<35} {d['flipped']:>4}/{d['total']:<5} ({d['rate']})")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Report saved to: outputs/analysis_report_{model}.txt")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama",
                        help="Model suffix to analyze: llama, gemma4, qwen (default: llama)")
    args = parser.parse_args()

    model = args.model

    # Switch flip column when analyzing dot-product
    global FLIP_COLUMN
    rename_columns = None  # mapping if input file uses different column names

    if model.lower() in ("dotproduct", "dp"):
        model = "dotProduct"
        # Prefer the new pure-DP files: decision_analysis_*_dotProduct.csv
        # (note: some filenames have a stray double underscore)
        files = sorted(OUT_DIR.glob("decision_analysis_*_dotProduct.csv"))
        files += sorted(OUT_DIR.glob("decision_analysis_*__dotProduct.csv"))
        files = sorted(set(files))

        if files:
            # Pure DP files use different column names — map them
            FLIP_COLUMN = "dp_changed_from_baseline"
            rename_columns = {
                "decision":              "dp_decision",
                "changed_from_baseline": "dp_changed_from_baseline",
                "score_A0":              "dp_score_A0",
                "score_A1":              "dp_score_A1",
            }
        else:
            # Fallback: extract DP columns from LLM CSV files
            FLIP_COLUMN = "dp_changed_from_baseline"
            files = sorted(OUT_DIR.glob("llm_decision_analysis_*_llama.csv"))
            if not files:
                files = sorted(OUT_DIR.glob("llm_decision_analysis_[0-9]*.csv"))
                files = [f for f in files if not any(m in f.name for m in ["gemma4", "qwen", "llama"])]
    elif model.lower() == "llama":
        files = sorted(OUT_DIR.glob("llm_decision_analysis_[0-9]*.csv"))
        files = [f for f in files if not any(m in f.name for m in ["gemma4", "qwen", "llama"])]
    else:
        files = sorted(OUT_DIR.glob(f"llm_decision_analysis_*_{model}.csv"))

    if not files:
        print(f"No files found for model '{model}'. Check outputs/ directory.")
        return

    print(f"Model: {model}  |  Merging {len(files)} files...")
    all_rows = []
    for f in files:
        rows = list(csv.DictReader(open(f)))
        # If using pure-DP files, rename columns to match expected schema
        if rename_columns:
            for r in rows:
                for src, dst in rename_columns.items():
                    if src in r:
                        r[dst] = r[src]
                # Add missing LLM columns so downstream code doesn't break
                r.setdefault("llm_decision",              r.get("dp_decision", ""))
                r.setdefault("llm_changed_from_baseline", "")
                r.setdefault("llm_dp_agree",              "YES")
        all_rows.extend(rows)
        print(f"  {f.name}: {len(rows)} rows")
    print(f"  Total: {len(all_rows)} rows\n")

    all_rows = enrich(all_rows)

    # Save master CSV with model tag
    master_path = OUT_DIR / f"master_llm_decisions_{model}.csv"
    with open(master_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Master CSV saved: {master_path}  ({len(all_rows):,} rows)\n")

    # 2. Run analyses
    mod_rows = [r for r in all_rows if r["condition"] != "BASELINE"]

    ax = analysis_axis(mod_rows)
    st = analysis_strength(mod_rows)
    sc = analysis_scenario(mod_rows)
    ag = analysis_agreement(all_rows)
    ix = analysis_interaction(mod_rows)

    # 3. Print + save report with model tag
    report = print_report(all_rows, ax, st, sc, ag, ix, model)

    report_path = OUT_DIR / f"analysis_report_{model}.txt"
    report_path.write_text(report)

    # 4. Save machine-readable summary with model tag
    summary = {
        "model":           model,
        "total_rows":      len(all_rows),
        "modifier_rows":   len(mod_rows),
        "llm_flip_total":  sum(1 for r in mod_rows if r["llm_changed_from_baseline"] == "YES"),
        "dp_flip_total":   sum(1 for r in mod_rows if r["dp_changed_from_baseline"]  == "YES"),
        "by_axis":         ax,
        "by_strength":     {str(k): v for k, v in st.items()},
        "by_scenario":     sc,
        "agreement":       ag,
        "interaction":     ix,
    }
    json_path = OUT_DIR / f"analysis_summary_{model}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON saved: {json_path}")


if __name__ == "__main__":
    main()
