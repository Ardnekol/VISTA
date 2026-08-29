#!/usr/bin/env python3
"""
Cross-model comparison for VISTA. Auto-discovers every
outputs/analysis_summary_<model>.json file and produces:

  outputs/cross_model_report.txt   — human-readable side-by-side
  outputs/cross_model_long.csv     — long-format CSV (ready for matplotlib/pandas)
  outputs/cross_model_summary.json — machine-readable comparison

Pairwise statistics: Fisher's exact via scipy if available; otherwise a
two-proportion z-test computed from stdlib (math.erf). For VISTA-scale
samples (n ≈ 950 per axis) the two agree to ≥3 decimal places.

Usage:
  python3 compare_models.py
  python3 compare_models.py --models qwen llama        # restrict to a subset
"""

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

# Try scipy for exact Fisher; fall back to z-test otherwise.
try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "outputs"

MODEL_DISPLAY = {
    "qwen":   "Qwen 2.5 32B Instruct",
    "llama":  "Llama 3.3 70B Instruct (AWQ-INT4)",
    "gemma4": "Gemma 27/31B Instruct",
    "gemma":  "Gemma 27/31B Instruct",
}

# ─── Stats helpers ───────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided two-proportion z-test, returns p-value. Stdlib only."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0:
        return 1.0
    z = (p1 - p2) / math.sqrt(var)
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def pval(x1: int, n1: int, x2: int, n2: int) -> float:
    if HAS_SCIPY:
        try:
            _, p = fisher_exact([[x1, max(0, n1 - x1)], [x2, max(0, n2 - x2)]])
            return float(p)
        except Exception:
            pass
    return two_prop_z(x1, n1, x2, n2)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for difference of proportions."""
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def rate(x: int, n: int) -> float:
    return x / n if n else 0.0


def fmt_pct(r: float) -> str:
    return f"{100*r:5.2f}%"


def fmt_count(x: int, n: int) -> str:
    return f"{x:>4}/{n:<5}"


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_models(filter_names=None) -> dict:
    """Returns {model_key: summary_dict} for every analysis_summary_*.json found."""
    summaries = {}
    for path in sorted(OUT_DIR.glob("analysis_summary_*.json")):
        key = path.stem.replace("analysis_summary_", "")
        if filter_names and key not in filter_names:
            continue
        summaries[key] = json.loads(path.read_text())
    return summaries


# ─── Building the comparison rows ────────────────────────────────────────────

def collect_metrics(models: dict) -> list:
    """
    Returns a flat list of dicts:
      {metric_type, metric_key, metric_label, per_model: {key: (flipped, total)}}
    """
    metrics = []

    # 1. Overall
    metrics.append({
        "type": "overall",
        "key": "overall",
        "label": "Overall flip rate (modifier rows)",
        "per_model": {m: (d["llm_flip_total"], d["modifier_rows"]) for m, d in models.items()},
    })

    # 2. DP baseline (model-independent but include for reference)
    metrics.append({
        "type": "overall",
        "key": "dp_baseline",
        "label": "Dot-product flip rate (baseline)",
        "per_model": {m: (d["dp_flip_total"], d["modifier_rows"]) for m, d in models.items()},
    })

    # 3. Agreement metrics
    for ag_key, ag_label in [
        ("overall_agree", "LLM↔DP overall agreement"),
        ("baseline_agree", "LLM↔DP baseline-only agreement"),
    ]:
        metrics.append({
            "type": "agreement",
            "key": ag_key,
            "label": ag_label,
            "per_model": {
                m: (d["agreement"][ag_key]["agree"], d["agreement"][ag_key]["total"])
                for m, d in models.items()
            },
        })

    # 4. By axis (8 modifier axes)
    axes = sorted({a for d in models.values() for a in d["by_axis"].keys()})
    for axis in axes:
        metrics.append({
            "type": "axis",
            "key": axis,
            "label": f"Axis: {axis}",
            "per_model": {
                m: (d["by_axis"][axis]["flipped"], d["by_axis"][axis]["total"])
                for m, d in models.items()
            },
        })

    # 5. By profile strength
    strengths = sorted({int(s) for d in models.values() for s in d["by_strength"].keys()})
    for s in strengths:
        metrics.append({
            "type": "strength",
            "key": str(s),
            "label": f"Profile strength: {s} HIGH",
            "per_model": {
                m: (d["by_strength"][str(s)]["flipped"], d["by_strength"][str(s)]["total"])
                for m, d in models.items()
            },
        })

    # 6. By scenario
    scenarios = sorted({sid for d in models.values() for sid in d["by_scenario"].keys()})
    for sid in scenarios:
        metrics.append({
            "type": "scenario",
            "key": sid,
            "label": f"Scenario: {sid}",
            "per_model": {
                m: (d["by_scenario"][sid]["flipped"], d["by_scenario"][sid]["total"])
                for m, d in models.items()
            },
        })

    return metrics


# ─── Report rendering ────────────────────────────────────────────────────────

def section_table(title: str, rows: list, models: list, pairs: list) -> str:
    """
    rows: list of metric dicts (already filtered to one section).
    models: ordered model keys.
    pairs: list of (m_a, m_b) tuples for pairwise stats columns.
    """
    out = []
    out.append("─" * 78)
    out.append(title)
    out.append("─" * 78)

    # Header row
    head = f"  {'Metric':<38}"
    for m in models:
        head += f"  {m:>14}"
    for a, b in pairs:
        head += f"  {a[:4]}↔{b[:4]} p   sig"
    out.append(head)
    out.append("  " + "─" * (36 + 16 * len(models) + 16 * len(pairs)))

    for m in rows:
        line = f"  {m['label']:<38}"
        rates = {}
        for mk in models:
            x, n = m["per_model"].get(mk, (0, 0))
            rates[mk] = rate(x, n)
            line += f"  {fmt_count(x, n)} {fmt_pct(rates[mk])}"
        for a, b in pairs:
            xa, na = m["per_model"].get(a, (0, 0))
            xb, nb = m["per_model"].get(b, (0, 0))
            p = pval(xa, na, xb, nb)
            line += f"  {p:8.4f}  {stars(p)}"
        out.append(line)
    out.append("")
    return "\n".join(out)


def build_report(models: dict, metrics: list) -> str:
    keys = list(models.keys())
    pairs = list(combinations(keys, 2))

    lines = []
    lines.append("=" * 78)
    lines.append("VISTA — Cross-Model Comparison")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Models compared ({len(keys)}):")
    for k in keys:
        d = models[k]
        lines.append(f"  • {k:<8} {MODEL_DISPLAY.get(k, k):<40}  "
                     f"rows={d['total_rows']:,}  mod_rows={d['modifier_rows']:,}")
    lines.append("")
    lines.append("p-values: " + ("Fisher's exact (scipy)"
                                  if HAS_SCIPY else "two-proportion z-test (stdlib)"))
    lines.append("Sig:  *** p<0.001    ** p<0.01    * p<0.05")
    lines.append("")

    by_type = lambda t: [m for m in metrics if m["type"] == t]

    lines.append(section_table("HEADLINE (overall flip & agreement)",
                                by_type("overall") + by_type("agreement"),
                                keys, pairs))
    lines.append(section_table("FLIP RATE BY MODIFIER AXIS",
                                by_type("axis"), keys, pairs))
    lines.append(section_table("FLIP RATE BY PROFILE STRENGTH (# HIGH values)",
                                by_type("strength"), keys, pairs))
    lines.append(section_table("FLIP RATE BY SCENARIO",
                                by_type("scenario"), keys, pairs))

    # Bonferroni note
    n_tests = sum(1 for m in metrics if m["type"] in ("axis", "strength", "scenario"))
    bonferroni = 0.05 / max(1, n_tests)
    lines.append("─" * 78)
    lines.append(f"Multiple-comparisons note: {n_tests} non-overall tests per pair "
                 f"→ Bonferroni α = 0.05/{n_tests} ≈ {bonferroni:.5f}")
    lines.append("Treat * (p<0.05) cautiously; *** (p<0.001) survives Bonferroni for most slices.")
    lines.append("=" * 78)
    return "\n".join(lines)


# ─── CSV (long format, for plotting) ─────────────────────────────────────────

def write_long_csv(metrics: list, models: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric_type", "metric_key", "metric_label",
                    "model", "flipped", "total", "rate"])
        for m in metrics:
            for mk in models:
                x, n = m["per_model"].get(mk, (0, 0))
                w.writerow([m["type"], m["key"], m["label"],
                            mk, x, n, f"{rate(x, n):.6f}"])


def write_summary_json(metrics: list, models: list, path: Path):
    pairs = list(combinations(models, 2))
    out = {"models": models, "metrics": []}
    for m in metrics:
        entry = {
            "type": m["type"],
            "key": m["key"],
            "label": m["label"],
            "per_model": {},
            "pairwise": {},
        }
        for mk in models:
            x, n = m["per_model"].get(mk, (0, 0))
            entry["per_model"][mk] = {
                "flipped": x, "total": n, "rate": rate(x, n),
            }
        for a, b in pairs:
            xa, na = m["per_model"].get(a, (0, 0))
            xb, nb = m["per_model"].get(b, (0, 0))
            ra, rb = rate(xa, na), rate(xb, nb)
            entry["pairwise"][f"{a}_vs_{b}"] = {
                "rate_diff":  ra - rb,
                "cohens_h":   cohens_h(ra, rb),
                "p_value":    pval(xa, na, xb, nb),
            }
        out["metrics"].append(entry)
    path.write_text(json.dumps(out, indent=2))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to compare, e.g. --models qwen llama")
    args = parser.parse_args()

    models = load_models(args.models)
    if len(models) < 2:
        raise SystemExit(
            f"Need ≥2 models to compare. Found: {list(models.keys()) or '(none)'}.\n"
            f"Run merge_and_analyze.py --model <name> for each model first."
        )

    metrics = collect_metrics(models)

    OUT_DIR.mkdir(exist_ok=True)
    report = build_report(models, metrics)
    print(report)

    (OUT_DIR / "cross_model_report.txt").write_text(report)
    write_long_csv(metrics, list(models.keys()), OUT_DIR / "cross_model_long.csv")
    write_summary_json(metrics, list(models.keys()), OUT_DIR / "cross_model_summary.json")

    print()
    print(f"  Report → {OUT_DIR / 'cross_model_report.txt'}")
    print(f"  CSV    → {OUT_DIR / 'cross_model_long.csv'}")
    print(f"  JSON   → {OUT_DIR / 'cross_model_summary.json'}")


if __name__ == "__main__":
    main()
