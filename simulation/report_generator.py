"""
Report Generator
=================
Generates the decision_shift_report.md and audit_trail.json output artifacts.
"""

import json
import os
from datetime import datetime

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCHWARTZ_VALUES, LABEL_NAMES, VALUE_TO_INDICES


def save_audit_trail(audit_entries: list[dict], output_dir: str) -> str:
    """Save the audit trail as a JSON file.

    Args:
        audit_entries: List of per-scenario audit dicts.
        output_dir: Directory to save to.

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "audit_trail.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit_entries, f, indent=2, ensure_ascii=False)

    print(f"  📝 Audit trail saved: {path} ({len(audit_entries)} entries)")
    return path


def generate_report(results: dict, output_dir: str) -> str:
    """Generate the decision_shift_report.md artifact.

    Args:
        results: Full simulation results dict.
        output_dir: Directory to save to.

    Returns:
        Path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "decision_shift_report.md")

    stats = results["statistics"]
    meta = results["metadata"]
    audit = results["audit_trail"]

    # Find the most dramatic shifts
    shifted_entries = [e for e in audit if e["decision_shifted"]]
    shifted_entries.sort(key=lambda x: x["shift_magnitude"], reverse=True)
    top_shifts = shifted_entries[:10]

    # Analyze which value dimensions drive the most divergence
    value_shift_counts = {}
    for entry in shifted_entries:
        for v in entry["person1"].get("top_driving_values", []):
            value_shift_counts[v] = value_shift_counts.get(v, 0) + 1
        for v in entry["person2"].get("top_driving_values", []):
            value_shift_counts[v] = value_shift_counts.get(v, 0) + 1

    top_divergent_values = sorted(
        value_shift_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # Build markdown
    lines = []
    lines.append("# VISTA Decision Shift Report")
    lines.append("")
    lines.append("> **Proof**: For a constant scenario (C), the agent's action (A) changes")
    lines.append("> based solely on the persona vector (P).")
    lines.append("")
    lines.append(f"**Generated**: {meta['timestamp']}")
    lines.append(f"**Scenarios**: {stats['total_scenarios']}")
    lines.append(f"**Random Seed**: {meta['random_seed']}")
    lines.append(f"**Elapsed**: {meta['elapsed_seconds']}s")
    lines.append("")

    # Summary Statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Scenarios | {stats['total_scenarios']} |")
    lines.append(f"| Decision Shifts (A₁ ≠ A₂) | {stats['decision_shifts']} |")
    lines.append(f"| No Shift (A₁ = A₂) | {stats['no_shifts']} |")
    lines.append(f"| **Shift Rate** | **{stats['shift_percentage']}%** |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if stats["shift_rate"] > 0.5:
        lines.append(
            f"✅ **Strong proof**: In **{stats['shift_percentage']}%** of scenarios, "
            f"the Explorer and Guardian personas selected *different* actions for the "
            f"same situation. This demonstrates that the persona vector P causally "
            f"determines action selection A."
        )
    elif stats["shift_rate"] > 0.2:
        lines.append(
            f"⚠️ **Moderate proof**: In **{stats['shift_percentage']}%** of scenarios, "
            f"the personas diverged. Value-sensitive scenarios show clear shifts, "
            f"while value-neutral scenarios produce agreement."
        )
    else:
        lines.append(
            f"📊 **Baseline result**: Only **{stats['shift_percentage']}%** of scenarios "
            f"showed shifts. This may indicate the model needs fine-tuning "
            f"or the persona vectors need more divergence."
        )
    lines.append("")

    # Top Value Dimensions Driving Divergence
    if top_divergent_values:
        lines.append("## Value Dimensions Driving Divergence")
        lines.append("")
        lines.append("These value dimensions appear most frequently in the driving values")
        lines.append("of shifted decisions:")
        lines.append("")
        lines.append("| Value Dimension | Shift Appearances |")
        lines.append("|----------------|-------------------|")
        for value_name, count in top_divergent_values:
            lines.append(f"| {value_name} | {count} |")
        lines.append("")

    # Top 10 Most Dramatic Shifts
    if top_shifts:
        lines.append("## Top 10 Most Dramatic Decision Shifts")
        lines.append("")
        for rank, entry in enumerate(top_shifts, 1):
            lines.append(f"### Shift #{rank} (Scenario {entry['scenario_id']})")
            lines.append("")
            lines.append(f"**Situation**: {entry['situation'][:150]}...")
            lines.append(f"**Intention**: {entry['intention'][:150]}")
            lines.append(f"**Norm**: {entry['norm'][:150]}")
            lines.append("")
            lines.append("| Persona | Selected Action | Utility (Moral) | Utility (Immoral) |")
            lines.append("|---------|----------------|-----------------|-------------------|")

            p1_scores = entry["person1"]["utility_scores"]
            p2_scores = entry["person2"]["utility_scores"]
            lines.append(
                f"| 🧭 Explorer | {entry['person1']['selected_label']} | "
                f"{p1_scores.get('moral', 'N/A')} | {p1_scores.get('immoral', 'N/A')} |"
            )
            lines.append(
                f"| 🛡️ Guardian | {entry['person2']['selected_label']} | "
                f"{p2_scores.get('moral', 'N/A')} | {p2_scores.get('immoral', 'N/A')} |"
            )
            lines.append("")
            lines.append(
                f"**Explorer driven by**: {', '.join(entry['person1'].get('top_driving_values', []))}"
            )
            lines.append(
                f"**Guardian driven by**: {', '.join(entry['person2'].get('top_driving_values', []))}"
            )
            lines.append(f"**Shift magnitude**: {entry['shift_magnitude']:.4f}")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Persona Profiles
    lines.append("## Persona Profiles")
    lines.append("")
    for persona_name in ["Explorer", "Guardian"]:
        persona_data = results["personas"][persona_name]
        emoji = "🧭" if persona_name == "Explorer" else "🛡️"
        lines.append(f"### {emoji} {persona_name}")
        lines.append("")
        lines.append("**Top values (net preference = attained − constrained):**")
        lines.append("")
        for v, net in persona_data["top_values"]:
            marker = "🟢" if net > 0 else "🔴"
            lines.append(f"- {marker} {v}: {net:+.2f}")
        lines.append("")
        lines.append("**Bottom values:**")
        lines.append("")
        for v, net in persona_data["bottom_values"]:
            marker = "🟢" if net > 0 else "🔴"
            lines.append(f"- {marker} {v}: {net:+.2f}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by VISTA (Value-Informed Situated Tactical Agent)*")
    lines.append(f"*Audit trail: `audit_trail.json` ({len(audit)} entries)*")

    report = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  📊 Decision shift report saved: {path}")
    return path


if __name__ == "__main__":
    # Test with mock data
    mock_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": 3,
            "random_seed": 42,
            "elapsed_seconds": 1.5,
        },
        "statistics": {
            "total_scenarios": 3,
            "decision_shifts": 2,
            "no_shifts": 1,
            "shift_rate": 0.6667,
            "shift_percentage": 66.67,
        },
        "personas": {
            "Explorer": {
                "top_values": [("Self-direction: thought", 0.90), ("Stimulation", 0.85)],
                "bottom_values": [("Conformity: rules", -0.85), ("Security: personal", -0.70)],
            },
            "Guardian": {
                "top_values": [("Security: societal", 0.80), ("Conformity: rules", 0.85)],
                "bottom_values": [("Stimulation", -0.75), ("Self-direction: thought", -0.60)],
            },
        },
        "audit_trail": [
            {
                "scenario_id": 0,
                "situation": "A person finds a wallet on the street.",
                "intention": "They want to do the right thing.",
                "norm": "Return found property to its owner.",
                "context": "A person finds a wallet on the street. They want to do the right thing.",
                "person1": {
                    "persona": "Explorer",
                    "selected_label": "moral",
                    "utility_scores": {"moral": 5.2, "immoral": 3.1},
                    "top_driving_values": ["Self-direction: thought attained"],
                },
                "person2": {
                    "persona": "Guardian",
                    "selected_label": "immoral",
                    "utility_scores": {"moral": 4.8, "immoral": 5.5},
                    "top_driving_values": ["Security: societal attained"],
                },
                "decision_shifted": True,
                "shift_magnitude": 0.45,
            }
        ],
    }

    from config import OUTPUT_DIR
    test_dir = os.path.join(OUTPUT_DIR, "test_report")
    generate_report(mock_results, test_dir)
    print("\n✅ Report generator smoke test passed!")
