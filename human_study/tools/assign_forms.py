"""Assign each contact to one of the 8 VISTA form versions, balancing by domain.

Input  : human_study/contacts.csv with columns: name, email, domain
         domain values: 'CS', 'software', 'medical'
Output : human_study/assignments.csv with the assigned form per person

Run from VISTA/ directory:
    python human_study/tools/assign_forms.py
"""

import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTACTS = REPO_ROOT / "human_study" / "contacts.csv"
ASSIGNMENTS = REPO_ROOT / "human_study" / "assignments.csv"

FORMS = ["F1", "F2"]
RANDOM_SEED = 42  # fix so reruns give the same assignment


def main() -> None:
    if not CONTACTS.exists():
        print(f"Missing {CONTACTS}. Create a CSV with columns: name, email, domain")
        print("domain values must be one of: CS, software, medical")
        return

    rng = random.Random(RANDOM_SEED)
    rows = list(csv.DictReader(CONTACTS.open()))

    by_domain: dict[str, list[dict]] = {"CS": [], "software": [], "medical": []}
    for r in rows:
        d = r["domain"].strip()
        if d not in by_domain:
            print(f"Skipping row with unknown domain: {r}")
            continue
        by_domain[d].append(r)

    assignments: list[dict] = []
    for domain, people in by_domain.items():
        rng.shuffle(people)  # randomize order within domain before round-robin
        for i, person in enumerate(people):
            form = FORMS[i % len(FORMS)]
            assignments.append({
                "name": person["name"],
                "email": person["email"],
                "domain": domain,
                "assigned_form": form,
            })

    rng.shuffle(assignments)  # randomize output order so you don't email by form
    with ASSIGNMENTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email", "domain", "assigned_form"])
        writer.writeheader()
        writer.writerows(assignments)

    # Print per-form count summary
    counts: dict[str, dict[str, int]] = {form: {"CS": 0, "software": 0, "medical": 0} for form in FORMS}
    for a in assignments:
        counts[a["assigned_form"]][a["domain"]] += 1
    print(f"\nWrote {len(assignments)} assignments to {ASSIGNMENTS}\n")
    print(f"{'Form':<6} {'CS':>5} {'software':>9} {'medical':>9} {'total':>7}")
    for form in FORMS:
        c = counts[form]
        total = c["CS"] + c["software"] + c["medical"]
        print(f"{form:<6} {c['CS']:>5} {c['software']:>9} {c['medical']:>9} {total:>7}")


if __name__ == "__main__":
    main()
