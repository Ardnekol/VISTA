import json

# ============================================================
# ORDER OF SCHWARTZ VALUES
# ============================================================

SCHWARTZ_VALUES = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism"
]

# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = "/home/manu/VISTA/Dataset/uniform_schwartz_profile_sample_100.txt"

# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = "/home/manu/VISTA/Dataset/value_profiles_batch1.json"

# ============================================================
# PARSE INPUT
# ============================================================

profiles = []

with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

for line in lines:

    line = line.strip()

    if not line:
        continue

    parts = line.split()

    # Example:
    # 0 0000000000 weight=0

    profile_index = int(parts[0])
    binary_string = parts[1]

    # ========================================================
    # MAP BITS TO SCHWARTZ VALUES
    # ========================================================

    profile = {}

    for i, value_name in enumerate(SCHWARTZ_VALUES):

        bit = int(binary_string[i])

        profile[value_name] = bit

    # ========================================================
    # CREATE OUTPUT OBJECT
    # ========================================================

    obj = {
        "vsw_id": f"VSW_{profile_index:04d}",
        "profile_index": profile_index,
        "binary_profile": binary_string,
        "active_values_count": binary_string.count("1"),
        "profile": profile
    }

    profiles.append(obj)

# ============================================================
# SAVE JSON
# ============================================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(profiles, f, indent=2)

# ============================================================
# DONE
# ============================================================

print(f"Generated {len(profiles)} value profiles.")
print(f"Saved to: {OUTPUT_FILE}")