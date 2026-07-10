import csv
from collections import defaultdict

path = 'human_study/analysis_ready.csv'
rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # normalize fields
        r['is_modified'] = int(r['is_modified'])
        r['choice'] = r['choice'].strip()
        r['excluded'] = int(r['excluded']) if r['excluded'] else 0
        rows.append(r)

# group by participant and theme (baseline rows use variant _1, modifiers use variant _2)
pairs = defaultdict(dict)
for r in rows:
    if r['excluded']:
        continue
    pid = r['participant_id']
    theme = r.get('theme_id') or r.get('scenario_id')
    key = (pid, theme)
    # baseline rows have is_modified==0, modifier rows have is_modified==1 and share the same theme_id
    if r['is_modified'] == 0:
        pairs[key]['base'] = r['choice']
    else:
        pairs[key].setdefault('mods', []).append(r['choice'])

# Now for each key with base and at least one modified, compare base to each modified
n_pairs = 0
n_flips = 0
for key, v in pairs.items():
    if 'base' not in v or 'mods' not in v:
        continue
    base_choice = v['base']
    for mod_choice in v['mods']:
        if mod_choice == '':
            continue
        n_pairs += 1
        if mod_choice != base_choice:
            n_flips += 1

print('total_pairs', n_pairs)
print('flips', n_flips)
print('flip_rate', f"{(n_flips / n_pairs * 100) if n_pairs else 0:.2f}%")
