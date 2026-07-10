#!/usr/bin/env python3
"""
Map human modifier rows to MOD_SC IDs and compute Cohen's kappa between humans and each model.

Outputs:
 - outputs/human_modifier_flips.csv : per-participant per-modifier flip boolean
 - outputs/kappa_per_model.csv : Cohen's kappa per model
 - outputs/kappa_table.tex : LaTeX table of kappas
 - outputs/kappa_bar.png : bar plot of kappas

This script uses modifiers_batch1.json as the canonical ordering of modifiers per scenario.
If exact ordering doesn't match human rows, it will attempt to match by modifier_text.
"""

import csv
import json
import os
from collections import defaultdict
from glob import glob

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

MOD_FILE = os.path.join(ROOT, 'modifiers_batch1.json')
HUMAN_CSV = os.path.join(ROOT, 'human_study', 'analysis_ready.csv')
REFINED_HUMAN_FLIPS = os.path.join(ROOT, 'outputs', 'human_modifier_flips_refined.csv')

MASTER_PATTERN = os.path.join(OUT, 'master_llm_decisions_*.csv')


def load_modifiers():
    # Try primary modifiers file, fall back to Dataset/modifiers_batch1.json if needed
    paths_to_try = [MOD_FILE, os.path.join(ROOT, 'Dataset', 'modifiers_batch1.json')]
    mods = None
    for p in paths_to_try:
        try:
            with open(p, 'r') as f:
                mods = json.load(f)
            print(f"Loaded modifiers from {p}")
            break
        except Exception as e:
            print(f"Could not load modifiers from {p}: {e}")
            mods = None

    if mods is None:
        raise RuntimeError('No valid modifiers_batch1.json found')

    by_scenario = defaultdict(list)
    # mods can be either a flat list of modifier dicts, or a list of scenario dicts with 'modifiers'
    if isinstance(mods, list) and mods and isinstance(mods[0], dict):
        first = mods[0]
        if 'modifier_id' in first:
            # flat list
            for m in mods:
                mid = m.get('modifier_id')
                txt = m.get('modifier_text','').strip()
                axis = m.get('axis')
                parts = mid.split('_') if mid else []
                if len(parts) >= 3:
                    sc = parts[1] + '_' + parts[2]
                    by_scenario[sc].append({'id':mid,'text':txt,'axis':axis})
        else:
            # scenario-wrapped list
            for scen in mods:
                sc = scen.get('scenario_id')
                if not sc:
                    continue
                for m in scen.get('modifiers', []):
                    mid = m.get('modifier_id')
                    txt = m.get('modifier_text','').strip()
                    axis = m.get('axis')
                    by_scenario[sc].append({'id':mid,'text':txt,'axis':axis})

    return by_scenario


def read_human_rows():
    rows = []
    with open(HUMAN_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_human_mapping(mod_by_scenario, human_rows):
    """Return mapping (participant_id, modifier_id) -> flip (0/1)

    Strategy:
    - Group human rows by participant_id and theme_id. Within each group, baseline row has condition BASELINE; modifier rows have condition like MOD_SC001_1_03 in 'scenario_id' or 'condition'. We'll try to find scenario/modifier info; otherwise, map by order using modifier_text.
    """
    mapping = {}
    # first build text->id lookup per scenario
    text_lookup = {}
    for sc, mods in mod_by_scenario.items():
        text_lookup[sc] = {m['text']: m['id'] for m in mods}

    # group rows by participant and theme_id (baseline is per-theme)
    groups = defaultdict(list)
    for r in human_rows:
        pid = r.get('participant_id') or r.get('participant') or r.get('worker_id')
        theme = (r.get('theme_id') or r.get('theme') or '').strip()
        if not pid or not theme:
            continue
        groups[(pid, theme)].append(r)

    total = 0
    for (pid, sc), rows in groups.items():
        # find baseline choice (is_modified == 0)
        baseline_choice = None
        for r in rows:
            try:
                is_mod = int(r.get('is_modified','0'))
            except Exception:
                is_mod = 0
            if is_mod == 0:
                baseline_choice = r.get('choice')
                break

        if baseline_choice is None:
            # no baseline for this participant+scenario
            continue

        # for each modified row, map via scenario_id and axis to modifier id and compute flip
        for r in rows:
            try:
                is_mod = int(r.get('is_modified','0'))
            except Exception:
                is_mod = 0
            if is_mod == 0:
                continue
            axis = (r.get('axis') or '').strip()
            sc = (r.get('scenario_id') or r.get('scenario') or '').strip()
            mod_id = None
            if sc and sc in mod_by_scenario:
                for m in mod_by_scenario[sc]:
                    if m.get('axis') == axis or m.get('text') == (r.get('modifier_text') or '').strip():
                        mod_id = m['id']
                        break

            if mod_id is None:
                # try matching by axis across scenarios
                for scn, mods in mod_by_scenario.items():
                    for m in mods:
                        if m.get('axis') == axis:
                            mod_id = m['id']
                            break
                    if mod_id:
                        break

            if mod_id is None:
                continue

            # determine flip: choice differs from baseline
            ch = r.get('choice')
            try:
                flip = 1 if str(ch) != str(baseline_choice) else 0
            except Exception:
                flip = 0

            mapping[(pid, mod_id)] = flip
            total += 1

    print(f"Built human mapping for {total} modifier rows")
    return mapping


def read_master_model(mod_pattern):
    model_data = {}
    for path in glob(mod_pattern):
        name = os.path.basename(path).replace('master_llm_decisions_','').replace('.csv','')
        rows = {}
        with open(path,newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                # modifier id is in 'condition' column for master files
                mid = row.get('condition') or row.get('modifier_id') or row.get('modifier')
                if not mid or not mid.startswith('MOD_SC'):
                    continue
                # try llm_changed_from_baseline
                flipped = None
                for k in ['llm_changed_from_baseline','changed_from_baseline','dp_changed_from_baseline']:
                    if k in row and row[k] is not None and str(row[k]).strip()!='':
                        v = str(row[k]).strip().upper()
                        if v in ['YES','Y','1','TRUE']:
                            flipped = 1
                        else:
                            flipped = 0
                        break

                if flipped is None:
                    # fallback: compare llm_decision to baseline llm_decision per profile; we'll collect raw decisions and post-process
                    # store decision as temporary value
                    flipped = row.get('llm_decision') or row.get('decision') or ''

                if mid:
                    rows.setdefault(mid, []).append(flipped)
        model_data[name] = rows
        print(f"Read {len(rows)} modifiers from {path} as model '{name}'")
    return model_data


def write_human_csv(hmap, outpath):
    # collect all modifier ids
    mids = sorted({mid for (_,mid) in hmap.keys()})
    pids = sorted({pid for (pid,_) in hmap.keys()})
    with open(outpath,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['participant_id','modifier_id','flip'])
        for (pid,mid),flip in hmap.items():
            w.writerow([pid,mid,flip])
    print(f"Wrote human modifier flips to {outpath}")


def compute_kappas(hmap, model_data):
    # aggregate human mapping to per-modifier majority
    human_per_mod = defaultdict(list)
    for (pid,mid), hflip in hmap.items():
        human_per_mod[mid].append(hflip)

    human_major = {mid: (1 if sum(v)/len(v) >= 0.5 else 0) for mid,v in human_per_mod.items()}

    kappas = {}
    for mname, mrows in model_data.items():
        # mrows: mid -> list of flipped or decisions
        model_major = {}
        for mid, vals in mrows.items():
            # vals may be list of 1/0 or decision strings; normalize
            normalized = []
            for v in vals:
                if isinstance(v, (int,float)):
                    normalized.append(1 if int(v) else 0)
                else:
                    sv = str(v).strip().upper()
                    if sv in ['YES','Y','1','TRUE']:
                        normalized.append(1)
                    elif sv in ['NO','N','0','FALSE','']:
                        normalized.append(0)
                    else:
                        # assume decision tokens like A0/A1/TIE; treat TIE as 0 (no change), A0/A1 as change
                        if sv in ['TIE','BASELINE']:
                            normalized.append(0)
                        else:
                            normalized.append(1)
            if normalized:
                model_major[mid] = 1 if sum(normalized)/len(normalized) >= 0.5 else 0

        # compare on intersection of modifiers
        mids = sorted(set(human_major.keys()) & set(model_major.keys()))
        y1 = [human_major[mid] for mid in mids]
        y2 = [model_major[mid] for mid in mids]
        if not y1:
            kappas[mname] = None
            print(f"Model {mname}: no overlapping modifiers with human data")
            continue
        if cohen_kappa_score is None:
            acc = sum(1 for a,b in zip(y1,y2) if a==b)/len(y1)
            kappas[mname] = acc
        else:
            kappas[mname] = cohen_kappa_score(y1,y2)
        print(f"Model {mname}: compared {len(y1)} modifiers -> kappa {kappas[mname]}")
    return kappas


def write_kappa_outputs(kappas, out_csv, out_tex, out_png):
    with open(out_csv,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','kappa'])
        for m,v in sorted(kappas.items(), key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0)):
            w.writerow([m,'' if v is None else v])
    print(f"Wrote kappas to {out_csv}")

    # write simple tex table
    with open(out_tex,'w') as f:
        f.write('\\begin{tabular}{lr}\n')
        f.write('Model & Cohen\\' + 's kappa \\\n')
        f.write('\\hline\\n')
        for m,v in sorted(kappas.items(), key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0)):
            f.write(f"{m} & {'' if v is None else f'{v:.3f}'} \\\\n")
        f.write('\\end{tabular}\n')
    print(f"Wrote LaTeX table to {out_tex}")

    # plot
    try:
        import matplotlib.pyplot as plt
        names = []
        vals = []
        for m,v in sorted(kappas.items(), key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0)):
            names.append(m)
            vals.append(0 if v is None else v)
        plt.figure(figsize=(8,4))
        plt.bar(names, vals)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Cohen's kappa (or agreement)")
        plt.tight_layout()
        plt.savefig(out_png)
        print(f"Wrote kappa bar plot to {out_png}")
    except Exception as e:
        print("Plotting kappas failed:", e)


def main():
    mod_by_scenario = load_modifiers()
    # prefer refined human flips if available
    if os.path.exists(REFINED_HUMAN_FLIPS):
        print(f"Using refined human flips from {REFINED_HUMAN_FLIPS}")
        # read refined flips into hmap format
        hmap = {}
        with open(REFINED_HUMAN_FLIPS,newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                hmap[(row['participant_id'], row['modifier_id'])] = int(row['flip'])
    else:
        human_rows = read_human_rows()
        hmap = build_human_mapping(mod_by_scenario, human_rows)
        write_human_csv(hmap, os.path.join(OUT,'human_modifier_flips.csv'))
    model_data = read_master_model(MASTER_PATTERN)
    kappas = compute_kappas(hmap, model_data)
    write_kappa_outputs(kappas, os.path.join(OUT,'kappa_per_model.csv'), os.path.join(OUT,'kappa_table.tex'), os.path.join(OUT,'kappa_bar.png'))


if __name__ == '__main__':
    main()
import csv
import os
from collections import defaultdict

# paths
master_files = {
    'llama70': 'outputs/master_llm_decisions_llama_8B.csv',
    'gemma31': 'outputs/master_llm_decisions_gemma4.csv',
    'qwen32': 'outputs/master_llm_decisions_dotProduct.csv'  # placeholder; will adjust below
}
# find actual files
# use outputs/master_llm_decisions_*.csv
for fn in os.listdir('outputs'):
    if fn.startswith('master_llm_decisions_') and fn.endswith('.csv'):
        if 'gemma' in fn:
            master_files['gemma31'] = os.path.join('outputs',fn)
        elif 'llama_8B' in fn:
            master_files['llama70'] = os.path.join('outputs',fn)
        elif 'dotProduct' in fn:
            master_files['dot'] = os.path.join('outputs',fn)
        elif 'llama' in fn and '8B' not in fn:
            master_files['llama70'] = os.path.join('outputs',fn)

# Quick mapping of model keys to friendly names
friendly = {'llama70':'LLaMA-70B','gemma31':'Gemma-31B','qwen32':'Qwen-32B','dot':'Dot-product','llama8':'LLaMA-8B'}

# read human pairs computed earlier: we will use outputs/per_scenario_flip_rates.csv and human_study CSV for detailed pairs
human_pairs = defaultdict(lambda: {'base':None,'mods':[]})
import csv
with open('human_study/analysis_ready.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get('excluded') and row['excluded'].strip()!='0':
            continue
        pid = row['participant_id']
        theme = row['theme_id']
        sc = row['scenario_id']
        key = (pid, theme)
        is_mod = int(row['is_modified'])
        choice = row['choice'].strip()
        if is_mod==0:
            human_pairs[key]['base'] = choice
        else:
            human_pairs[key]['mods'].append((sc,choice))

# build human flip map at (scenario, modifier_id) level: for each modifier row (sc with modifier id appended), record flip yes/no
# in the master files, modifier_text looks like MOD_SC001_1_01 etc. We'll match by scenario and the modifier order.
human_flip = {}  # key: (scenario_id, modifier_text) -> 0/1
for (pid,theme),v in human_pairs.items():
    base = v['base']
    if base is None: continue
    for sc, choice in v['mods']:
        # we don't have modifier id in human csv; use a generic key (sc, index) approach instead: aggregate at (scenario,modifier-index)
        # simpler approach: aggregate flips at (scenario, modifier_text) is difficult without mapping; instead compute flip boolean per (scenario, modifier_type)
        pass

print('This script is a work-in-progress: mapping human modifier rows to model modifier ids requires the modifier ordering mapping. Aborting.')
