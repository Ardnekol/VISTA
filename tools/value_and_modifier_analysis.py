#!/usr/bin/env python3
"""Produce value-dimension contribution table and modifier-type analysis.

Outputs:
 - outputs/value_axis_table.csv
 - outputs/value_axis_table.tex
 - outputs/modifier_type_table.csv
 - outputs/modifier_type_table.tex
 - outputs/modifier_type_plot.png

Uses:
 - outputs/step4_axis_coefficients.csv for axis coefficients
 - Dataset/modifiers_batch1.json for mapping modifiers -> axis
 - outputs/master_llm_decisions_*.csv for per-modifier flips aggregated by model
 - outputs/human_modifier_flips_refined.csv for human flips
"""

import csv, json, os
from collections import Counter, defaultdict
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

AXIS_FILE = os.path.join(OUT, 'step4_axis_coefficients.csv')
MOD_FILE = os.path.join(ROOT, 'Dataset', 'modifiers_batch1.json')
HUMAN_REFINED = os.path.join(OUT, 'human_modifier_flips_refined.csv')
MASTER_PATTERN = os.path.join(OUT, 'master_llm_decisions_*.csv')


def load_axis_table():
    axes = []
    if not os.path.exists(AXIS_FILE):
        return axes
    with open(AXIS_FILE) as f:
        r = csv.DictReader(f)
        for row in r:
            axes.append(row)
    return axes


def load_mods():
    with open(MOD_FILE) as f:
        mods = json.load(f)
    mid_to_axis = {}
    for s in mods:
        sc = s.get('scenario_id')
        for m in s.get('modifiers',[]):
            mid_to_axis[m['modifier_id']] = m.get('axis')
    return mid_to_axis


def load_human_flips():
    flips = defaultdict(list)
    if not os.path.exists(HUMAN_REFINED):
        return flips
    with open(HUMAN_REFINED) as f:
        r = csv.DictReader(f)
        for row in r:
            flips[row['modifier_id']].append(int(row['flip']))
    return flips


def load_model_flips():
    model_flips = defaultdict(lambda: defaultdict(list))
    for path in glob(MASTER_PATTERN):
        name = os.path.basename(path).replace('master_llm_decisions_','').replace('.csv','')
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                mid = row.get('condition')
                if not mid or not mid.startswith('MOD_SC'):
                    continue
                # prefer llm_changed_from_baseline
                val = None
                for k in ['llm_changed_from_baseline','changed_from_baseline','dp_changed_from_baseline']:
                    if k in row and row[k] not in [None,'']:
                        v = str(row[k]).strip().upper()
                        val = 1 if v in ['YES','Y','1','TRUE'] else 0
                        break
                if val is None:
                    # fallback: decision tokens
                    dec = (row.get('llm_decision') or row.get('decision') or '').upper()
                    if dec in ['TIE','BASELINE','']:
                        val = 0
                    else:
                        val = 1
                model_flips[name][mid].append(val)
    # convert lists to majority
    model_major = defaultdict(dict)
    for m, d in model_flips.items():
        for mid, vals in d.items():
            model_major[m][mid] = 1 if sum(vals)/len(vals) >= 0.5 else 0
    return model_major


def main():
    axes = load_axis_table()
    mid_to_axis = load_mods()
    human = load_human_flips()
    models = load_model_flips()

    # Value axis table: use axes file to pick top axes by abs(coef_logodds)
    axis_rows = []
    for r in axes:
        try:
            axis_rows.append((r['axis'], abs(float(r.get('coef_logodds',0))), r))
        except Exception:
            continue
    axis_rows.sort(key=lambda x: -x[1])
    top = axis_rows[:10]
    # write CSV and tex
    with open(os.path.join(OUT,'value_axis_table.csv'),'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['axis','abs_coef_logodds','model'])
        for a,score,r in top:
            w.writerow([a,score,r.get('model')])

    with open(os.path.join(OUT,'value_axis_table.tex'),'w') as f:
        f.write('\\begin{tabular}{lrl}\\n')
        f.write('Axis & |coef_logodds| & Model\\\\n')
        f.write('\\hline\\n')
        for a,score,r in top:
            f.write(f"{a} & {score:.3f} & {r.get('model')}\\\\n")
        f.write('\\end{tabular}\\n')

    # Modifier-type analysis: infer axis -> type from modifiers JSON
    # Heuristic rules:
    # - resource* -> stakes
    # - self* or time* -> personal-cost
    # - social*, in_out_group, authority_signal -> affective
    # - diffused*, competence* -> informational
    # Fallback -> 'other'
    type_map = {}
    # build a set of axes from modifiers JSON
    try:
        with open(MOD_FILE) as f:
            mods_all = json.load(f)
    except Exception:
        mods_all = []
    axes_set = set()
    for s in mods_all:
        for m in s.get('modifiers', []):
            axes_set.add(m.get('axis'))
    for ax in axes_set:
        if not ax:
            continue
        a = ax.lower()
        if 'resource' in a:
            type_map[ax] = 'stakes'
        elif a.startswith('self') or 'time' in a:
            type_map[ax] = 'personal-cost'
        elif 'social' in a or 'in_out' in a or 'authority' in a:
            type_map[ax] = 'affective'
        elif 'diffused' in a or 'competence' in a:
            type_map[ax] = 'informational'
        else:
            type_map[ax] = 'other'

    # aggregate flips per type
    human_type = defaultdict(list)
    for mid, vals in human.items():
        axis = mid_to_axis.get(mid)
        t = type_map.get(axis,'other')
        human_type[t].append(sum(vals)/len(vals))

    model_type = defaultdict(lambda: defaultdict(list))
    for mname, mdict in models.items():
        for mid, val in mdict.items():
            axis = mid_to_axis.get(mid)
            t = type_map.get(axis,'other')
            model_type[mname][t].append(val)

    # compute human average flip rate per type
    human_type_rate = {t: (sum(v)/len(v) if v else 0.0) for t,v in human_type.items()}

    # model rates
    model_type_rate = {}
    for mname, d in model_type.items():
        model_type_rate[mname] = {t: (sum(vals)/len(vals) if vals else 0.0) for t,vals in d.items()}

    # write CSV and tex
    types = sorted(set(list(human_type_rate.keys()) + [t for d in model_type_rate.values() for t in d.keys()]))
    with open(os.path.join(OUT,'modifier_type_table.csv'),'w',newline='') as f:
        w = csv.writer(f)
        header = ['type','human_rate'] + sorted(model_type_rate.keys())
        w.writerow(header)
        for t in types:
            row = [t, human_type_rate.get(t,0.0)]
            for m in sorted(model_type_rate.keys()):
                row.append(model_type_rate[m].get(t,0.0))
            w.writerow(row)

    with open(os.path.join(OUT,'modifier_type_table.tex'),'w') as f:
        f.write('\\begin{tabular}{l' + 'r'* (1+len(model_type_rate)) + '}\\n')
        f.write('Type & Human ')
        for m in sorted(model_type_rate.keys()): f.write(f" & {m}")
        f.write('\\\\n')
        f.write('\\hline\\n')
        for t in types:
            f.write(t)
            f.write(f" & {human_type_rate.get(t,0.0):.3f}")
            for m in sorted(model_type_rate.keys()):
                f.write(f" & {model_type_rate[m].get(t,0.0):.3f}")
            f.write('\\\\n')
        f.write('\\end{tabular}\\n')

    # simple plot for modifier types
    try:
        import matplotlib.pyplot as plt
        models_sorted = sorted(model_type_rate.keys())
        types_sorted = types
        x = range(len(types_sorted))
        plt.figure(figsize=(8,4))
        plt.bar([i-0.2 for i in x], [human_type_rate.get(t,0) for t in types_sorted], width=0.4, label='Human')
        for j,m in enumerate(models_sorted):
            vals = [model_type_rate[m].get(t,0) for t in types_sorted]
            plt.bar([i-0.2+ (j+1)*0.4/(len(models_sorted)+1) for i in x], vals, width=0.4/(len(models_sorted)+1), label=m)
        plt.xticks(x, types_sorted, rotation=45, ha='right')
        plt.ylabel('Flip rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,'modifier_type_plot.png'))
    except Exception as e:
        print('Could not produce modifier type plot:', e)

    print('Wrote value axis and modifier type outputs to outputs/')

if __name__=='__main__':
    main()
