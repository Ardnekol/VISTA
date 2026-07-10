#!/usr/bin/env python3
"""Compute Cohen's kappa per participant between human flips and model decisions.

Writes:
 - outputs/kappa_per_participant.csv (participant, model, kappa)
 - outputs/kappa_per_model_summary.csv (model, mean_kappa, n_participants)
 - updates outputs/kappa_table.tex with mean kappa per model
"""
import csv, os
from collections import defaultdict
import math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
HUMAN = os.path.join(OUT, 'human_modifier_flips_refined.csv')
MASTER_PATTERN = os.path.join(OUT, 'master_llm_decisions_*.csv')
from glob import glob

def load_human():
    # returns dict participant -> {modifier: flip}
    d = defaultdict(dict)
    if not os.path.exists(HUMAN):
        return d
    with open(HUMAN) as f:
        r = csv.DictReader(f)
        for row in r:
            d[row['participant_id']][row['modifier_id']] = int(row['flip'])
    return d

def load_models():
    # returns dict model -> modifier -> majority_flip
    model_flips = defaultdict(lambda: defaultdict(list))
    for path in glob(MASTER_PATTERN):
        name = os.path.basename(path).replace('master_llm_decisions_','').replace('.csv','')
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                mid = row.get('condition')
                if not mid or not mid.startswith('MOD_SC'):
                    continue
                val = None
                for k in ['llm_changed_from_baseline','changed_from_baseline','dp_changed_from_baseline']:
                    if k in row and row[k] not in [None,'']:
                        v = str(row[k]).strip().upper()
                        val = 1 if v in ['YES','Y','1','TRUE'] else 0
                        break
                if val is None:
                    dec = (row.get('llm_decision') or row.get('decision') or '').upper()
                    if dec in ['TIE','BASELINE','']:
                        val = 0
                    else:
                        val = 1
                model_flips[name][mid].append(val)
    # majority
    model_major = defaultdict(dict)
    for m,d in model_flips.items():
        for mid, vals in d.items():
            model_major[m][mid] = 1 if sum(vals)/len(vals) >= 0.5 else 0
    return model_major

def cohen_kappa(a,b):
    # a and b are lists of 0/1, same length
    if len(a)==0:
        return float('nan')
    n = len(a)
    agree = sum(1 for x,y in zip(a,b) if x==y)
    p0 = agree / n
    pa = sum(a)/n
    pb = sum(b)/n
    pe = pa*pb + (1-pa)*(1-pb)
    if pe==1:
        return 0.0
    return (p0 - pe) / (1 - pe)

def main():
    human = load_human()
    models = load_models()
    # For each participant and model, align on modifiers present in both
    rows = []
    per_model = defaultdict(list)
    for pid, hmap in human.items():
        for mname, mdict in models.items():
            # intersection of modifiers
            mids = set(hmap.keys()).intersection(set(mdict.keys()))
            if not mids:
                continue
            a = [hmap[mid] for mid in sorted(mids)]
            b = [mdict[mid] for mid in sorted(mids)]
            k = cohen_kappa(a,b)
            if k!=k: # nan
                continue
            rows.append((pid,mname,k,len(mids)))
            per_model[mname].append(k)

    # write per-participant
    with open(os.path.join(OUT,'kappa_per_participant.csv'),'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['participant_id','model','kappa','n_modifiers'])
        for pid,m,k,nm in rows:
            w.writerow([pid,m,f'{k:.3f}',nm])

    # summary per model
    summary = []
    for m,ks in per_model.items():
        if not ks:
            continue
        mean = sum(ks)/len(ks)
        summary.append((m,mean,len(ks)))

    with open(os.path.join(OUT,'kappa_per_model_summary.csv'),'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','mean_kappa','n_participants'])
        for m,mean,n in summary:
            w.writerow([m,f'{mean:.3f}',n])

    # update kappa_table.tex with mean kappas (simple replacement)
    texp = os.path.join(OUT,'kappa_table.tex')
    try:
        lines = ['\\begin{tabular}{lr}\\n','Model & Cohen\\s kappa \\\\n+\\hline\\n']
        for m,mean,n in summary:
            lines.append(f"{m} & {mean:.3f} \\\\n+")
        lines.append('\\end{tabular}\\n')
        with open(texp,'w') as f:
            f.writelines(lines)
    except Exception as e:
        print('Could not update kappa_table.tex:',e)

    print('Wrote kappa per participant and summary')

if __name__=='__main__':
    main()
