import re
import csv
from collections import defaultdict, OrderedDict
import os

# paths to system reports
reports = {
    'dot': 'outputs/analysis_report_dotProduct.txt',
    'llama70': 'outputs/analysis_report_llama.txt',
    'gemma31': 'outputs/analysis_report_gemma4.txt',
    'qwen32': 'outputs/analysis_report_qwen.txt',
    'llama8': 'outputs/analysis_report_llama_8B.txt',
}

scenario_order = []
# parse scenario sections like: "SC003_1      27.4%    208/760"
pattern = re.compile(r'^\s*(SC\d{3}_\d)\s+(\d+\.\d+)%\s+(\d+)/(\d+)', re.MULTILINE)

per_sys = {k: {} for k in reports}
for key, path in reports.items():
    if not os.path.exists(path):
        continue
    with open(path) as f:
        txt = f.read()
    for m in pattern.finditer(txt):
        sc, pct, n, denom = m.groups()
        per_sys[key][sc] = {'pct': float(pct), 'n': int(n), 'denom': int(denom)}
        if sc not in scenario_order:
            scenario_order.append(sc)

# sort scenario_order by name
scenario_order = sorted(scenario_order)

# compute human per-scenario flips
human_file = 'human_study/analysis_ready.csv'
# for each participant and scenario, find base and modified choices (exclude excluded==1)
# Group human responses by (participant_id, theme_id).
# baseline rows have is_modified==0 and give the base choice for the theme;
# modifier rows have is_modified==1 and a scenario_id indicating the modified scenario.
pairs = defaultdict(lambda: {'base': None, 'mods': []})
with open(human_file, newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        if r.get('excluded') and r['excluded'].strip()!='0':
            continue
        pid = r['participant_id']
        theme = r['theme_id']
        sc = r['scenario_id']
        key = (pid, theme)
        is_mod = int(r['is_modified'])
        choice = r['choice'].strip()
        if is_mod == 0:
            pairs[key]['base'] = choice
        else:
            # store the modifier scenario id along with the choice
            pairs[key]['mods'].append((sc, choice))

human_by_scenario = defaultdict(lambda: {'pairs':0,'flips':0})
for (pid, theme), v in pairs.items():
    base = v['base']
    mods = v['mods']
    if base is None or not mods:
        continue
    for mod_sc, mod_choice in mods:
        human_by_scenario[mod_sc]['pairs'] += 1
        if mod_choice != base:
            human_by_scenario[mod_sc]['flips'] += 1

# prepare table rows
rows = []
for sc in scenario_order:
    dot = per_sys.get('dot',{}).get(sc, {'pct': None, 'n': None, 'denom': None})
    l70 = per_sys.get('llama70',{}).get(sc, {'pct': None})
    gem = per_sys.get('gemma31',{}).get(sc, {'pct': None})
    qwen = per_sys.get('qwen32',{}).get(sc, {'pct': None})
    l8 = per_sys.get('llama8',{}).get(sc, {'pct': None})
    # mean across three LLMs (llama70, gemma31, qwen32)
    vals = [v for v in [l70.get('pct'), gem.get('pct'), qwen.get('pct')] if v is not None]
    llm_mean = sum(vals)/len(vals) if vals else None
    human = human_by_scenario.get(sc, {'pairs':0,'flips':0})
    human_pairs = human['pairs']
    human_flips = human['flips']
    human_pct = (human_flips/human_pairs*100) if human_pairs else None
    rows.append({'scenario':sc,'dot_pct':dot['pct'],'llama70_pct':l70.get('pct'),'gemma_pct':gem.get('pct'),'qwen_pct':qwen.get('pct'),'llm_mean_pct':llm_mean,'human_pct':human_pct,'human_pairs':human_pairs})

# compute Spearman between llm_mean_pct and human_pct
llm_vals = []
human_vals = []
for r in rows:
    if r['llm_mean_pct'] is None or r['human_pct'] is None:
        continue
    llm_vals.append(r['llm_mean_pct'])
    human_vals.append(r['human_pct'])

rho = None
pval = None
if llm_vals:
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(llm_vals, human_vals)
    except Exception:
        # fallback compute Spearman rho using ranks and Pearson
        try:
            import numpy as np
            def rankdata(a):
                temp = sorted((val,i) for i,val in enumerate(a))
                ranks = [0]*len(a)
                i=0
                while i<len(temp):
                    j=i
                    s=0
                    while j<len(temp) and temp[j][0]==temp[i][0]:
                        s+=1
                        j+=1
                    avg = (i+j-1)/2.0+1
                    for k in range(i,j):
                        ranks[temp[k][1]] = avg
                    i=j
                return ranks
            rx = rankdata(llm_vals)
            ry = rankdata(human_vals)
            rx = np.array(rx); ry = np.array(ry)
            xm = rx.mean(); ym = ry.mean()
            num = ((rx-xm)*(ry-ym)).sum()
            den = ((rx-xm)**2).sum()**0.5 * ((ry-ym)**2).sum()**0.5
            rho = num/den
        except Exception:
            rho = None

# print LaTeX table
print('%% LaTeX table: per-scenario flip rates')
print('\\begin{table}[t]')
print('\\centering')
print('\\caption{Per-scenario flip rates across systems.}')
print('\\label{tab:per_scenario_flip_rates}')
print('\\begin{tabular}{lrrrrr}')
print('\\toprule')
print('Scenario & Dot-product & LLaMA-70B & Gemma-31B & Qwen-32B & Humans \\\\')
print(' & rate (\%) & rate (\%) & rate (\%) & rate (\%) & rate (\%) \\\\')
print('\\midrule')
for r in rows:
    sc = r['scenario']
    def fmt(x):
        return f"{x:.1f}" if x is not None else '---'
    row = (
        f"{sc} & {fmt(r['dot_pct'])} & {fmt(r['llama70_pct'])} "
        f"& {fmt(r['gemma_pct'])} & {fmt(r['qwen_pct'])} & {fmt(r['human_pct'])} \\\\"
    )
    print(row)
print('\\bottomrule')
print('\\end{tabular}')
if rho is not None:
    pstr = f", p = {pval:.3f}" if pval is not None else ''
    print(f"\\vspace{{2mm}}\\par\nSpearman\\'s $\rho$ (three-LLM mean vs humans) = {rho:.3f}{pstr}.")
print('\\end{table}')

# also print a simple CSV for convenience
out_csv = 'outputs/per_scenario_flip_rates.csv'
with open(out_csv,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['scenario','dot_pct','llama70_pct','gemma_pct','qwen_pct','llm_mean_pct','human_pct','human_pairs'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'Wrote {out_csv}')
