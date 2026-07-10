import csv
import os
import math
from collections import OrderedDict

csv_path = 'outputs/per_scenario_flip_rates.csv'
out_png = 'outputs/per_scenario_flip_rates.png'
latex_include = 'outputs/fig_per_scenario.tex'

rows = []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

scenarios = [r['scenario'] for r in rows]
dot = [float(r['dot_pct']) if r['dot_pct'] else float('nan') for r in rows]
llama70 = [float(r['llama70_pct']) if r['llama70_pct'] else float('nan') for r in rows]
gemma = [float(r['gemma_pct']) if r['gemma_pct'] else float('nan') for r in rows]
qwen = [float(r['qwen_pct']) if r['qwen_pct'] else float('nan') for r in rows]
human = [float(r['human_pct']) if r['human_pct'] else float('nan') for r in rows]

# compute three-LLM mean for display
llm_mean = []
for a,b,c in zip(llama70,gemma,qwen):
    vals = [v for v in (a,b,c) if not math.isnan(v)]
    llm_mean.append(sum(vals)/len(vals) if vals else float('nan'))

# try plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 17,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    # order scenarios by human flip rate (descending)
    order = sorted(range(len(scenarios)), key=lambda i: human[i] if not math.isnan(human[i]) else -1, reverse=True)
    scenarios_ord = [scenarios[i] for i in order]
    dot_ord = [dot[i] for i in order]
    llama70_ord = [llama70[i] for i in order]
    gemma_ord = [gemma[i] for i in order]
    qwen_ord = [qwen[i] for i in order]
    human_ord = [human[i] for i in order]
    pairs_ord = [int(rows[i]['human_pairs']) if rows[i]['human_pairs'] else 0 for i in order]

    x = np.arange(len(scenarios_ord))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 6.0))

    # dot-product as left-most thin bar
    ax.bar(x - 0.6, dot_ord, width=0.25, label='Dot-product', color='#4C78A8')

    # stacked LLMs: llama70 bottom, gemma middle, qwen top
    bottom = np.array([0.0]*len(x))
    p1 = ax.bar(x - 0.2, llama70_ord, width=0.4, bottom=bottom, label='LLaMA-70B', color='#F58518')
    bottom = bottom + np.array([v if not math.isnan(v) else 0.0 for v in llama70_ord])
    p2 = ax.bar(x - 0.2, gemma_ord, width=0.4, bottom=bottom, label='Gemma-31B', color='#54A24B')
    bottom = bottom + np.array([v if not math.isnan(v) else 0.0 for v in gemma_ord])
    p3 = ax.bar(x - 0.2, qwen_ord, width=0.4, bottom=bottom, label='Qwen-32B', color='#B279A2')

    # humans as right-most bars with error bars (binomial se)
    human_vals = np.array([v if not math.isnan(v) else 0.0 for v in human_ord]) / 100.0
    pairs = np.array(pairs_ord)
    # compute standard error for proportion: sqrt(p*(1-p)/n)
    se = np.sqrt(human_vals * (1-human_vals) / np.where(pairs>0, pairs, 1)) * 100.0
    human_pct = human_vals * 100.0
    ph = ax.bar(x + 0.6, human_pct, width=0.4, yerr=se, capsize=4, label='Humans', color='#E15759')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_ord, rotation=45, ha='right')
    ax.set_ylabel('Flip rate (%)')
    ax.set_ylim(0, max(max(dot_ord), max(llama70_ord), max(gemma_ord), max(qwen_ord), max(human_ord)) * 1.18)
    ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    # annotate with Spearman rho if present in CSV (we can recompute quickly)
    try:
        from scipy.stats import spearmanr
        import numpy as np
        llm_vals = [v for v in llm_mean if not math.isnan(v)]
        hum_vals = [v for v in human if not math.isnan(v)]
        # align: here we assume same length and order; compute only for positions where both exist
        L=[]; H=[]
        for lm,hm in zip(llm_mean,human):
            if not math.isnan(lm) and not math.isnan(hm):
                L.append(lm); H.append(hm)
        if L:
            rho,p = spearmanr(L,H)
            caption = f"Per-scenario flip rates. Spearman\'s $\rho$ = {rho:.3f}, p = {p:.3f}."
        else:
            caption = "Per-scenario flip rates."
    except Exception:
        # fallback: compute rho without p
        try:
            import numpy as np
            def rankdata(a):
                temp = sorted((val,i) for i,val in enumerate(a))
                ranks = [0]*len(a)
                i=0
                while i<len(temp):
                    j=i
                    while j<len(temp) and temp[j][0]==temp[i][0]:
                        j+=1
                    avg=(i+j-1)/2.0+1
                    for k in range(i,j):
                        ranks[temp[k][1]]=avg
                    i=j
                return ranks
            L=[]; H=[]
            for lm,hm in zip(llm_mean,human):
                if not math.isnan(lm) and not math.isnan(hm):
                    L.append(lm); H.append(hm)
            if L:
                rx=rankdata(L); ry=rankdata(H)
                rx=np.array(rx); ry=np.array(ry)
                xm=rx.mean(); ym=ry.mean()
                num=((rx-xm)*(ry-ym)).sum()
                den=((rx-xm)**2).sum()**0.5 * ((ry-ym)**2).sum()**0.5
                rho=num/den
                caption = f"Per-scenario flip rates. Spearman\'s $\\rho$ = {rho:.3f}."
            else:
                caption = "Per-scenario flip rates."
        except Exception:
            caption = "Per-scenario flip rates."

    # put caption text on the figure
    plt.subplots_adjust(bottom=0.25)
    # escape dollar signs and backslashes so matplotlib doesn't try to parse TeX math
    caption_display = caption.replace('\\', '\\\\').replace('$', '\\$')
    fig.text(0.01, 0.01, caption_display, fontsize=11)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print('Wrote', out_png)
    # write a LaTeX include snippet (path: outputs/fig_per_scenario.tex)
    caption = caption if 'caption' in locals() else 'Per-scenario flip rates.'
    with open(latex_include,'w') as f:
        f.write('\\begin{figure}[t]\n')
        f.write('\\centering\n')
        f.write('\\includegraphics[width=0.95\\linewidth]{per_scenario_flip_rates.png}\n')
        f.write('\\caption{')
        f.write(caption.replace('%','\\%'))
        f.write('}\\n')
        f.write('\\label{fig:per_scenario_flip_rates}\n')
        f.write('\\end{figure}\n')
    print('Wrote', latex_include)
except Exception as e:
    print('Plotting failed:', e)
    # fallback: write a small text summary
    with open('outputs/per_scenario_flip_rates.txt','w') as f:
        f.write('Scenarios:\n')
        for s,d,l,g,q,h in zip(scenarios,dot,llama70,gemma,qwen,human):
            f.write(f"{s}: dot={d}, llama70={l}, gemma={g}, qwen={q}, human={h}\n")
    print('Wrote outputs/per_scenario_flip_rates.txt')
