"""Aggregate both supervised budgets. Optionally render the saved curves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ('ID','range','length','joint')
TASKS = ('sum','fib','tri','sort')
CELLS = ('rnn','gru')


def summarize():
    rows = []
    for path in [ROOT/'results.json',ROOT/'longer'/'results.json']:
        data = json.loads(path.read_text())
        models = [r for r in data['rows'] if r['method']=='bp']
        for task in TASKS:
            for cell in CELLS:
                group = [r for r in models if (r['task'],r['cell'])==(task,cell)]
                for d in sorted(set(r['width'] for r in group)):
                    runs = [r for r in group if r['width']==d]
                    for name in SPLITS:
                        row = dict(adam_steps=data['settings']['steps'],task=task,cell=cell,width=d,
                                   params=runs[0]['params'],seeds=len(runs),split=name)
                        for metric in ('nrmse','rmse','exact','exact_all','one_ulp_all','eps8_all'):
                            a = [r['metrics'][name][metric] for r in runs]
                            for fn,stat in [(np.median,'median'),(np.min,'min'),(np.max,'max')]:
                                row[metric+'_'+stat] = fn(a)
                        row['median_dev_loss'] = np.median([r['fit']['best_validation_loss'] for r in runs])
                        rows.append(row)
    with (ROOT/'summary.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    rows = [r for r in rows if r['adam_steps']==5000]
    fig,axes = plt.subplots(2,2,figsize=(9.,6.3),sharey=False,layout='constrained')
    names = dict(sum='Running sum',fib='Seeded Fibonacci',tri='Seeded triangular sequence',sort='Recurrent sorting')
    colors = dict(rnn='#22588f',gru='#b45d1a')
    for ax,task in zip(axes.ravel(),TASKS):
        for cell in CELLS:
            for split,ls in [('ID','--'),('joint','-')]:
                rs = [r for r in rows if (r['task'],r['cell'],r['split'])==(task,cell,split)]
                x = [r['width'] for r in rs]
                y = [r['nrmse_median'] for r in rs]
                ax.plot(x,y,ls=ls,marker='o',ms=3.5,lw=1.8,c=colors[cell],
                        label=('tanh RNN' if cell=='rnn' else 'GRU')+' '+split)
        ticks = [1,2,4,8]+([16] if task=='sort' else [])
        ax.set_xscale('log',base=2)
        ax.set_xticks(ticks,labels=[str(d) for d in ticks])
        ax.set_ylim(-.025,1.045)
        ax.set_title(names[task],loc='left',weight='bold',size=11)
        ax.grid(axis='y',alpha=.20)
        ax.spines[['right','top']].set_visible(False)
    for ax in axes[:,0]:
        ax.set_ylabel('NRMSE')
    for ax in axes[-1,:]:
        ax.set_xlabel('Hidden-state width')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='outside lower center',frameon=False)
    fig.savefig(ROOT/'frontier.png',dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figure',action='store_true')
    args = ap.parse_args()
    rows = summarize()
    if args.figure:
        figure(rows)
