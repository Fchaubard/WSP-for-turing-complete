"""Aggregate the precommitted solver comparisons, preserving per-fixture pairings."""
import csv
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
methods=['xavier','adam800','adam_bfgs','ls_xavier','ls_adam_bfgs','examples','ls_examples','formula','ls_formula']
rows=list(csv.DictReader((HERE/'results.csv').open()))
index={(r['task'],r['size'],r['seed'],r['method']):r for r in rows}
summary=[]
for m in methods:
    group=[r for r in rows if r['method']==m]
    assert len(group)==90
    hinge=[r for r in group if r['certificate'] in ('affine','observed_knot','interior_intersection')]
    summary.append(dict(method=m,n=len(group),train_zero=sum(float(r['train_loss'])==0 for r in group),
                        train_near_best=sum(float(r['gap_to_best'])<=1e-10 for r in group),
                        hinge_certified_near=sum(abs(float(r['train_loss'])-float(r['real_optimum']))<=1e-10 for r in hinge),
                        median_loss=median(float(r['train_loss'])for r in group),
                        shell_all_exact=sum(float(r['outer_exact'])==1 for r in group),
                        dev_all_exact=sum(float(r['dev_exact'])==1 for r in group),
                        residual_evaluations=sum(int(r['nfev'])for r in group),
                        gradient_or_jacobian_evaluations=sum(int(r['njev'])for r in group)))
pairs={}
for a,b in [('adam800','adam_bfgs'),('bp_archived','adam_bfgs'),('adam_bfgs','ls_adam_bfgs'),
            ('formula','ls_formula'),('examples','ls_examples')]:
    matched=[(index[(r['task'],r['size'],r['seed'],a)],r) for r in rows if r['method']==b]
    pairs[f'{a}->{b}']={}
    for metric in ('train_loss','dev_scaled','outer_scaled'):
        delta=[float(y[metric])-float(x[metric]) for x,y in matched]
        pairs[f'{a}->{b}'][metric]=dict(improved=sum(d < -1e-10 for d in delta),worsened=sum(d > 1e-10 for d in delta),
                                               largest_absolute_change=max(map(abs,delta)))
(HERE/'summary.json').write_text(json.dumps(dict(methods=summary,paired=pairs),indent=2)+'\n')
for r in summary:print(r)
print(pairs)
