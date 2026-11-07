"""Independent scalar evaluation of every plotted prediction from saved weights."""
from pathlib import Path
import csv,json,math
import numpy as np
R=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent

def rows(name):
 with (R/name).open() as f:return list(csv.DictReader(f))
weights={}
for p in (R/'focus/iterative/trajectories').glob('*.json'):
 d=json.loads(p.read_text())
 if d['task'] in ['square','sin','tanh'] and d['seed']==164 and d['arm']=='examples':
  weights[d['task'],'examples']=d['states']['16']['initial']
  weights[d['task'],'iter_examples']=d['states']['16']['weights']
with np.load(R/'focus/lbfgs/weights.npz') as fit:
 for t in ['square','sin','tanh']:weights[t,'gd_lbfgs']=fit[f'{t}_16_164_gd_lbfgs'].tolist()
checks=0;maxdiff=0.
for row in rows('focus/main_visuals/prediction_curves.csv'):
 assert int(row['size'])==16 and int(row['seed'])==164
 x=float(row['x']);task=row['task']
 target={'square':lambda x:x*x,'sin':math.sin,'tanh':math.tanh}[task](x)
 assert math.isclose(target,float(row['target']),rel_tol=1e-14,abs_tol=1e-15)
 for method in ['examples','iter_examples','gd_lbfgs']:
  p=weights[task,method];assert len(p)==50
  prediction=p[0]*x+p[1]+sum(p[34+i]*max(0,p[2+i]*x+p[18+i]) for i in range(16))
  y=float(row[method]);delta=abs(prediction-y);maxdiff=max(maxdiff,delta)
  assert math.isclose(prediction,y,rel_tol=1e-12,abs_tol=1e-12),(task,method,x,prediction,y)
  checks+=1
for task in ['square','sin','tanh']:
 original=np.random.default_rng(164+442).uniform(-3,3,12)
 observed=[r for r in rows('focus/main_visuals/observed_points.csv') if r['task']==task]
 assert len(observed)==12
 for x,r in zip(original,observed): assert x==float(r['x'])
report=dict(curve_prediction_values_checked=checks,target_values_checked=3003,training_coordinates_checked=36,max_rounding_difference_from_scalar_accumulation=maxdiff,weights='Archived original, retained iterative, and gradient+L-BFGS weights; no imported prediction helper',visual_review='Clear axes, training/dev shading, all methods distinguished, no clipping',interpretation='Three illustrative regressions, not aggregate rank or representative median')
(H/'curve_review.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
