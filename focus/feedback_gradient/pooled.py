"""Independent enlarged-label GD control; fit and held-out scoring are separate commands."""
import csv, hashlib, importlib.util, json, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
gd=module('old_gradient',ROOT/'focus/lbfgs/experiment.py')
feedback=module('model_feedback',ROOT/'focus/iterative/run.py')

def cases():
    manifests=[json.loads((ROOT/'focus'/name/'manifest.json').read_text()) for name in ('objective','arithmetic')]
    return sorted({(r['task'],int(w),int(r['seed'])) for group in manifests for r in group for w in r['sizes']})
def one(case):
    with np.errstate(all='ignore'):
        task,w,seed=case
        obs=feedback.central(task,seed)
        x,y=(np.concatenate([obs['train'][i],obs['dev'][i]],axis=0) for i in range(2))
        if task in gd.SERIES: assert np.array_equal(x,np.arange(17))
        if task in gd.arithmetic.TASKS:
            start=gd.arithmetic.initial(w,seed)
            f=lambda p:gd.reverse_loss(w,p,x,y)
        else:
            start=gd.initial(task,w,seed); f=gd.objective(task,w,start,x,y)
        began=time.monotonic()
        p,d1=gd.gradient_descent(f,start); q,d2=gd.lbfgs(f,p,'pooled_gd')
        assert np.isfinite([*p,*q]).all() and f(q)[0]<=f(p)[0]<=f(start)[0]
        return task,w,seed,p,q,dict(seconds=time.monotonic()-began,gd=d1,gd_lbfgs=d2,n=len(x))
def fit():
    pool_cases=cases(); assert len(pool_cases)==87
    weights={}; diags={}
    with ProcessPoolExecutor(max_workers=4) as procs:
        for t,w,s,p,q,d in procs.map(one,pool_cases):
            key=f'{t}_{w}_{s}'
            weights[key+'_pool_gd']=p;weights[key+'_pool_gd_lbfgs']=q;diags[key]=d
    np.savez_compressed(HERE/'weights.npz',**weights)
    (HERE/'fit.json').write_text(json.dumps(diags,indent=1)+'\n')
    print('Fitted',len(diags),'settings; weights SHA-256',hashlib.sha256((HERE/'weights.npz').read_bytes()).hexdigest())
def evaluate():
    assert (ROOT/'focus/iterative/FROZEN.json').is_file()
    rows=[]
    with np.load(HERE/'weights.npz') as z:
        for t,w,s in cases():
            data=feedback.central(t,s)
            tests=gd.inputs(t)
            if t in gd.SERIES: inner=np.arange(17,33)
            else: inner=tests['inner']
            for m in ('pool_gd','pool_gd_lbfgs'):
                p=z[f'{t}_{w}_{s}_{m}']
                metrics=feedback.measure(t,w,p,dict(train=data['train'],feedback=data['dev'],dev=(inner,gd.reference(t,inner)),outer=(tests['outer'],gd.reference(t,tests['outer']))))
                assert metrics is not None,(t,w,s,m)
                for spl,score in metrics.items():
                    rows.append(dict(task=t,size=w,params=len(p),seed=s,method=m,split=spl,n_examples=score['n'],exact_count=score['exact'],exact_fraction=score['exact']/score['n'],raw_rmse=score['rmse'],split_loss=score['loss'],train_loss=metrics['train']['loss']))
    with (HERE/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(len(rows),'score rows')
if __name__=='__main__':
    if sys.argv[1]=='fit': fit()
    elif sys.argv[1]=='evaluate': evaluate()
