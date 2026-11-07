"""Fit exactly the displayed examples with BP, and score literal or baseline arrays."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from design import HERE, SMOOTH, SERIES, SIZES, TASKS, SEEDS, observed, initial, predict, reference, inputs

ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from error_metrics import rmse
sys.path.insert(0,str(ROOT/'scalar'))
from benchmark import hinge
sys.path.insert(0,str(ROOT/'sequence'))
from run import Recurrence, fit

def objective(task,size,p0,x,y):
    scale=np.maximum(1,np.abs(y))
    rec=Recurrence(task,size) if size and task in ('sum',*SERIES) else None
    def lossgrad(p):
        if task in SMOOTH: yh=hinge(x,p)
        elif rec:
            xx=x if task=='sum' else np.zeros((1,len(x)))
            yh,hidden=rec.forward(p,xx)
            if task in SERIES: yh=yh[0]
        elif task=='sort':
            h=x; cache=[]
            for w,v in p.reshape(size,2,4):
                gate=np.maximum(h@w,0.)
                cache.append((h,gate))
                h=h+gate[:,None]*v
            yh=h
        else: yh=predict(task,size,p,x)
        error=(yh-y)/scale
        adj=error/(y.size*scale)
        if task in SMOOTH: grad=hinge(x,p,adj)
        elif rec:
            grad=rec.backward(p,xx,hidden,adj[None,:] if task in SERIES else adj)
        elif task=='sort':
            grad=np.zeros((size,2,4))
            for i in range(size-1,-1,-1):
                h,gate=cache[i]; w,v=p.reshape(size,2,4)[i]
                dg=(adj@v)*(gate>0)
                grad[i,1]=gate@adj; grad[i,0]=h.T@dg
                adj=adj+dg[:,None]*w
            grad=grad.ravel()
        elif task=='sum': grad=np.array([np.sum(adj*x)])
        else:
            grad=x[:,:min(size,2)].T@adj
            if size==3:grad=np.r_[grad,adj.sum()]
        return float(.5*np.mean(error**2)),grad
    numerical=[]; delta=1e-6
    for u in np.eye(len(p0)):
        numerical.append((lossgrad(p0+delta*u)[0]-lossgrad(p0-delta*u)[0])/(2*delta))
    analytic=lossgrad(p0)[1]
    diff=np.max(np.abs(numerical-analytic)/np.maximum(1,np.abs(numerical)))
    assert diff < 3e-5,(task,size,diff)
    return lossgrad

def fit_baselines():
    rows=[]
    for task in TASKS:
        for seed in SEEDS:
            x,y=observed(task,seed)
            for size in SIZES[task]:
                p0=initial(task,size,seed)
                fun=objective(task,size,p0,x,y)
                p,info=fit(fun,p0,800)
                rows.append(dict(task=task,seed=seed,size=size,weights=p.tolist(),fit=info))
    (HERE/'bp.json').write_text(json.dumps(rows,indent=1)+'\n')

def parse(text):
    raw=text.strip()
    if raw.startswith(chr(96)*3): raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
    return json.loads(raw)

def score(task,seed,size,p,arm):
    x,y=observed(task,seed)
    splits={'observed':x,**inputs(task)}
    rows=[]
    for split,x in splits.items():
        y=reference(task,x)
        with np.errstate(over='ignore',invalid='ignore'):
            yh=predict(task,size,p,x)
            ea=np.abs(yh-y)
            es=ea/np.maximum(1,np.abs(y))
            exact=(yh==y);ulp=(yh>=np.nextafter(y,-np.inf))&(yh<=np.nextafter(y,np.inf))
            tol=es<=1e-8
        def whole(a):
            return float((a.all(axis=1) if a.ndim==2 else a).mean())
        rows.append(dict(task=task,seed=seed,arm=arm,size=size,slots=len(p),split=split,n=len(x),
                         exact=whole(exact),rmse=rmse(yh,y),ulp1=whole(ulp),within_1e_minus_8=whole(tol),finite=float(np.isfinite(yh).mean()),
                         mae=float(ea.mean()) if np.isfinite(ea).all() else None,
                         max_scaled=float(es.max()) if np.isfinite(es).all() else None,
                         mean_scaled=float(es.mean()) if np.isfinite(es).all() else None))
    return rows

def score_all():
    rows=[];status=[]
    manifest=json.loads((HERE/'manifest.json').read_text())
    for case in manifest:
        path=HERE/'trials'/(case['id']+'.reply.txt')
        out=None
        if path.exists():
            try:out=parse(path.read_text())
            except (ValueError,IndexError):pass
        for j,size in enumerate(case['sizes']):
            task,seed,arm=case['task'],case['seed'],case['arm']
            p0=initial(task,size,seed)
            valid=False
            if isinstance(out,list) and len(out)==3:
                try:
                    p=np.asarray(out[j],dtype=float)
                    valid=p.shape==p0.shape and np.isfinite(p).all()
                except (TypeError,ValueError):pass
            status.append(dict(id=case['id'],task=task,seed=seed,arm=arm,size=size,
                               status='valid' if valid else ('malformed' if path.exists() else 'missing')))
            if valid:rows+=score(task,seed,size,p,arm)
    for r in json.loads((HERE/'bp.json').read_text()):
        task,seed,size=r['task'],r['seed'],r['size']
        for method,p in [('bp',r['weights']),('xavier',initial(task,size,seed))]:
            rows+=score(task,seed,size,np.asarray(p),method)
    for name,table in [('scores',rows),('status',status)]:
        with (HERE/(name+'.csv')).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=table[0]);wr.writeheader();wr.writerows(table)
    from collections import Counter
    print(Counter(r['status'] for r in status))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score'])
    if p.parse_args().action=='fit':fit_baselines()
    else:score_all()
