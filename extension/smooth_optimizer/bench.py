"""Fit all coefficients in the exact exported affine/ReLU self-power or tanh graph."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import gmpy2 as mp
import numpy as np

ROOT=Path(__file__).resolve().parent
SCALAR=ROOT.parent/'scalar'
sys.path.insert(0,str(SCALAR))
from train import backward
from run import samples,ulps

def package(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_graph(task):
    if task=='tanh':
        return package(ROOT.parent/'tanh'/'infer.py','frozen_tanh_infer').load()
    return package(SCALAR/'infer.py','frozen_selfpower_infer').load('selfpower')

def reference(x,task,bits):
    with mp.context(precision=bits):
        return np.asarray([float(mp.tanh(mp.mpfr(float(a))) if task=='tanh' else mp.mpfr(float(a))**mp.mpfr(float(a))) for a in x])

def measure(g,p,x,y):
    with np.errstate(all='ignore'):
        pred=g.run(x,p)
        finite=np.isfinite(pred)
        loss=np.mean((pred-y)**2)
        mae=np.mean(abs(pred-y))
        delta=ulps(pred,y)
    return dict(n=len(x),finite=int(finite.sum()),exact=int(np.sum(pred==y)),
                within1=int(np.sum(finite&(delta<=1))),
                mse=float(loss) if np.isfinite(loss) else None,
                mse_status='finite' if np.isfinite(loss) else 'nan' if np.isnan(loss) else 'overflow',
                mae=float(mae) if np.isfinite(mae) else None)

def number(v):
    return float(v) if np.isfinite(v) else None

def run(task):
    g,w=load_graph(task)
    sets=samples(task)
    x,y=sets['train'],reference(sets['train'],task,53)
    sets['outer']=np.r_[sets['left'],sets['right']] if task=='tanh' else sets['right']
    labels={k:(a,reference(a,task,53)) for k,a in sets.items() if k in ('train','inner','outer')}
    for a,b in labels.values():assert np.array_equal(b,reference(a,task,200))
    scores=lambda p:{k:measure(g,p,*xy) for k,xy in labels.items()}
    rows=[]
    traces=[]
    arrays={'written':w}
    rows.append(dict(task=task,method='written',slots=len(w),scores=scores(w)))
    for seed in (11,22,33):
        rng=np.random.default_rng(seed)
        p=rng.uniform(-np.sqrt(3),np.sqrt(3),len(w))
        initial=p.copy()
        initial_scores=scores(p)
        first=np.zeros_like(p)
        second=np.zeros_like(p)
        active=np.zeros(p.size,bool)
        max_grad=0.
        completed=0
        failure=None
        for step in range(1,101):
            idx=rng.integers(len(x),size=32)
            with np.errstate(all='ignore'):
                loss,grad=backward(g,p,x[idx],y[idx])
                maxabs=np.max(np.abs(grad))
                valid=np.isfinite(grad)
                active|=(grad!=0)&valid
                if valid.any():max_grad=max(max_grad,float(np.max(np.abs(grad[valid]))))
                traces.append(dict(task=task,seed=seed,step=step,loss=number(loss),
                                   loss_finite=bool(np.isfinite(loss)),
                                   finite_gradient=int(valid.sum()),nonzero_finite_gradient=int(np.count_nonzero(valid & (grad!=0))),
                                   max_abs_gradient=number(maxabs)))
                if not np.isfinite(loss) or not valid.all():
                    failure=dict(step=step,reason='nonfinite_loss_or_gradient',nonfinite_gradient=int(np.sum(~valid)),
                                 finite_loss=bool(np.isfinite(loss)))
                    break
                first=.9*first+.1*grad
                second=.999*second+.001*grad**2
                update=.01*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
                next_p=p-update
                if not np.isfinite(second).all() or not np.isfinite(next_p).all():
                    failure=dict(step=step,reason='nonfinite_adam_state_or_update',nonfinite_second=int(np.sum(~np.isfinite(second))),
                                 nonfinite_proposal=int(np.sum(~np.isfinite(next_p))))
                    break
                p=next_p
                completed=step
        end=scores(p)
        rows.append(dict(task=task,method='all_slot_adam',slots=len(w),seed=seed,completed_updates=completed,
                         initial=initial_scores,final=end,first_numerical_failure=failure,
                         moved_slots=int(np.count_nonzero(p!=initial)),ever_nonzero_finite_grad=int(active.sum()),
                         max_abs_finite_gradient=max_grad))
        arrays[f'initial_{seed}']=initial
        arrays[f'terminal_{seed}']=p
        print(task,seed,completed,failure,end['train'],end['outer'],flush=True)
    (ROOT/f'{task}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (ROOT/f'{task}_steps.json').write_text(json.dumps(traces,separators=(',',':'),allow_nan=False)+'\n')
    np.savez_compressed(ROOT/f'{task}_weights.npz',**arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['tanh','selfpower'])
    run(p.parse_args().task)
