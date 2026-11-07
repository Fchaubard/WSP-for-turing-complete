"""Matched full-batch GD and limited-memory BFGS, all 87 nonconvex fixtures."""
import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'extension/writer'), str(ROOT / 'focus/arithmetic'), str(ROOT)]
from design import SERIES, inputs, initial, observed, predict, reference
from compare import objective
from error_metrics import rmse
import fixtures as arithmetic
from fit import reverse_loss

ARMS = ('gd','gd_lbfgs','adam_lbfgs','lbfgs_xavier')
OPTIONS = dict(maxcor=10,maxiter=1000,maxfun=20000,maxls=40,gtol=1e-14,ftol=0.0)
ARMIJO = 1e-4
MAX_HALVINGS = 80
GD_UPDATES = 800


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def gradient_descent(lossgrad, start):
    p=start.copy()
    value,g=lossgrad(p)
    nfev,accepted,reductions=1,0,0
    minimum_rate, maximum_rate = 1., 0.
    reason='update budget'
    for _ in range(GD_UPDATES):
        if np.max(abs(g)) <= 1e-14:
            reason='gradient infinity norm at or below 1e-14'
            break
        squared=g @ g
        rate=1.
        for trials in range(MAX_HALVINGS+1):
            candidate=p-rate*g
            if np.array_equal(candidate,p):
                reason='no representable step'
                break
            new,grad=lossgrad(candidate)
            nfev+=1
            if (np.isfinite(new) and np.isfinite(grad).all()
                and new < value and new <= value-ARMIJO*rate*squared):
                p,value,g=candidate,new,grad
                minimum_rate=min(minimum_rate,rate)
                maximum_rate=max(maximum_rate,rate)
                accepted+=1
                break
            if trials<MAX_HALVINGS:
                rate*=.5
                reductions+=1
        else:
            reason='exhausted 80 backtracking halvings'
            break
        if reason=='no representable step':
            break
    return p,dict(start='xavier',iterations=accepted,joint_evals=nfev,termination=reason,
                  backtrack_halvings=reductions,min_accepted_step=minimum_rate if accepted else '',
                  max_accepted_step=maximum_rate if accepted else '',
                  final_gradient_max=float(np.max(abs(g))),selected_stage='GD' if accepted else 'start')


def lbfgs(lossgrad, start, start_name):
    best=start.copy()
    best_loss=float(lossgrad(best)[0])
    start_loss=best_loss
    accepted=0
    best_iteration=0
    def track(intermediate_result):
        # SciPy supplies the accepted iterate and its cached objective to this callback.
        nonlocal best, best_loss, accepted, best_iteration
        accepted+=1
        if np.isfinite(intermediate_result.fun) and intermediate_result.fun < best_loss:
            best,best_loss=intermediate_result.x.copy(),float(intermediate_result.fun)
            best_iteration=accepted
    fit=minimize(lossgrad,start.copy(),jac=True,method='L-BFGS-B',bounds=None,
                 callback=track,options=OPTIONS)
    # Some line-search failures return a function value from a rejected trial.
    # Re-evaluate the returned coordinates before considering this fallback.
    actual,gradient=lossgrad(fit.x)
    if np.isfinite(actual) and actual<best_loss:
        best,best_loss,best_iteration=fit.x.copy(),float(actual),int(fit.nit)
    if np.array_equal(best,start): stage='start'
    elif np.array_equal(best,fit.x): stage='L-BFGS final'
    else: stage='L-BFGS accepted intermediate'
    return best,dict(start=start_name,iterations=int(fit.nit),joint_evals=int(fit.nfev)+2,
                     solver_joint_evals=int(fit.nfev),termination=str(fit.message),status=int(fit.status),
                     success=bool(fit.success),accepted_callbacks=accepted,best_iteration=best_iteration,
                     start_loss=start_loss,return_loss=actual,selected_loss=best_loss,selected_stage=stage,
                     returned_gradient_max=float(np.max(abs(gradient))))


def adam_stage(lossgrad, start):
    theta=start.copy()
    first=np.zeros_like(theta)
    second=np.zeros_like(theta)
    for k in range(1,801):
        _,grad=lossgrad(theta)
        grad*=min(1.,5./max(np.linalg.norm(grad),1e-250))
        first=.9*first+.1*grad
        second=.999*second+.001*grad**2
        theta-=.02*(first/(1-.9**k))/(np.sqrt(second/(1-.999**k))+1e-8)
    return theta


def fit_case(case):
    began=time.monotonic()
    task,size,seed=case
    x,y=observed(task,seed)
    if task in arithmetic.TASKS:
        start=arithmetic.initial(size,seed)
        lossgrad=lambda p: reverse_loss(size,p,x,y)
        adam=adam_stage(lossgrad,start)
        # This freshly reconstructed Adam checkpoint must yield the saved full-BFGS endpoint.
        check=minimize(lossgrad,adam.copy(),jac=True,method='BFGS',options=dict(gtol=1e-14,maxiter=1000))
        final=check.x if check.fun <= lossgrad(adam)[0] else adam
        with np.load(ROOT / 'focus/arithmetic/fits.npz') as archived:
            assert np.array_equal(final,archived[f'{task}_{seed}_{size}_adam_bfgs'])
    else:
        start=initial(task,size,seed)
        lossgrad=objective(task,size,start,x,y)
        with np.load(ROOT / 'extension/optimizer/states.npz') as archived:
            assert np.array_equal(start,archived[f'{task}_{size}_{seed}_xavier'])
            adam=archived[f'{task}_{size}_{seed}_adam800'].copy()
    p={'xavier':start,'adam800':adam}
    diag={}
    with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
        p['gd'],diag['gd']=gradient_descent(lossgrad,start)
        for method,origin in [('gd_lbfgs','gd'),('adam_lbfgs','adam800'),('lbfgs_xavier','xavier')]:
            p[method],diag[method]=lbfgs(lossgrad,p[origin],origin)
            assert lossgrad(p[method])[0] <= lossgrad(p[origin])[0]
    # No evaluation set has been constructed or used by the four fits.
    test=inputs(task)
    dev='early' if task in SERIES else 'inner'
    splits={'train':(x,y),'dev':(test[dev],reference(task,test[dev])),
            'outer':(test['outer'],reference(task,test['outer']))}
    provenance=('matched objective on new arithmetic graphs' if task in arithmetic.TASKS else
                'matched objective on eight existing graphs')
    scores=[]
    for method,theta in p.items():
        train_loss=lossgrad(theta)[0]
        for split,(xx,yy) in splits.items():
            with np.errstate(over='ignore', invalid='ignore'):
                yh=(arithmetic.forward(size,theta,xx)[0] if task in arithmetic.TASKS else
                    predict(task,size,theta,xx))
                e=yh==yy
                equal=e.all(axis=1) if e.ndim==2 else e
                err=rmse(yh,yy)
                split_loss=float(.5*np.mean(((yh-yy)/np.maximum(1,np.abs(yy)))**2))
            if split=='train':
                assert np.isclose(split_loss,train_loss,rtol=1e-12,atol=1e-29),(case,method,split_loss,train_loss)
            scores.append(dict(protocol=provenance,task=task,size=size,params=len(start),seed=seed,
                               method=method,split=split,split_description=(
                                'early indices 9–32' if split=='dev' and task in SERIES else
                                {'dev':'central held-out','train':'observed','outer':'outer held-out'}[split]),
                               n_examples=equal.size,n_scalars=yy.size,exact_count=int(equal.sum()),
                               exact_fraction=float(equal.mean()),raw_rmse=err,train_loss=train_loss,
                               split_loss=split_loss))
    seconds=time.monotonic()-began
    ds=[dict(task=task,size=size,seed=seed,method=name,case_wall_seconds=seconds,**info)
        for name,info in diag.items()]
    return scores, ds, {f'{task}_{size}_{seed}_{method}':v for method,v in p.items()}


def main(workers):
    eligible=read_csv(ROOT / 'focus/nonconvex/eligibility.csv')
    cases=[(r['task'],int(r['size']),int(r['seed'])) for r in eligible if r['status']=='nonconvex']
    cases.extend((task,w,s) for task in arithmetic.TASKS for w in arithmetic.WIDTHS for s in arithmetic.SEEDS)
    assert len(set(cases))==87
    scores,diags,states=[],[],{}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index,(rows,counts,params) in enumerate(pool.map(fit_case,cases)):
            scores.extend(rows);diags.extend(counts);states.update(params)
            if (index+1)%15==0: print(f'Completed {index+1}/{len(cases)}',flush=True)
    assert len(scores)==87*6*3 and len(diags)==87*4
    write_csv('scores.csv',scores)
    write_csv('diagnostics.csv',diags)
    np.savez_compressed(HERE/'weights.npz',**states)
    settings=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),workers=workers,blas_environment={
                    k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                  gd=dict(max_updates=GD_UPDATES,initial_rate=1.,armijo=ARMIJO,
                          max_halvings=MAX_HALVINGS,gradient_infinity_tolerance=1e-14),
                  lbfgs=dict(method='L-BFGS-B',bounds=None,**OPTIONS))
    (HERE/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args().workers)
