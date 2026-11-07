"""Bidirectional integer clock control using only seven central labels for fitting."""
import json
from pathlib import Path

import numpy as np

from run import fit, metrics, xavier
from synthesize import recurrence_from_labels

HERE = Path(__file__).resolve().parent


def fib_pair(n):
    """Integer fast doubling, independent of the stepwise network."""
    if n == 0:
        return 0,1
    a,b = fib_pair(n//2)
    f = a*(2*b-a)
    g = a*a+b*b
    return (g,f+g) if n%2 else (f,g)


def oracle(task, n):
    if task == 'tri':
        return n*(n+1)//2
    value = fib_pair(abs(n))[0]
    return -value if n < 0 and abs(n)%2 == 0 else value


def compile_from_xavier(Aplus,Aminus,h,v,seed=101):
    d = len(h)
    rng = np.random.default_rng(seed)
    initial = (xavier(rng,d,d), xavier(rng,d,d), xavier(rng,1,d).ravel(),
               xavier(rng,d,1).ravel())
    final = tuple(a.copy() for a in initial)
    for slot,weight in zip(final,(Aplus,Aminus,h,v)):
        slot[:] = weight
    return initial,final


def forward(weights,n):
    plus,minus,initial,readout = weights
    h = initial.copy()
    A = plus if n >= 0 else minus
    for _ in range(abs(n)):
        h = A @ h
    return float(readout @ h)


def direct(task):
    if task=='fib':
        return ([[0,1],[1,1]], [[-1,1],[1,0]], [0,1], [1,0])
    return ([[1,1,0],[0,1,1],[0,0,1]], [[1,-1,1],[0,1,-1],[0,0,1]],
            [0,1,1], [1,0,0])


def companion(q,h):
    """A backward tick shifts the register right and solves the first component."""
    d = len(q)
    assert q[0] != 0
    plus = np.eye(d,k=1)
    plus[-1] = q
    minus = np.eye(d,k=-1)
    minus[0] = np.array(list(-q[1:])+[1.])/q[0]
    return plus,minus,h,np.eye(d)[0]


def records(task,weights):
    output = {}
    for split,idx in [('central',range(-3,4)),('negative_outer',range(-10,-3)),
                      ('positive_outer',range(4,11))]:
        xs = list(idx)
        yh = np.array([forward(weights,n) for n in xs])[None,:]
        y = np.array([oracle(task,n) for n in xs],float)[None,:]
        output[split] = dict(indices=xs,predictions=yh[0].tolist(),labels=y[0].tolist(),
                             scores=metrics(yh,y))
    return output


def solve_task(task):
    labels = [oracle(task,n) for n in range(-3,4)]
    d,c = recurrence_from_labels(labels)
    q = np.array([float(x) for x in c])
    h = np.array(labels[3:3+d],float)
    output = {'central_labels':labels,'width':d,'rational_coefficients':list(map(str,c)),
              'dense_slots':2*d*d+2*d}
    for kind,targets in [('direct',direct(task)),('synthesized',companion(q,h))]:
        initial,weights = compile_from_xavier(*targets)
        scores = records(task,weights)
        assert all(s['scores']['bit_fraction']==1 for s in scores.values())
        output[kind] = dict(initial=[a.tolist() for a in initial],
                            final=[a.tolist() for a in weights], splits=scores)
    # The gradient control receives a shift-register architecture and its width.
    # It predicts each next central label from the previous d *observed* labels.
    # It changes only the d recurrence coefficients from Xavier; h0 uses the
    # same observed stencil as the exact data-only solver.
    X = np.array([labels[i:i+d] for i in range(len(labels)-d)],float)
    Y = np.array(labels[d:],float)
    def fun(q):
        r = X @ q-Y
        return float(.5*np.mean(r*r)), X.T @ r/len(Y)
    fits = []
    for seed in [101,202,303,404,505]:
        rng = np.random.default_rng(seed)
        initial = xavier(rng,d,1).ravel()
        q0,info = fit(fun,initial,800)
        _,weights = compile_from_xavier(*companion(q0,h),seed=seed)
        fits.append(dict(seed=seed,initial=initial.tolist(),coefficients=q0.tolist(),fit=info,
                         splits=records(task,weights)))
    output['gradient_teacher_forced'] = fits
    return output


if __name__=='__main__':
    results = {t:solve_task(t) for t in ('fib','tri')}
    (HERE/'signed_indices.json').write_text(json.dumps(results,indent=1,allow_nan=False)+'\n')
    for task,r in results.items():
        print(task,r['rational_coefficients'],r['dense_slots'],
              [(q['seed'],sum(sum(y==p for y,p in zip(s['labels'],s['predictions']))
                              for k,s in q['splits'].items() if k!='central'))
               for q in r['gradient_teacher_forced']])
