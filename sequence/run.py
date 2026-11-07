"""Reproduce the recurrent and sorting experiments (NumPy float64 on CPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def xavier(rng, fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_out, fan_in))


def pack(*arrays):
    return np.concatenate([a.reshape(-1) for a in arrays])


class Recurrence:
    """Column-state convention. Parameter order is transition, input/initial, readout."""
    def __init__(self, task, width):
        self.task, self.d = task, width

    def init(self, seed):
        rng = np.random.default_rng(seed)
        return pack(xavier(rng, self.d, self.d), xavier(rng, 1, self.d),
                    xavier(rng, self.d, 1))

    def parts(self, p):
        d = self.d
        return p[:d*d].reshape(d, d), p[d*d:d*d+d], p[d*d+d:]

    def burn(self, initial):
        d = self.d
        p = initial.copy()
        p.fill(0.0)
        A, b, v = self.parts(p)
        if self.task == 'sum':
            A[0,0] = b[0] = v[0] = 1
        elif self.task == 'fib' and d >= 2:
            A[:2,:2] = [[0,1],[1,1]]
            b[1] = v[0] = 1
        elif self.task == 'tri' and d >= 3:
            A[:3,:3] = [[1,1,0],[0,1,1],[0,0,1]]
            b[1:3] = 1
            v[0] = 1
        else:
            return None
        return p

    def forward(self, p, x):
        A, b, v = self.parts(p)
        N, T = x.shape
        h = np.empty((T,N,self.d))
        if self.task == 'sum':
            h[0] = x[:,0,None] * b
            for i in range(1,T):
                h[i] = h[i-1] @ A.T + x[:,i,None]*b
        else:
            h[0] = b
            for i in range(1,T):
                h[i] = h[i-1] @ A.T
        return (h @ v).T, h

    def backward(self, p, x, h, err):
        A, b, v = self.parts(p)
        gA, gb = np.zeros_like(A), np.zeros_like(b)
        gv = np.einsum('nt,tnd->d', err, h)
        adj = np.zeros_like(h[0])
        for i in range(x.shape[1] - 1, -1, -1):
            adj = err[:,i,None]*v + adj @ A
            if i:
                gA += adj.T @ h[i-1]
            if self.task == 'sum':
                gb += x[:,i] @ adj
            elif i == 0:
                gb += adj.sum(axis=0)
        return pack(gA, gb, gv)


def odd_even_network(n):
    """Batcher odd-even mergesort for a power-of-two arity."""
    assert n >= 2 and n & (n - 1) == 0
    pairs = []
    def merge(lo, size, step):
        if 2*step >= size:
            pairs.append((lo, lo+step))
        else:
            merge(lo, size, 2*step)
            merge(lo+step, size, 2*step)
            for i in range(lo+step, lo+size-step, 2*step):
                pairs.append((i, i+step))
    def sort(lo, size):
        if size > 1:
            half = size//2
            sort(lo, half)
            sort(lo+half, half)
            merge(lo, size, 1)
    sort(0,n)
    return pairs


class Sort:
    """Each gate sends (a,b) to (a-u*r,b+v*r), r=ReLU(q*(a-b))."""
    def __init__(self, gates=5, tied=False):
        self.gates, self.tied = gates, tied

    def init(self, seed):
        rng = np.random.default_rng(seed)
        return pack(*[xavier(rng,1,1) for _ in range(3*(1 if self.tied else self.gates))])

    def burn(self, initial):
        p = initial.copy()
        p[:] = 1
        return p

    def forward(self, p, x):
        state, trace = x.copy(), []
        pairs = odd_even_network(x.shape[1])
        if not self.tied:
            pairs = pairs[:self.gates]
        for j, (l,r) in enumerate(pairs):
            q,u,v = p.reshape(-1,3)[0 if self.tied else j]
            delta = state[:,l] - state[:,r]
            spike = np.maximum(0, q*delta)
            state[:,l] -= u*spike
            state[:,r] += v*spike
            trace.append((l,r,delta,spike))
        return state, trace

    def backward(self, p, x, trace, err):
        grad = np.zeros_like(p).reshape(-1,3)
        adj = err.copy()
        for j in reversed(range(len(trace))):
            l,r,delta,spike = trace[j]
            k = 0 if self.tied else j
            q,u,v = p.reshape(-1,3)[k]
            dl, dr = adj[:,l], adj[:,r]
            dz = (dr*v - dl*u)*(spike > 0)
            grad[k] += [dz @ delta, -dl @ spike, dr @ spike]
            adj[:,l] += dz*q
            adj[:,r] -= dz*q
        return grad.reshape(-1)


def targets(task, x):
    if task == 'sum':
        return np.cumsum(x,axis=1)
    if task == 'sort':
        return np.sort(x,axis=1)
    n = x.shape[1]
    if task == 'tri':
        numbers = [i*(i+1)//2 for i in range(n)]
    else:
        numbers = [0,1]
        while len(numbers) < n:
            numbers.append(numbers[-2]+numbers[-1])
    return np.array(numbers[:n],dtype=np.float64)[None,:]


def data(N, T, split, seed, dyadic=False):
    rng = np.random.default_rng(seed)
    if dyadic:
        if split == 'ID':
            return rng.integers(-3*2**20, 3*2**20+1, (N,T)).astype(float)/2**20
        mags = rng.integers(3*2**20 + 1, 10*2**20 + 1, (N,T))/2**20
    elif split == 'ID':
        return rng.uniform(-3,3,(N,T))
    else:
        mags = rng.uniform(np.nextafter(3.,10.),10,(N,T))
    return mags*rng.choice([-1.,1.],(N,T))


def objective(model, x, y, scale):
    def fun(p):
        yh, cache = model.forward(p,x)
        residual = (yh-y)/scale
        loss = .5*np.mean(residual**2)
        grad = model.backward(p,x,cache,residual/(residual.size*scale))
        return loss, grad
    return fun


def fit(fun, initial, steps):
    from scipy.optimize import minimize
    p = initial.copy()
    m, v = np.zeros_like(p), np.zeros_like(p)
    for step in range(1,steps+1):
        _, grad = fun(p)
        grad *= min(1., 5./max(np.linalg.norm(grad),1e-250))
        m = .9*m + .1*grad
        v = .999*v + .001*grad**2
        p -= .02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    adam_loss = float(fun(p)[0])
    result = minimize(fun, p, jac=True, method='BFGS',
                      options={'maxiter':1000,'gtol':1e-14})
    p = result.x if result.fun <= adam_loss else p
    return p, {'adam_loss':adam_loss,'final_loss':float(fun(p)[0]),
               'bfgs_iterations':int(result.nit),'optimizer_status':str(result.message)}


def metrics(yh,y):
    error = np.abs(yh-y)
    exact = yh == y
    one = (np.nextafter(y,-np.inf) <= yh) & (yh <= np.nextafter(y,np.inf))
    bits = yh.copy(order='C').view('uint64') == y.copy(order='C').view('uint64')
    scaled = error <= (8*np.finfo('float64').eps)*np.maximum(1,np.abs(y))
    return {'n':y.shape[0], 'length':y.shape[1], 'bit_fraction':float(bits.mean()),
            'exact':float(exact.mean()),'exact_all':float(exact.all(axis=1).mean()),
            'one_ulp':float(one.mean()),'one_ulp_all':float(one.all(axis=1).mean()),
            'eps8':float(scaled.mean()),'eps8_all':float(scaled.all(axis=1).mean()),
            'rmse':float(np.sqrt(np.mean(error**2))),'linf':float(error.max()),
            'relative_linf':float((error/np.maximum(1.,np.abs(y))).max())}


def tests(task, model):
    if task in ('fib','tri'):
        x = np.zeros((1,65))
        return [('train_indices_0_8',x,slice(0,9)),
                ('ood_indices_9_32',x,slice(9,33)),
                ('ood_indices_33_64',x,slice(33,65))]
    length = 4 if task == 'sort' else 8
    arities = [length, length*2 if task=='sort' else 32]
    if task=='sort' and not model.tied:
        arities = [length]
    return [(f'{domain}_T{t}'+('_dyadic' if dyadic else ''),
             data(2048,t,domain,7001 + t + (domain=='OOD') + 101*dyadic,dyadic),
             slice(None))
            for t in arities for domain in ['ID','OOD']
            for dyadic in ([False,True] if task=='sort' else [False])]


def run(steps, seeds):
    results, audits = [], []
    models = [(task, Recurrence(task,d), str(d)) for task in ('sum','fib','tri')
              for d in ({'sum':[1,2,4], 'fib':[1,2,3,4], 'tri':[1,2,3,4]}[task])]
    models += [('sort', Sort(g), f'g{g}') for g in (3,4,5)]
    models += [('sort', Sort(tied=True),'tied')]
    for task, model, size in models:
        T = {'fib':9,'tri':9,'sum':8,'sort':4}[task]
        x = data(256,T,'ID',3330) if task in ['sum','sort'] else np.zeros((1,T))
        y = targets(task,x)
        scale = np.maximum(1.,np.abs(y)) if task in ['fib','tri'] else 1.
        fun = objective(model,x,y,scale)
        # Central finite-difference audit at Xavier init. At random generic inputs
        # ReLU zero preactivations have probability zero.
        initial = model.init(seeds[0])
        _, grad = fun(initial)
        eye = np.eye(initial.size)
        delta = 1e-6
        num = np.array([(fun(initial+u*delta)[0]-fun(initial-u*delta)[0])/(2*delta) for u in eye])
        err = float(np.max(np.abs(grad-num)/np.maximum(1.,np.abs(num))))
        assert err < 2e-5, (task,size,err)
        audits.append({'task':task,'size':size,'max_scaled_gradient_discrepancy':err})
        splits = tests(task,model)
        for seed in seeds:
            initial = model.init(seed)
            burn = model.burn(initial)
            trained, info = fit(fun, initial,steps)
            common = dict(task=task,size=size,seed=seed,parameter_count=initial.size,
                          initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                          initial=initial.tolist())
            for method, p in [('program',burn),('bp',trained)]:
                if p is None:
                    continue
                scores = {}
                for name, xt, segment in splits:
                    yh = model.forward(p,xt)[0][:,segment]
                    yt = targets(task,xt)[:,segment]
                    scores[name] = metrics(yh,yt)
                results.append(dict(common, method=method, final=p.tolist(),
                                    fit=info if method=='bp' else None,scores=scores))
        print(task,size,'done',flush=True)
    return results, audits


if __name__ == '__main__':
    import scipy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adam-steps',type=int,default=800)
    ap.add_argument('--seeds',type=int,nargs='+',default=[101,202,303,404,505])
    ap.add_argument('--output',type=Path,default=HERE/'results.json')
    args = ap.parse_args()
    start = time.perf_counter()
    results, audits = run(args.adam_steps,args.seeds)
    meta = {'dtype':'float64','python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'machine':platform.machine(),'platform':platform.platform(),
            'seconds':time.perf_counter()-start,'adam_steps':args.adam_steps,'seeds':args.seeds}
    args.output.write_text(json.dumps(dict(metadata=meta,gradient_audits=audits,results=results),
                                      indent=1,allow_nan=False)+'\n')
