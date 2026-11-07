"""Reproduce the binary64 scalar experiments. Requires NumPy and SciPy."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np

WIDTHS = (0, 1, 3, 7, 15, 31, 63)
DEPTHS = (1, 4, 8, 12, 16, 20, 24, 27, 30)
SEEDS = (11, 22, 33)
TARGETS = {"square": np.square, "sin": np.sin, "tanh": np.tanh,
           "selfpower": lambda x: np.power(x, x)}
ROOT = Path(__file__).resolve().parent

def xavier(rng, m, n):
    r = np.sqrt(6 / (m + n))
    return rng.uniform(-r, r, (m, n))

def data(name):
    # The split arrays use independent streams. All seeds and methods share them.
    positive = name == "selfpower"
    left = 0 if positive else -3
    rng = np.random.default_rng(20261105)
    ans = {"train": rng.uniform(left, 3, 1024),
           "in": rng.uniform(left, 3, 4096)}
    if not positive:
        ans["left"] = rng.uniform(-10, -3, 4096)
    ans["right"] = rng.uniform(3, 10, 4096)
    return ans

def metrics(pred, y):
    abs_err = np.abs(pred - y)
    # IEEE-754 float bits mapped monotonically to an unsigned integer lattice.
    def ordered(v):
        u = np.ascontiguousarray(v, dtype=np.float64).view(np.uint64)
        sign = np.uint64(1 << 63)
        return np.where(u & sign, ~u + np.uint64(1), u | sign)
    a, b = ordered(pred), ordered(y)
    ulps = np.maximum(a, b) - np.minimum(a, b)
    return dict(n=y.size, exact=np.mean(pred == y), ulp1=np.mean(ulps <= 1),
                scaled_eps=np.mean(abs_err <= np.finfo(float).eps * np.maximum(1, np.abs(y))),
                mae=np.mean(abs_err), rmse=np.sqrt(np.mean(abs_err**2)), max_abs=abs_err.max())

def write_records(rows, name):
    with (ROOT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def pack(parts):
    return np.concatenate([a.ravel() for a in parts])

def initial_hinge(h, seed):
    rng = np.random.default_rng(seed)
    ab = [xavier(rng, 1, 1), np.zeros(1)]
    if not h:
        return pack(ab)
    return pack(ab + [xavier(rng, 1, h), np.zeros(h), xavier(rng, h, 1)])

def hinge(x, p, grad_y=None):
    h = (p.size - 2) // 3
    a, b = p[:2]
    if not h:
        return a*x+b if grad_y is None else np.array([x @ grad_y, grad_y.sum()])
    w, c, v = p[2:].reshape(3, h)
    z = x[:, None] * w + c
    features = np.maximum(0, z)
    if grad_y is None:
        return a*x + b + features @ v
    deriv = (grad_y[:, None] * v) * (z > 0)
    return pack([np.array([x @ grad_y, grad_y.sum()]), x @ deriv, deriv.sum(0), grad_y @ features])

def program_spline(initial, h, lo, hi, target):
    # Fill the existing trainable slots. Every ordinate uses the supplied specification.
    knots = np.linspace(lo, hi, h + 2)
    y = TARGETS[target](knots)
    slopes = np.diff(y) / np.diff(knots)
    p = initial.copy()
    p[0] = slopes[0]
    p[1] = y[0] - p[0]*lo
    if target == "square":
        p[1] -= (hi-lo)**2 / (8*(h+1)**2)
    if h:
        p[2:] = pack([np.ones(h), -knots[1:-1], np.diff(slopes)])
    return p

def train(p, x, y, forward, seed, adam_steps=500, max_lbfgs=250):
    from scipy.optimize import minimize
    # Gradient of unweighted MSE. Select the lower training loss between the
    # Adam endpoint and its L-BFGS refinement.
    rng = np.random.default_rng(seed + 100)
    p = p.copy()
    def objective(p, x=x, y=y):
        residual = forward(x, p) - y
        return np.mean(residual**2), forward(x, p, 2*residual/y.size)
    first, second = np.zeros_like(p), np.zeros_like(p)
    for i in range(1, adam_steps + 1):
        idx = rng.integers(x.size, size=64)
        _, g = objective(p, x[idx], y[idx])
        first = 0.9*first + 0.1*g
        second = 0.999*second + 0.001*g*g
        p -= 0.01*(first/(1-0.9**i)) / (np.sqrt(second/(1-0.999**i))+1e-8)
    result = minimize(objective, p, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=max_lbfgs, maxls=30, ftol=1e-16, gtol=1e-12))
    return result.x if result.fun < objective(p)[0] else p

def evaluate(rows, task, arch, size, p, forward, sets, method, seed):
    for split, x in sets.items():
        y = TARGETS[task](x)
        rows.append(dict(task=task, architecture=arch, size=size, params=p.size,
                         nonzero=np.count_nonzero(p), method=method, seed=seed,
                         split=split, **metrics(forward(x, p), y)))

def shallow_suite():
    rows = []
    for task in TARGETS:
        sets = data(task)
        for h in WIDTHS:
            initial = initial_hinge(h, SEEDS[0])
            for method, lo, hi in (("write_global", -10, 10), ("write_local", -3, 3)):
                if task == "selfpower":
                    lo = 0
                p = program_spline(initial, h, lo, hi, task)
                evaluate(rows, task, "hinge", h, p, hinge, sets, method, SEEDS[0])
            for seed in SEEDS:
                initial = initial_hinge(h, seed)
                x = sets["train"]
                p = train(initial, x, TARGETS[task](x), hinge, seed)
                evaluate(rows, task, "hinge", h, p, hinge, sets, "adam_lbfgs", seed)
        write_records(rows, "shallow.csv")
        print("finished", task, flush=True)

class DeepSquare:
    """Ordinary dense ReLU hidden layers, final affine readout."""
    def __init__(self, depth):
        self.depth = depth
        widths = [1, 2] + [3]*depth + [1]
        self.shapes = list(zip(widths, widths[1:]))
        self.cuts = np.cumsum([v for m,n in self.shapes for v in (m*n,n)])[:-1]
    def initial(self, seed):
        rng = np.random.default_rng(seed)
        return pack([v for m,n in self.shapes for v in (xavier(rng,m,n), np.zeros(n))])
    def unpack(self, p):
        parts = np.split(p, self.cuts)
        return [(parts[2*i].reshape(m,n), parts[2*i+1]) for i,(m,n) in enumerate(self.shapes)]
    def write(self, initial):
        p = initial.copy()
        layers = self.unpack(p)
        w, b = layers[0]
        w[:] = [[1/16, -1/16]]
        b[:] = 0
        w,b = layers[1]
        w[:] = 1
        b[:] = [0, 0, -0.5]
        # Each hidden state is [interpolant, current tent, positive half hinge].
        for i,(w,b) in enumerate(layers[2:-1], 1):
            w[:] = [[1,0,0], [-2*4.**(-i),2,2], [4*4.**(-i),-4,-4]]
            b[:] = [0, 0, -0.5]
        w,b = layers[-1]
        w[:] = 256*np.array([[1], [-2*4.**(-self.depth)], [4*4.**(-self.depth)]])
        b[:] = 0
        return p
    def forward(self, x, p, grad_y=None):
        layers = self.unpack(p)
        a = x[:,None]
        saved = []
        for w,b in layers[:-1]:
            z = a @ w + b
            if grad_y is not None:
                saved.append(a)
            a = np.maximum(0,z)
        w,b = layers[-1]
        if grad_y is None:
            return (a @ w + b).ravel()
        d = grad_y[:,None]
        grads = [(a.T @ d, d.sum(0))]
        for prev, (prev_w, _) in zip(reversed(saved), reversed(layers[:-1])):
            d = (d @ w.T)*(a>0)
            grads.append((prev.T @ d, d.sum(0)))
            w, a = prev_w, prev
        return pack([v for pair in reversed(grads) for v in pair])

def deep_suite():
    rows = []
    sets = data("square")
    for depth in DEPTHS:
        model = DeepSquare(depth)
        p = model.write(model.initial(SEEDS[0]))
        evaluate(rows, "square", "tent_mlp", depth, p, model.forward, sets, "write_global", SEEDS[0])
        for seed in SEEDS:
            x = sets["train"]
            p = train(model.initial(seed), x, x**2, model.forward, seed, 300, 100)
            evaluate(rows, "square", "tent_mlp", depth, p, model.forward, sets, "adam_lbfgs", seed)
        write_records(rows, "deep.csv")
        print("finished depth", depth, flush=True)

def native_tanh(x,p,grad_y=None):
    w,b,v,c = p
    t = np.tanh(w*x+b)
    if grad_y is None:
        return v*t+c
    d = grad_y*v*(1-t*t)
    return np.array([x @ d, d.sum(), t @ grad_y, grad_y.sum()])

def controls():
    rows = []
    for seed in SEEDS:
        sets = data("tanh")
        rng = np.random.default_rng(seed)
        p = pack([xavier(rng,1,1),np.zeros(1),xavier(rng,1,1),np.zeros(1)])
        for method in (["write_global","adam_lbfgs"] if seed == SEEDS[0] else ["adam_lbfgs"]):
            if method == "write_global":
                fit = p.copy()
                fit[:] = [1,0,1,0]
            else:
                x = sets["train"]
                fit = train(p,x,np.tanh(x),native_tanh,seed)
            evaluate(rows,"tanh","native_tanh",1,fit,native_tanh,sets,method,seed)
            if method == "adam_lbfgs":
                # Full Jacobian of the same four-slot network, for a least-squares
                # refinement without test data or rounding learned coefficients.
                polished = fit.copy()
                for _ in range(10):
                    w,b,v,c = polished
                    t = np.tanh(w*x+b)
                    d = v*(1-t*t)
                    jac = np.column_stack([x*d, d, t, np.ones(x.size)])
                    residual = native_tanh(x, polished)-np.tanh(x)
                    polished -= np.linalg.lstsq(jac, residual, rcond=None)[0]
                evaluate(rows,"tanh","native_tanh",1,polished,native_tanh,sets,
                         "adam_lbfgs_gauss_newton",seed)
    write_records(rows,"native.csv")

def arithmetic():
    rows = []
    rng = np.random.default_rng(20261105)
    sets = {"train": rng.uniform(-3,3,(1024,2)), "in":rng.uniform(-3,3,(4096,2))}
    sets["shell"] = rng.choice([-1.,1.],(8192,2))*rng.uniform(3,10,(8192,2))
    for name,sign in (("add",1),("subtract",-1)):
        x = sets["train"]
        y = x[:,0]+sign*x[:,1]
        design = np.column_stack([x,np.ones(x.shape[0])])
        for seed in SEEDS:
            initial = pack([xavier(np.random.default_rng(seed),2,1),np.zeros(1)])
            for method in (["write_global","newton"] if seed == SEEDS[0] else ["newton"]):
                p = initial.copy()
                if method == "write_global":
                    p[:] = [1,sign,0]
                else:
                    # Full-batch Gauss-Newton equals Newton for this quadratic.
                    for _ in range(10):
                        residual = (x @ p[:2] + p[2]) - y
                        p -= np.linalg.lstsq(design, residual, rcond=None)[0]
                for split,v in sets.items():
                    pred = v @ p[:2]+p[2]
                    target = v[:,0]+sign*v[:,1]
                    rows.append(dict(task=name,architecture="affine",size=0,params=3,
                                     nonzero=np.count_nonzero(p),method=method,seed=seed,
                                     split=split,**metrics(pred,target)))
            # Minimal bias-free class. GD on half mean squared error uses a
            # Lipschitz step computed from training inputs alone.
            for method in (["write_global","gradient_descent"] if seed == SEEDS[0] else ["gradient_descent"]):
                p = initial[:2].copy()
                if method == "write_global":
                    p[:] = [1, sign]
                else:
                    step = 1/np.linalg.eigvalsh(x.T @ x / len(x)).max()
                    for _ in range(64):
                        p -= step*(x.T @ (x @ p - y))/len(x)
                for split,v in sets.items():
                    rows.append(dict(task=name,architecture="linear",size=0,params=2,
                                     nonzero=np.count_nonzero(p),method=method,seed=seed,
                                     split=split,**metrics(v @ p,v[:,0]+sign*v[:,1])))
    write_records(rows,"affine.csv")

def sine_fold():
    rows = []
    sets = data("sin")
    fold = initial_hinge(6, SEEDS[0])
    fold[0:2] = [-1, -3*np.pi]
    fold[2:] = pack([np.ones(6),-(np.arange(-3,3)+0.5)*np.pi,2*(-1.)**np.arange(6)])
    # All target labels come from the original training window. The folding
    # stage uses the declared period, reflection law and evaluation envelope.
    identity = fold.copy()
    identity[:] = 0
    identity[0] = 1
    def composite(x, p, grad_y=None):
        first, second = p[:20], p[20:]
        z = hinge(x, first)
        if grad_y is None:
            return hinge(z, second)
        w,c,v = second[2:].reshape(3,-1)
        deriv = second[0] + ((z[:,None]*w+c > 0)*(v*w)).sum(1)
        return pack([hinge(x, first, grad_y*deriv), hinge(z, second, grad_y)])
    for h in WIDTHS:
        for method, layer, radius in (("write_phase_train_readout",fold,np.pi/2),
                                      ("identity_train_readout",identity,3)):
            p = initial_hinge(h, SEEDS[0])
            knots = np.linspace(-radius, radius, h+2)[1:-1]
            if h:
                p[2:] = pack([np.ones(h),-knots,np.zeros(h)])
            z = hinge(sets["train"],layer)
            features = np.column_stack([z, np.ones(z.size), np.maximum(0, z[:,None]-knots)])
            theta = np.linalg.lstsq(features, np.sin(sets["train"]),rcond=None)[0]
            p[:2] = theta[:2]
            if h:
                p[-h:] = theta[2:]
            evaluate(rows,"sin","fold_hinge",h,pack([layer,p]),composite,sets,method,SEEDS[0])
        for seed in SEEDS:
            # Use distinct generator streams for two same-shaped layers.
            initial = pack([initial_hinge(6,seed),initial_hinge(h,seed+1)])
            x = sets["train"]
            p = train(initial,x,np.sin(x),composite,seed)
            evaluate(rows,"sin","fold_hinge",h,p,composite,sets,"adam_lbfgs",seed)
    write_records(rows,"fold.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite",choices=["shallow","deep","native","affine","fold","all"], default="all",nargs="?")
    args = parser.parse_args()
    for name, fn in [("shallow",shallow_suite),("deep",deep_suite),("native",controls),("affine",arithmetic),("fold",sine_fold)]:
        if args.suite in (name,"all"):
            fn()
