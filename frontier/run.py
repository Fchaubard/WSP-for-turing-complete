"""Test impulse interpolation, one-edit deletions, and two optimal recurrent writes."""
from pathlib import Path
import numpy as np
from numpy.random import default_rng
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from error_metrics import rmse


HERE = Path(__file__).resolve().parent
WIDTHS = (1,2,4,8,16,32)
HORIZONS = (1,2,4,8,16,32)
STEPS = (4,8,16,32,64,128)
SEEDS = 64
BATCH = 64

def draw(d, seed):
    rng = default_rng(90001 + 100*d + seed)
    return (rng.uniform(-np.sqrt(3/d), np.sqrt(3/d), (d,d)),
            rng.uniform(-np.sqrt(6/(1+d)), np.sqrt(6/(1+d)), d))

def row_write(A,b):
    A, b = A.copy(), b.copy()
    A[0] = 0
    A[0,0] = b[0] = 1
    return A, b

def eigen_write(A,b):
    # Impose A b = b with one entry per row using b[0]=1.
    A,b = A.copy(),b.copy()
    b[0] = 1
    A[:,0] = b - A[:,1:] @ b[1:]
    return A,b

def short_write(A, b, H):
    # The first H columns form an invertible minor a.s. (shift-matrix witness).
    O = np.empty((H,len(b)))
    O[0] = np.eye(len(b))[0]
    for t in range(1,H):
        O[t] = O[t-1] @ A
    O_H = O[:,:H]
    z = b.copy()
    z[:H] = np.linalg.solve(O_H, np.ones(H) - O[:,H:] @ z[H:])
    return z, float(np.linalg.cond(O_H))

def short_experiment():
    out = []
    for d in WIDTHS:
        for H in [h for h in HORIZONS if h <= d]:
            for s in range(SEEDS):
                A, b = draw(d,s)
                z, cond = short_write(A,b,H)
                # Independent execution through the recurrence for a unit impulse.
                u = z.copy()
                y = []
                for t in range(H+1):
                    y.append(float(u[0]))
                    u = A @ u
                out.append(dict(d=d,H=H,seed=s,edits=int(np.count_nonzero(z!=b)),
                                condition=cond,fit_max=max(abs(k-1) for k in y[:H]),
                                next_abs=abs(y[H]-1), fit_rmse=rmse(y[:H], 1.),
                                fit_exact=int(np.count_nonzero(np.asarray(y[:H]) == 1.)),
                                next_rmse=abs(y[H]-1), next_exact=int(y[H] == 1.)))
    return out

def trajectory_experiment():
    out = []
    for d in WIDTHS:
        for s in range(SEEDS):
            A, b = draw(d,s)
            for method in ['row','eigenvector','omit_one']:
                M, v = eigen_write(A,b) if method=='eigenvector' else row_write(A,b)
                if method=='omit_one':
                    entry = s % (d+1)
                    if entry<d: M[0,entry] = A[0,entry]
                    else: v[0]=b[0]
                sroot = np.max(abs(np.linalg.eigvals(M)))
                for split in ['inner','shell']:
                    # Identical inputs across methods, widths and initialization replicates.
                    data_rng = default_rng(871 if split=='inner' else 873)
                    x = data_rng.uniform(-3,3,(128,BATCH)) if split=='inner' else (
                        data_rng.uniform(3,10,(128,BATCH))*data_rng.choice([-1,1],(128,BATCH)))
                    h, y = np.zeros((BATCH,d)), np.zeros(BATCH)
                    exact = np.ones(BATCH,dtype=bool)
                    close = exact.copy()
                    maxerr = 0.
                    all_errors = np.empty_like(x)
                    for t, u in enumerate(x,1):
                        h = h @ M.T + u[:,None]*v
                        y += u
                        e = np.abs(h[:,0]-y)
                        all_errors[t-1] = e
                        exact &= e==0
                        close &= e <= 1e-10*(1+np.abs(y))
                        maxerr = max(maxerr,float(max(e)))
                        if t in STEPS:
                            out.append(dict(d=d,seed=s,t=t,split=split,method=method,
                                            edits=int(np.sum(M!=A)+np.sum(v!=b)),
                                            rho=float(sroot), whole_exact=float(exact.mean()),
                                            whole_close=float(close.mean()), max_abs=maxerr,
                                            rmse=rmse(all_errors[:t], 0.)))
    return out

if __name__=='__main__':
    import csv
    for name,experiment in [('short',short_experiment),('trajectory',trajectory_experiment)]:
        rows = experiment()
        with (HERE/f'{name}.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(name,len(rows))
