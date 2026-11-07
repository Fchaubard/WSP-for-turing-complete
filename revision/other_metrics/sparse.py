"""Gradient fits on the positions edited by each original sparse WSP reply."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from error_metrics import rmse

sys.path.insert(0, str(ROOT/'writer'))
from probe import build as initial_state
from short_probe import HORIZONS

sys.path.append(str(ROOT/'sequence'))
from run import fit


def load_reply(index):
    text = (ROOT/'writer'/f'case{index}.reply.txt').read_text().strip()
    if text.startswith(chr(96)*3):
        text = text.split('\n', 1)[1].rsplit(chr(96)*3, 1)[0]
    return json.loads(text)


def impulse(initial, width, readout, placement, values, horizon):
    flat = initial.copy()
    flat[placement] = values
    A = flat[:width**2].reshape(width, width)
    state = flat[width**2:]
    hidden = [state.copy()]
    for _ in range(1, horizon):
        state = A @ state
        hidden.append(state.copy())
    return A, np.array(hidden), np.array(hidden)[:,readout]


def loss_grad(initial, width, readout, placement, labels):
    def objective(values):
        A, states, outputs = impulse(initial, width, readout, placement, values, len(labels))
        residual = outputs - labels
        row_grad = np.zeros_like(A)
        adj = np.zeros(width)
        for t in reversed(range(len(labels))):
            adj[readout] += residual[t] / len(labels)
            if t:
                row_grad += np.outer(adj, states[t-1])
                adj = A.T @ adj
        grad = np.r_[row_grad.ravel(), adj]
        return float(np.mean(residual**2)/2), grad[placement]
    return objective


def measure(values, target, unit):
    eq = np.equal(values, target)
    exact = eq if len(eq.shape)==1 else np.all(eq, axis=1)
    return dict(rmse=rmse(values,target),exact_count=int(np.sum(exact)),example_count=len(exact),
                scalar_count=eq.size,unit=unit)


def run():
    rows = []
    parameters = {}
    for i in range(6):
        d,j,leak,gain,A,b = initial_state(i)
        base = np.r_[A.ravel(), b]
        for condition,case,H in [('short',7+i,HORIZONS[i]),('all',1+i,d+1)]:
            patch = load_reply(case)
            positions = np.array([k for k,_ in patch],dtype=int)
            literal = np.array([v for _,v in patch])
            targets = gain*leak**np.arange(H+1)
            obj = loss_grad(base,d,j,positions,targets[:H])
            p0 = base[positions].copy()
            # Audit the derivative on the actual Xavier inputs.
            test_direction = np.cos(np.arange(len(p0))+.61)
            observed = (obj(p0+1e-6*test_direction)[0]-obj(p0-1e-6*test_direction)[0])/2e-6
            assert abs(observed - obj(p0)[1]@test_direction) < 1e-5*(1+abs(observed))
            fitted, fit_details = fit(obj, p0, 800)
            for method,vals in [('WSP',literal),('BP',fitted)]:
                mat,states,pred=impulse(base,d,j,positions,vals,H+1)
                parameters[f'{case}_{method}']=vals
                common=dict(case=case, method=method, condition=condition, d=d, fit_horizon=H, slots=len(positions))
                for split,z,y in [('fit_impulse',pred[:H],targets[:H]),('next_impulse',pred[H:],targets[H:])]:
                    rows.append(dict(**common,split=split,**measure(z,y,'impulse coefficient')))
                for split,T in [('dev',8),('shell',64)]:
                    rng=np.random.default_rng(9251+i)
                    x=(rng.uniform(-3,3,(512,T)) if split=='dev' else rng.uniform(3,10,(512,T))*rng.choice([-1,1],(512,T)))
                    state=np.zeros((512,d))
                    target=np.zeros(512)
                    pred_all=np.empty((512,T));ref_all=np.empty_like(pred_all)
                    for t,u in enumerate(x.T):
                        state=state@mat.T + states[0]*u[:,None]
                        target=leak*target+gain*u
                        pred_all[:,t]=state[:,j];ref_all[:,t]=target
                    rows.append(dict(**common,split=split,**measure(pred_all,ref_all,'whole stream')))
    here=Path(__file__).parent
    with (here/'sparse_paired.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=rows[0]);out.writeheader();out.writerows(rows)
    np.savez_compressed(here/'sparse_states.npz',**parameters)


if __name__=='__main__':
    run()
