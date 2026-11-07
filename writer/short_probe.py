"""Formula-aware Astra can use arithmetic scratch work for finite-horizon editing."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from error_metrics import rmse

from probe import build
HERE = Path(__file__).resolve().parent
HORIZONS = [2,3,4,2,3,4]

def make_prompt(i):
    d,j,leak,gain,A,b = build(i)
    return (
        'You may use a calculator or local Python for arithmetic. Return only a JSON array '
        'of literal [index, new_value] pairs. Minimize the NUMBER of changed scalar entries, '
        'including zeros. Unlisted entries retain their values. '
        f'The initial state h is zero, h = A @ h + b*x, output h[{j}]. '
        f'For ONLY the first {HORIZONS[i]} steps, on EVERY real input stream, the target '
        f'is s starting at zero, updated s=({leak})*s+({gain})*x. '
        'You may sacrifice behavior after the specified horizon. Supply enough digits '
        'to reproduce the target impulse coefficients within absolute tolerance 1e-9. '
        f'Index base zero. A is row-major, indices 0 through {d*d-1}, followed by b.\n'
        'A='+json.dumps(A.tolist())+'\nb='+json.dumps(b.tolist())+'\n')

def score_one(i):
    H = HORIZONS[i]
    d,j,leak,gain,A,b = build(i)
    z = np.r_[A.ravel(),b]
    old = z.copy()
    t = (HERE/f'case{i+7}.reply.txt').read_text().strip()
    if t.startswith('```'): t=t.split('\n',1)[1].rsplit('```',1)[0]
    seen=set()
    for k,v in json.loads(t):
        assert isinstance(k,int) and 0<=k<len(z) and k not in seen and np.isfinite(v)
        z[k]=v
        seen.add(k)
    A, h = z[:d*d].reshape(d,d),z[d*d:]
    err=[]
    for k in range(H+1):
        err.append(abs(float(h[j]) - gain*leak**k))
        h=A@h
    return dict(case=i+7,d=d,H=H,edits=int(sum(old!=z)),fit_max=max(err[:H]),next_abs=err[H],
                fit_rmse=rmse(err[:H], np.zeros(H)),
                fit_exact=int(np.count_nonzero(np.asarray(err[:H]) == 0)),
                next_rmse=err[H],next_exact=int(err[H] == 0))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','score'])
    if parser.parse_args().action=='generate':
        for i in range(6): (HERE/f'case{i+7}.prompt.txt').write_text(make_prompt(i))
    else:
        scores=[score_one(i) for i in range(6)]
        (HERE/'short_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
        print(scores)
