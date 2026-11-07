"""A single independent literal-weight proposal per benchmark task."""
import argparse
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
TASKS=['square','sin','selfpower','tanh','add','subtract','fib','tri','sort']
def setup(i):
    task=TASKS[i]
    rng=np.random.default_rng(14314+i)
    def w(m,n):
        return rng.uniform(-np.sqrt(6/(m+n)),np.sqrt(6/(m+n)),(n,m)).ravel()
    if i<3:
        p=np.r_[w(1,1),0.,w(1,7),np.zeros(7),w(7,1)]
        arch='y=a*x+b+sum(v[k]*ReLU(w[k]*x+c[k]) for k in range(7)). Flatten as a,b,w[0:7],c[0:7],v[0:7].'
        target=['x squared on [-10,10], minimize uniform absolute error',
                'sine in radians on [-10,10], minimize uniform absolute error',
                'x raised to x on [0,10], defining 0^0=1, minimize maximum abs(y-target)/max(1,abs(target))'][i]
    elif i==3:
        p=np.r_[w(1,1),0.,w(1,1),0.]
        arch='y=v*tanh(w*x+b)+c. Flatten as w,b,v,c.'
        target='tanh(x) on [-10,10], numerically exact.'
    elif i in (4,5):
        p=w(2,1)
        arch='y=w[0]*x[0]+w[1]*x[1]. Flatten as w[0],w[1].'
        target='x[0] '+('plus' if i==4 else 'minus')+' x[1], numerically exact.'
    elif i in (6,7):
        d=i-4
        p=np.r_[w(d,d),w(1,d),w(d,1)]
        arch=f'The initial state is b. On integer index n, output v @ (A^n) @ b via n recurrent matrix steps. State width {d}. Flatten A row-major, then b, then v.'
        target=('Fibonacci, F0=0,F1=1' if i==6 else 'triangular numbers, T_n=n*(n+1)/2')+', exactly for all nonnegative integer indices in real arithmetic.'
    else:
        p=np.r_[w(1,1),w(1,1),w(1,1)]
        arch='The stages use wire pairs (0,1),(2,3),(0,2),(1,3),(1,2), in order. For each pair (a,b), set t=ReLU(q*(a-b)); outputs are (a-u*t,b+v*t). The same 3 coefficients appear in all five stages. Flatten as q,u,v.'
        target='sort a 4-vector ascending, exactly over the reals.'
    return p, ('You may use scratch arithmetic. Write literal weights into an existing Xavier-initialized neural network. '
        'Only the weights can change. Return a bare JSON array containing the complete flattened final weight vector, '
        'with enough decimal precision. No test feedback or follow-up. Architecture: '+arch+' Target: '+target+
        ' Initial parameters: '+json.dumps(p.tolist())+'\n')

def score(i,p):
    from numpy import maximum as relu
    rng=np.random.default_rng(20261105)
    task=TASKS[i]
    assert p.shape==setup(i)[0].shape and np.all(np.isfinite(p))
    out={}
    for split in ['in','shell']:
        if i<6 or i==8:
            n,T = (4096,(4 if i==8 else 2 if i in (4,5) else 1))
            x=rng.uniform(0 if i==2 else -3,3,(n,T)) if split=='in' else rng.uniform(3,10,(n,T))*(1 if i==2 else rng.choice([-1,1],(n,T)))
            if T==1: x=x[:,0]
        if i<3:
            pred=p[0]*x+p[1]+relu(0,x[:,None]*p[2:9]+p[9:16])@p[16:]
            y=[np.square,np.sin,lambda u:np.power(u,u)][i](x)
        elif i==3:
            pred=p[2]*np.tanh(p[0]*x+p[1])+p[3]; y=np.tanh(x)
        elif i in [4,5]:
            pred=x@p; y=x[:,0]+x[:,1] if i==4 else x[:,0]-x[:,1]
        elif i in [6,7]:
            d=i-4; A=p[:d*d].reshape(d,d); h=p[d*d:d*d+d]; v=p[-d:]
            pred=[]; f,g=0,1; y=[]
            for n in range(65):
                pred.append(v@h); h=A@h; y.append(f if i==6 else n*(n+1)//2); f,g=g,f+g
            sl=slice(0,9) if split=='in' else slice(9,65)
            pred=np.array(pred)[sl]; y=np.array(y,dtype=float)[sl]
        else:
            pred=x.copy(); q,u,v=p
            for l,r in [(0,1),(2,3),(0,2),(1,3),(1,2)]:
                a,b=pred[:,l].copy(),pred[:,r].copy()
                t=relu(0,q*(a-b)); pred[:,l]=a-u*t; pred[:,r]=b+v*t
            y=np.sort(x,axis=1)
        e=np.abs(y-pred)
        exact=(e==0) if e.ndim==1 else (e==0).all(1)
        out[split]=dict(n=len(exact),exact=float(exact.mean()),mae=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),max_abs=float(e.max()),max_scaled=float(np.max(e/np.maximum(1,np.abs(y)))))
    return dict(task=task,slots=len(p),**out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['generate','score'])
    if p.parse_args().action=='generate':
        for i in range(9): (HERE/f'case{i+13}.prompt.txt').write_text(setup(i)[1])
    else:
        rows=[]
        for i in range(9):
            raw=(HERE/f'case{i+13}.reply.txt').read_text().strip()
            if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
            try: rows.append(score(i,np.asarray(json.loads(raw),dtype=float)))
            except (ValueError,AssertionError) as err: rows.append(dict(task=TASKS[i],error='invalid vector'))
        (HERE/'task_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(rows)
