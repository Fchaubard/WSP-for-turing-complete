"""Vanilla tanh RNN on the same prefix-sum benchmark."""
import json
from pathlib import Path

import numpy as np

from run import Recurrence, data, fit, metrics, objective, pack, targets, tests


class TanhRNN(Recurrence):
    def forward(self,p,x):
        A,b,v = self.parts(p)
        N,T = x.shape
        h = np.empty((T,N,self.d))
        h[0] = np.tanh(x[:,0,None]*b)
        for t in range(1,T):
            h[t] = np.tanh(h[t-1] @ A.T + x[:,t,None]*b)
        return (h @ v).T,h

    def backward(self,p,x,h,err):
        A,b,v = self.parts(p)
        gA,gb = np.zeros_like(A),np.zeros_like(b)
        gv = np.einsum('nt,tnd->d',err,h)
        adj = np.zeros_like(h[0])
        for i in range(x.shape[1]-1,-1,-1):
            adj = (err[:,i,None]*v + adj @ A)*(1-h[i]**2)
            if i:
                gA += adj.T @ h[i-1]
            gb += x[:,i] @ adj
        return pack(gA,gb,gv)


def evaluate(model,p):
    return {name:metrics(model.forward(p,x)[0],targets('sum',x))
            for name,x,_ in tests('sum',model)}


def main():
    x = data(256,8,'ID',3330)
    rows = []
    for d in [1,2,4]:
        model = TanhRNN('sum',d)
        fun = objective(model,x,targets('sum',x),1.)
        p0 = model.init(101)
        eye = np.eye(len(p0))
        numeric = np.array([(fun(p0+1e-6*u)[0]-fun(p0-1e-6*u)[0])*5e5 for u in eye])
        grad = fun(p0)[1]
        audit = float(np.max(np.abs(grad-numeric)/np.maximum(1,np.abs(numeric))))
        assert audit < 1e-6
        for seed in [101,202,303,404,505]:
            initial = model.init(seed)
            p,info = fit(fun,initial,800)
            rows.append(dict(method='bp',width=d,seed=seed,fit=info,initial=initial.tolist(),
                             final=p.tolist(),gradient_audit=audit,scores=evaluate(model,p)))
        for exponent in [0,-8,-16,-24,-32,-40]:
            p = model.burn(model.init(101))
            A,b,v = model.parts(p)
            b[0],v[0] = 2.**exponent,2.**(-exponent)
            rows.append(dict(method='program',width=d,gain_exponent=exponent,final=p.tolist(),
                             scores=evaluate(model,p)))
    (Path(__file__).resolve().parent/'tanh_recurrence.json').write_text(
        json.dumps(rows,indent=1,allow_nan=False)+'\n')
    for d in [1,2,4]:
        bp = [r['scores']['OOD_T32'] for r in rows if r['width']==d and r['method']=='bp']
        pr = [(r['gain_exponent'],r['scores']['OOD_T32']['exact_all']) for r in rows
              if r['width']==d and r['method']=='program']
        print('d',d,'BP RMSE/whole exact',np.median([s['rmse'] for s in bp]),
              np.median([s['exact_all'] for s in bp]),'program whole exact',pr)


if __name__=='__main__':
    main()
