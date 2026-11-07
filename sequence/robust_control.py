"""Same-topology BP for the 50-scalar comparator plus scale ablation."""
import json
from pathlib import Path

import numpy as np

from robust_sort import compile_numeric, evaluate, sort, template
from run import data, fit, metrics, objective, odd_even_network, targets

HERE = Path(__file__).resolve().parent


class SharedComparator:
    def __init__(self):
        self.structure = []
        pos = 0
        for edges,bias,relu in template():
            incoming = []
            for src,_ in edges:
                incoming.append((src,pos))
                pos += 1
            bias_slot = None
            if bias:
                bias_slot = pos
                pos += 1
            self.structure.append((incoming,bias_slot,relu))
        assert pos == 50

    def init(self,seed):
        return np.array(compile_numeric(template(),seed)[1]['values'])

    def forward(self,p,x):
        state,trace = x.copy(),[]
        for l,r in odd_even_network(x.shape[1]):
            values = [state[:,l].copy(),state[:,r].copy()]
            for edges,bias,relu in self.structure:
                z = p[bias] if bias is not None else 0.
                for src,slot in edges:
                    z = z + p[slot]*values[src]
                values.append(np.maximum(0.,z) if relu else z)
            state[:,l],state[:,r] = values[-2],values[-1]
            trace.append((l,r,values))
        return state,trace

    def backward(self,p,x,trace,err):
        adj = err.copy()
        grad = np.zeros_like(p)
        n = len(self.structure)
        for l,r,values in reversed(trace):
            back = np.zeros((n+2,x.shape[0]))
            back[-2],back[-1] = adj[:,l],adj[:,r]
            for j in reversed(range(n)):
                edges,bias,relu = self.structure[j]
                dz = back[j+2]
                if relu:
                    dz = dz*(values[j+2]>0)
                if bias is not None:
                    grad[bias] += dz.sum()
                for src,slot in edges:
                    grad[slot] += dz @ values[src]
                    back[src] += p[slot]*dz
            adj[:,l],adj[:,r] = back[0],back[1]
        return grad


def scores(predict):
    output = {}
    for t in (4,8):
        for domain in ['ID','OOD']:
            x = data(2048,t,domain,7001+t+(domain=='OOD'))
            yh = predict(x)
            output[f'{domain}_T{t}'] = metrics(yh,targets('sort',x))
    return output


def main():
    x = data(256,4,'ID',3330)
    y = targets('sort',x)
    model = SharedComparator()
    fun = objective(model,x,y,1.)
    model0 = model.init(101)
    unit = np.eye(len(model0))
    # ReLU at exactly zero has a conventional training derivative. At constant
    # zero structural subgraphs finite differences need a nondegenerate probe.
    probe = model0 + np.random.default_rng(11).normal(0,.17,len(model0))
    analytic = fun(probe)[1]
    numeric = np.array([(fun(probe+u*1e-6)[0]-fun(probe-u*1e-6)[0])*5e5 for u in unit])
    audit = float(np.max(np.abs(analytic-numeric)/np.maximum(1,np.abs(numeric))))
    assert audit < 2e-5, audit
    runs = []
    for seed in [101,202,303,404,505]:
        p0 = model.init(seed)
        p,info = fit(fun,p0,800)
        ss = scores(lambda x: model.forward(p,x)[0])
        runs.append(dict(seed=seed,initial=p0.tolist(),final=p.tolist(),fit=info,scores=ss))
        print('BP',seed,info['final_loss'],ss['OOD_T4']['exact_all'],
              ss['OOD_T4']['rmse'],flush=True)
    ordered = np.sort(x,axis=1)
    mingap = float(np.min(np.diff(ordered,axis=1)))
    # K^2 >= 1/min_gap means the selector must resolve every training order.
    ktrain = float(2.**np.ceil(.5*np.log2(1/mingap)))
    # Retain M=10 explicitly, as in the format-certified model.
    _,_,final = compile_numeric(template())
    graph = [( [(i,(ktrain if c=='K' else -ktrain) if isinstance(c,str) and 'K' in c
                  else final['M'] if c=='M' else c) for i,c in edges],bias,relu)
             for edges,bias,relu in template()]
    tiny = np.nextafter(0.,1.)
    lo,hi = evaluate(graph,np.array([tiny]),np.array([0.]))
    moderate_a,moderate_b = 1., 1.-1e-5
    moderate = evaluate(graph,np.array([moderate_a]),np.array([moderate_b]))
    ablation = dict(moderately_close_descending={'input':[moderate_a,moderate_b],
                                                 'output':[float(v[0]) for v in moderate]},
                    min_training_gap=mingap,K_train=ktrain,train=metrics(sort(graph,x),y),
                    scores=scores(lambda x:sort(graph,x)),
                    min_subnormal_descending={'input':[tiny,0.],'output':[float(lo[0]),float(hi[0])]})
    (HERE/'robust_control.json').write_text(json.dumps(dict(gradient_audit=audit,bp=runs,
                                                    gain_ablation=ablation),indent=1)+'\n')


if __name__ == '__main__':
    main()
