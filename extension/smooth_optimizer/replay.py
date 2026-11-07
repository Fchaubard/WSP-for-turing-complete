"""Confirm archived scalar starts, terminal coefficients, reported losses and exact counts."""
import json
import numpy as np
from bench import ROOT,load_graph,measure,reference,samples

for task in ('selfpower','tanh'):
    g,w=load_graph(task)
    data=samples(task)
    data['outer']=np.r_[data['left'],data['right']] if task=='tanh' else data['right']
    data={k:(x,reference(x,task,53)) for k,x in data.items() if k in ('train','inner','outer')}
    scores=lambda p:{k:measure(g,p,*xy) for k,xy in data.items()}
    records=json.loads((ROOT/(task+'.json')).read_text())
    with np.load(ROOT/f'{task}_weights.npz',allow_pickle=False) as saved:
        assert np.array_equal(w,saved['written'])
        assert scores(w)==records[0]['scores']
        for r in records[1:]:
            seed=r['seed']
            p0,p1=saved[f'initial_{seed}'],saved[f'terminal_{seed}']
            assert np.array_equal(p0,np.random.default_rng(seed).uniform(-np.sqrt(3),np.sqrt(3),len(w)))
            assert scores(p0)==r['initial'] and scores(p1)==r['final']
            assert np.count_nonzero(p0!=p1)==r['moved_slots']
    print(task,'written plus three initial/terminal pairs reproduced')
