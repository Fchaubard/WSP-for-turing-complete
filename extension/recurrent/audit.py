"""Replay all released model states and bound error from the learned readout."""
import csv
import hashlib
import json

import numpy as np
from run import ROOT, Network, dataset, evaluate, lengths, parameter_arrays, restore, setup


def main():
    setup()
    bounds = []
    for base in [ROOT,ROOT/'longer']:
        report = json.loads((base/'results.json').read_text())
        checked = 0
        with np.load(base/'weights.npz',allow_pickle=False) as archive:
            for r in report['rows']:
                task,cell,d,seed = [r[k] for k in ('task','cell','width','seed')]
                m=Network(task,cell,d)
                tag=f'{task}_{cell}_{d}_{seed}/'
                m.reset_xavier(seed)
                initial = parameter_arrays(m)
                digest = hashlib.sha256(b''.join(a.tobytes() for a in initial.values())).hexdigest()
                assert digest==r['initial_sha256']
                for name,a in initial.items():
                    assert np.array_equal(a,archive[tag+'init_'+name])
                state = {k:archive[tag+r['method']+'_'+k] for k in initial}
                restore(m,state)
                observed=evaluate(m)
                for name,stats in r['metrics'].items():
                    for k,val in stats.items():
                        assert observed[name][k] == val, (task,cell,d,seed,name,k)
                checked += 1
                if r['method']!='bp':
                    continue
                # A tanh readout state lies in [-1,1], and a GRU is a convex
                # mixture of its previous bounded state and bounded candidate.
                # This diagnostic evaluates the real-arithmetic bound on saved
                # weights. Last-bit readout rounding has negligible effect on the
                # reported floors, which we also compare to actual errors.
                limit = float(np.abs(state['readout.weight']).sum()+np.abs(state['readout.bias']).sum())
                T=8 if task=='sort' else 32
                _,y=dataset(task,T,1024,'outer',7777)
                target=y.numpy()
                bound_error = np.maximum(0,np.abs(target)-limit)
                floor=float(np.sqrt(np.mean(bound_error**2)/np.mean(target**2)))
                assert floor <= observed['joint']['nrmse']+1e-12
                bounds.append(dict(steps=report['settings']['steps'],task=task,cell=cell,
                                   width=d,seed=seed,bound=limit,
                                   outside_fraction=float(np.mean(np.abs(target)>limit)),
                                   joint_nrmse_floor=floor,observed_nrmse=observed['joint']['nrmse']))
        print(base.name,checked,'full checkpoint replays verified.')
    with (ROOT/'readout_bounds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bounds[0]))
        w.writeheader()
        w.writerows(bounds)


if __name__=='__main__':
    main()
