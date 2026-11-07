"""Backprop on real-equivalent width-one recurrent realizations."""
import json
from pathlib import Path

import numpy as np

from run import Recurrence, data, fit, objective, targets
from synthesize import score


def main():
    model = Recurrence('sum',1)
    x = data(256,8,'ID',3330)
    fun = objective(model,x,targets('sum',x),1.)
    records = []
    for kind in ['quotient_xavier','quotient_mapped','fixed_output']:
        active = [0] if kind=='fixed_output' else [0,2]
        def expand(p):
            triple = np.ones(3)
            triple[active] = p
            return triple
        def restricted(p):
            loss,grad = fun(expand(p))
            return loss,grad[active]
        for seed in [101,202,303,404,505]:
            p = model.init(seed)
            if kind=='quotient_xavier':
                # First two independent draws from the same Xavier stream.
                start = p[:2]
            elif kind=='quotient_mapped':
                start = np.array([p[0],p[1]*p[2]])
            else:
                start = p[:1]
            final,info = fit(restricted,start,800)
            records.append(dict(kind=kind,seed=seed,initial=start.tolist(),final=final.tolist(),
                                fit=info,scores=score(model,expand(final),'sum')))
    path = Path(__file__).resolve().parent/'gauge_control.json'
    path.write_text(json.dumps(records,indent=1,allow_nan=False)+'\n')
    for kind in ['quotient_xavier','quotient_mapped','fixed_output']:
        rs = [r for r in records if r['kind']==kind]
        print(kind,[r['scores']['OOD_T32']['exact_all'] for r in rs])


if __name__=='__main__':
    main()
