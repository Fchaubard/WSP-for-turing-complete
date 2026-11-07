"""NumPy-only execution of the frozen affine/ReLU binary64 tanh circuit."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'scalar'))
from graph import Graph

OPCODES=('x','literal','weight','add','sub','mul','relu')


@lru_cache(maxsize=1)
def load():
    path=ROOT/'program.npz'
    info=json.loads((ROOT/'manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info['checkpoint_sha256']:
        raise ValueError('Tanh checkpoint signature mismatch')
    with np.load(path,allow_pickle=False) as archive:
        table,p,fixed=archive['ops'],archive['weights'],archive['fixed']
    g=Graph()
    g.nodes=[]
    static=[]
    for idx,a,b in table.tolist():
        tag=OPCODES[idx]
        args=() if tag=='x' else (fixed[a],) if tag=='literal' else (a,) if tag in ('weight','relu') else (a,b)
        g.nodes.append((tag,*args))
        if tag=='x':
            static.append(False)
        elif tag in ('literal','weight'):
            static.append(True)
        else:
            if tag=='mul' and not (static[a] or static[b]):
                raise ValueError('Dynamic coefficient in archived graph')
            static.append(all(static[i] for i in args))
    return g,p


def predict(x):
    x=np.asarray(x,np.float64)
    if not np.all((x>=-10)&(x<=10)):
        raise ValueError('The tanh graph accepts finite binary64 inputs in [-10,10]')
    if not x.size:
        return np.empty_like(x)
    g,p=load()
    y=g.run(x.reshape(-1),p,batch=2048).reshape(x.shape)
    return float(y) if x.ndim==0 else y


if __name__=='__main__':
    def number(s):
        return float.fromhex(s) if s.lower().lstrip('+-').startswith('0x') else float(s)
    print(predict([number(v) for v in sys.argv[1:]]))
