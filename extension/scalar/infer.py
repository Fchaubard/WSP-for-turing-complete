"""Evaluate a frozen program with NumPy only. Example: predict("selfpower", [2.5, 10.0])."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from graph import Graph

OPCODES=("x","literal","weight","add","sub","mul","relu")

@lru_cache(maxsize=2)
def load(name):
    if name not in ("sin","selfpower"):
        raise ValueError("Choose sin or selfpower")
    g=Graph()
    with np.load(Path(__file__).with_name("programs.npz"),allow_pickle=False) as data:
        table=data[name+"_ops"]
        p=data[name+"_weights"]
        fixed=data[name+"_fixed"]
    g.nodes=[]
    for idx,a,b in table.tolist():
        op=OPCODES[idx]
        args=() if op=="x" else (fixed[a],) if op=="literal" else (a,) if op in ("weight","relu") else (a,b)
        g.nodes.append((op,*args))
    return g,p

def predict(name,x):
    g,p=load(name)
    a=np.asarray(x,dtype=np.float64)
    valid=(a>=-16)&(a<=16) if name=="sin" else (a>=0)&(a<=128)
    if not np.all(valid):raise ValueError("Input outside the frozen program's declared domain")
    if a.size==0:return np.empty_like(a)
    result=g.run(a.reshape(-1),p).reshape(a.shape)
    return float(result) if a.ndim==0 else result

if __name__=="__main__":
    import sys
    print(predict(sys.argv[1],[float(x) for x in sys.argv[2:]]))
