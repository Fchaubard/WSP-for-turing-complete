"""Train every stored weight of the same 83-stage affine/ReLU sine graph."""
from pathlib import Path
import json
import numpy as np
from compile import relu_sin_graph
from run import samples,labeled,rows_for,save

def backward(g,p,x,y):
    vals=[x]
    for op,*a in g.nodes[1:]:
        if op=="literal":q=a[0]
        elif op=="weight":q=p[a[0]]
        elif op=="relu":q=np.maximum(0,vals[a[0]])
        elif op=="add":q=vals[a[0]]+vals[a[1]]
        elif op=="sub":q=vals[a[0]]-vals[a[1]]
        elif op=="mul":q=vals[a[0]]*vals[a[1]]
        vals.append(q)
    r=vals[-1]-y
    adj=[0.]*len(vals)
    adj[-1]=2*r/len(y)
    grad=np.zeros_like(p)
    for i in range(len(vals)-1,0,-1):
        op,*a=g.nodes[i]
        v=adj[i]
        if op=="weight":grad[a[0]]=np.sum(v)
        elif op=="relu":adj[a[0]]=adj[a[0]]+v*(vals[a[0]]>0)
        elif op in ("add","sub"):
            adj[a[0]]=adj[a[0]]+v
            adj[a[1]]=adj[a[1]]+v if op=="add" else adj[a[1]]-v
        elif op=="mul":
            adj[a[0]]=adj[a[0]]+v*vals[a[1]]
            adj[a[1]]=adj[a[1]]+v*vals[a[0]]
    return np.mean(r*r),grad

def run():
    g=relu_sin_graph(-78,True)
    data=labeled(samples("sin"),"sin")
    train_x,train_y,_=data["train"]
    rows=[]
    reports=[]
    for seed in (11,22,33):
        rng=np.random.default_rng(seed)
        p=rng.uniform(-np.sqrt(3),np.sqrt(3),len(g.params))
        old=p.copy()
        first=np.zeros_like(p)
        second=np.zeros_like(p)
        grad_norm=[]
        for i in range(1,101):
            idx=rng.integers(len(train_x),size=32)
            loss,grad=backward(g,p,train_x[idx],train_y[idx])
            grad_norm.append(float(np.linalg.norm(grad)))
            first=.9*first+.1*grad
            second=.999*second+.001*grad**2
            p-=.01*(first/(1-.9**i))/(np.sqrt(second/(1-.999**i))+1e-8)
        rows_for(rows,"sin",f"all_slot_adam_seed_{seed}",g,83,-78,data,p)
        reports.append(dict(seed=seed,steps=100,batch=32,lr=.01,
                            weights_changed=int(np.count_nonzero(old!=p)),
                            max_gradient_norm=max(grad_norm)))
        save(rows,"train.csv")
        print(reports[-1],flush=True)
    (Path(__file__).parent/"train_stats.json").write_text(json.dumps(reports,indent=2)+"\n")
if __name__=="__main__":
    run()
