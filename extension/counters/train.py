"""Compare dense reverse-mode training and a structured data-only fit on register cells."""
import itertools
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from scipy.optimize import minimize
from run import PROGRAMS, compile_program, forward, initial, interpret, one_step_pairs, pack_state, shapes, trials

HERE = Path(__file__).resolve().parent
TASKS = ["add","subtract_floor","multiply"]
SEEDS = [11,22,33]


def split_central(nreg, program):
    x, y = one_step_pairs(nreg,program,range(4))
    rng = np.random.default_rng(20261106)
    train, dev = [], []
    pcs = np.argmax(x[:,nreg:],axis=1)
    for pc in range(len(program)):
        ids = rng.permutation(np.flatnonzero(pcs == pc))
        n = len(ids)//4
        dev.extend(ids[:n]); train.extend(ids[n:])
    return (x[train],y[train]),(x[dev],y[dev])


def pack(weights):
    return np.concatenate([arr.ravel() for arr in weights])


def unpack(p, dims):
    parts, start = [],0
    for shape in dims:
        stop = start+np.prod(shape)
        parts.append(p[start:stop].reshape(shape))
        start = stop
    return parts


def loss_grad(p, dims, x, y):
    D,W,E = unpack(p,dims)
    pre = x@W.T
    active = np.maximum(0, pre)
    dest = x@D.T + active@E.T
    err = np.maximum(0,dest) - y
    adj = (err/err.size)*(dest > 0)
    return float(np.mean(err*err)/2), pack([adj.T@x, ((adj@E)*(pre > 0)).T@x, adj.T@active])


def train(nreg, program, seed, train_set, dev_set):
    dims = shapes(nreg,program)
    x,y = train_set
    xv,yv = dev_set
    p = pack(initial(nreg,program,seed)); rng = np.random.default_rng(seed+901)
    direction = rng.normal(size=p.shape); direction /= np.linalg.norm(direction)
    _, g = loss_grad(p,dims,x,y)
    eps = 1e-6
    diff = (loss_grad(p+eps*direction,dims,x,y)[0]-loss_grad(p-eps*direction,dims,x,y)[0])/(2*eps)
    check_error = float(abs(diff - g@direction))
    assert check_error < 5e-8
    best_p = p.copy(); best_stage = "initial"
    best_dev = loss_grad(p,dims,xv,yv)[0]
    history = []

    def consider(stage, v):
        nonlocal best_p,best_stage,best_dev
        dev_loss = loss_grad(v,dims,xv,yv)[0]
        history.append([stage,dev_loss])
        if dev_loss < best_dev:
            best_p,best_stage,best_dev = v.copy(),stage,dev_loss

    m,v = np.zeros_like(p),np.zeros_like(p)
    for t in range(1,1201):
        ids = rng.choice(len(x), min(128,len(x)),replace=False)
        _,g = loss_grad(p,dims,x[ids],y[ids])
        m = .9*m + .1*g; v = .999*v + .001*g*g
        p -= .015*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        if t%200==0:
            consider(f"adam_{t}",p)
    # Warm start a full-batch quasi-Newton polish from the central-dev Adam checkpoint.
    lb = minimize(loss_grad,best_p,args=(dims,x,y),jac=True,method="L-BFGS-B",
                  options={"maxiter":400,"maxcor":15,"ftol":1e-15,"gtol":1e-12})
    consider("lbfgs",lb.x)
    return unpack(best_p,dims),dict(selected=best_stage,dev_half_mse=best_dev,
                                    lbfgs_steps=int(lb.nit),lbfgs_status=int(lb.status),
                                    gradient_check_absolute=check_error,history=history)


def structured(nreg, program_length, x, y):
    """Identify unknown instructions using central labels and the finite ISA only."""
    pc_ids = np.argmax(x[:,nreg:],axis=1)
    options = [("halt",)]
    options += [("inc",r,q) for r,q in itertools.product(range(nreg),range(program_length))]
    options += [("dec",r,q,s) for r,q,s in itertools.product(range(nreg),range(program_length),range(program_length))]
    found = []
    for pc in range(program_length):
        rows = np.flatnonzero(pc_ids==pc)
        ok = []
        for op in options:
            table = [op]*program_length
            for idx in rows:
                regs,dest = interpret(x[idx,:nreg].astype(int).tolist(),pc,table)
                if not np.array_equal(pack_state(regs,dest,program_length), y[idx]):
                    break
            else:
                ok.append(op)
        assert len(ok)==1, (pc,ok)
        found.append(ok[0])
    # Build only the recovered tests. Destination and register update weights are fitted from labels.
    tests = [(pc,op[1]) for pc,op in enumerate(found) if op[0]=="dec"]
    d = x.shape[1]
    W = np.zeros((len(tests),d))
    for j,(pc,r) in enumerate(tests):
        W[j,nreg+pc] = 1; W[j,r] = -1
    feats = np.column_stack((x,np.maximum(0,x@W.T)))
    beta,_,rank,_ = np.linalg.lstsq(feats,y,rcond=None)
    assert rank == feats.shape[1]
    integer = np.rint(beta)
    assert np.max(np.abs(beta-integer))<1e-9 and np.array_equal(feats@integer,y)
    return [integer[:d].T,W,integer[d:].T],dict(isa_candidates_per_pc=len(options),recovered=found,
              feature_rank=int(rank),columns=feats.shape[1],largest_snap=float(np.max(np.abs(beta-integer))))


def step_score(weights, dataset):
    x,y = dataset; pred = forward(x,weights)
    return dict(n=len(y),exact=int(np.all(pred==y,axis=1).sum()),
                rounded=int(np.all(np.rint(pred)==y,axis=1).sum()),rmse=float(np.sqrt(np.mean((pred-y)**2))))


def rollout(weights, task, values):
    nreg,prog,readout = PROGRAMS[task]
    stats=dict(n=0,whole_exact=0,whole_rounded=0,final_exact=0,final_rounded=0,max_ticks=0,
               ticks=0,teacher_rounded_steps=0,teacher_rounded_traces=0,projected_final=0)
    for regs,goal in trials(task,values):
        pc=0; state=pack_state(regs,pc,len(prog)); ref=state.copy(); projected=state.copy()
        ticks=0; exact=rounded=teacher=True
        while prog[pc][0]!="halt":
            regs,pc = interpret(regs,pc,prog)
            oracle_step = forward(ref,weights)
            state = forward(state,weights)
            projected = np.rint(forward(projected,weights))
            ref=pack_state(regs,pc,len(prog))
            teacher_ok = np.array_equal(np.rint(oracle_step),ref)
            stats["teacher_rounded_steps"] += int(teacher_ok)
            teacher &= teacher_ok
            exact &= np.array_equal(state,ref)
            rounded &= np.array_equal(np.rint(state),ref)
            ticks+=1
        stats["n"]+=1;stats["whole_exact"]+=int(exact);stats["whole_rounded"]+=int(rounded)
        stats["final_exact"]+=int(state[readout]==goal);stats["final_rounded"]+=int(np.rint(state[readout])==goal)
        stats["max_ticks"]=max(stats["max_ticks"],ticks)
        stats["ticks"]+=ticks;stats["teacher_rounded_traces"]+=int(teacher)
        stats["projected_final"]+=int(projected[readout]==goal)
    return stats


def main():
    states, rows = {}, []
    for task in TASKS:
        nreg,prog,_=PROGRAMS[task]
        train_set,dev_set=split_central(nreg,prog)
        test_sets={"train":train_set,"dev":dev_set,"outer":one_step_pairs(nreg,prog,range(4,11)),
                   "mixed":one_step_pairs(nreg,prog,[0,1,4,10,1024])}
        runs=[]
        for seed in SEEDS:
            weights,meta=train(nreg,prog,seed,train_set,dev_set)
            runs.append((f"bp_{seed}",weights,meta))
        inferred,meta=structured(nreg,len(prog),*train_set)
        assert meta["recovered"] == prog
        assert all(np.array_equal(a,b) for a,b in zip(inferred,compile_program(nreg,prog,SEEDS[0])))
        runs.append(("structured",inferred,meta))
        for method,weights,meta in runs:
            key=f"{task}_{method}"
            for name,w in zip("DWE",weights): states[f"{key}_{name}"]=w
            row=dict(task=task,method=method,slots=sum(a.size for a in weights),fit=meta,
                     step={split:step_score(weights,dataset) for split,dataset in test_sets.items()},
                     trace={split:rollout(weights,task,mags) for split,mags in [("central",range(4)),("outer",range(4,11))]})
            rows.append(row)
            print(task,method,row["step"]["outer"],row["trace"]["outer"],flush=True)
    result=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,seeds=SEEDS,
                split_seed=20261106,initialization="run.initial; all D,W,E entries Xavier uniform",
                rows=rows)
    (HERE/"bp_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(HERE/"bp_states.npz",**states)


if __name__=="__main__":
    main()
