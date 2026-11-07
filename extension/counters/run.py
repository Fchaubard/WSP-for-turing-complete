"""Compile a register program into a ReLU recurrent cell, then independently step it."""
import itertools
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
# Each row gives an instruction and explicit destination(s).
PROGRAMS={
    'add':(2,[('dec',1,2,1),('inc',0,0),('halt',)],0),
    'subtract_floor':(2,[('dec',1,2,1),('dec',0,2,0),('halt',)],0),
    'multiply':(4,[('dec',0,6,1),('dec',1,4,2),('inc',2,3),('inc',3,1),
                   ('dec',3,0,5),('inc',1,4),('halt',)],2),
    'triangle':(3,[('dec',0,7,1),('inc',1,2),('dec',0,5,3),('inc',1,4),
                   ('inc',2,2),('dec',2,0,6),('inc',0,5),('halt',)],1),
    'fib':(4,[('dec',0,8,1),('dec',1,3,2),('inc',3,1),('dec',2,6,4),
              ('inc',1,5),('inc',3,3),('dec',3,0,7),('inc',2,6),('halt',)],1),
}
def shapes(nreg,program):
    d=nreg+len(program); g=sum(op[0]=='dec' for op in program)
    return [(d,d),(g,d),(d,g)]

def initial(nreg,program,seed):
    rng=np.random.default_rng(seed)
    return [rng.uniform(-np.sqrt(6/(a+b)),np.sqrt(6/(a+b)),(a,b)) for a,b in shapes(nreg,program)]

def compile_program(nreg,program,seed):
    # All matrix entries are allocated and initialized before writing the instruction table.
    D,W,E=initial(nreg,program,seed)
    for arr in (D,W,E):arr.fill(0.)
    D[:nreg,:nreg]=np.eye(nreg)
    gate=0
    for j,op in enumerate(program):
        pc=nreg+j
        if op[0]=='halt':D[pc,pc]=1.
        elif op[0]=='inc':
            _,r,nxt=op
            D[r,pc]+=1.;D[nreg+nxt,pc]+=1.
        else:
            _,r,zero,positive=op
            D[r,pc]-=1.;D[nreg+positive,pc]+=1.
            W[gate,pc]=1.;W[gate,r]=-1.
            E[r,gate]=1.;E[nreg+zero,gate]+=1.;E[nreg+positive,gate]-=1.
            gate+=1
    return D,W,E

def forward(state,weights):
    D,W,E=weights
    return np.maximum(0,state@D.T+np.maximum(0,state@W.T)@E.T)

def pack_state(counters,pc,length):
    s=np.zeros(len(counters)+length)
    s[:len(counters)]=counters;s[len(counters)+pc]=1
    return s

def interpret(counters,pc,program):
    op=program[pc]; c=counters.copy()
    if op[0]=='halt':return c,pc
    r=op[1]
    if op[0]=='inc':
        c[r]+=1
        return c,op[2]
    if c[r]==0:return c,op[2]
    c[r]-=1
    return c,op[3]

def one_step_pairs(nreg,program,magnitudes):
    states=[]; targets=[]
    for values in itertools.product(magnitudes,repeat=nreg):
        for pc in range(len(program)):
            v,q=interpret(list(values),pc,program)
            states.append(pack_state(values,pc,len(program)))
            targets.append(pack_state(v,q,len(program)))
    return np.array(states),np.array(targets)

def trials(task,values):
    if task in ('add','subtract_floor','multiply'):
        for a,b in itertools.product(values,repeat=2):
            z=[a,b] if task!='multiply' else [a,b,0,0]
            answer={'add':a+b,'subtract_floor':max(a-b,0),'multiply':a*b}[task]
            yield z,answer
    else:
        for n in values:
            if task=='triangle': yield [n,0,0],n*(n+1)//2
            else:
                a,b=0,1
                for _ in range(n):a,b=b,a+b
                yield [n,0,1,0],a

def trace(task,values,weights):
    nreg,program,readout=PROGRAMS[task]; runs=[]
    for regs,answer in trials(task,values):
        pc=0; state=pack_state(regs,pc,len(program)); exact=True;steps=0
        # Reference chooses the evaluation horizon and never feeds a state to the neural cell.
        while program[pc][0]!='halt':
            regs,pc=interpret(regs,pc,program)
            with np.errstate(over='ignore',invalid='ignore'):
                state=forward(state,weights)
            exact &= np.array_equal(state,pack_state(regs,pc,len(program)))
            steps+=1
            if steps>100_000:raise RuntimeError('unexpected reference horizon')
        assert regs[readout]==answer
        runs.append((steps,bool(exact),bool(state[readout]==answer)))
    return dict(runs=len(runs),largest_ticks=max(r[0] for r in runs),ticks=sum(r[0] for r in runs),
                whole_trace=sum(r[1] for r in runs),final_output=sum(r[2] for r in runs))

def replay():
    out=[]
    for task,(nreg,program,_) in PROGRAMS.items():
        w=compile_program(nreg,program,215)
        scores={k:trace(task,v,w) for k,v in [('inner',range(4)),('outer',range(4,11))]}
        for s in scores.values(): assert s['runs']==s['whole_trace']==s['final_output']
        one={}
        for split,vals in [('inner',range(4)),('outer',range(4,11)),('boundary',[0,1,3,10,100,2**30])]:
            x,y=one_step_pairs(nreg,program,vals)
            count=int(np.sum(np.all(forward(x,w)==y,axis=1)))
            assert count==len(x)
            one[split]=dict(total=len(x),correct=count)
        out.append(dict(task=task,registers=nreg,pc=len(program),gates=len(w[1]),slots=sum(a.size for a in w),
                        nonzero=sum(int(np.count_nonzero(a)) for a in w),one_step=one,trajectories=scores))
    # One increment of a counter already at 2^53 cannot be represented in binary64.
    regs,prog,_=PROGRAMS['add']
    got=forward(pack_state([2**53,1],1,len(prog)),compile_program(regs,prog,9))
    assert int(got[0])==2**53 and int(got[0]) != 2**53+1
    (HERE/'results.json').write_text(json.dumps(dict(programs=out,overflow_witness=dict(initial=2**53,
            expected=2**53+1,output=int(got[0]))),indent=2)+'\n')
    print([(r['task'],r['trajectories']['outer']) for r in out])

if __name__=='__main__':replay()
