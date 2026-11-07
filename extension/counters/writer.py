"""First-response Astra trial on unseen instruction tables and state permutations."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from run import initial,one_step_pairs,forward,shapes,compile_program
HERE=Path(__file__).resolve().parent/'literal'

def build(seed):
    rng=np.random.default_rng(seed)
    prog=[('dec',0,3,2),('inc',1,0),('dec',1,1,0),('halt',)]
    # Relabel instructions and registers while preserving branch structure.
    mapping=rng.permutation(4)
    if seed%2: regs=[1,0]
    else: regs=[0,1]
    out=[None]*4
    for i,op in enumerate(prog):
        if op[0]=='halt':out[mapping[i]]=op
        else:out[mapping[i]]=(op[0],regs[op[1]],*[int(mapping[j]) for j in op[2:]])
    # Some variants carry new transitions too.
    if seed in [901,902,903]:
        k=int(mapping[0])
        o=out[k]; out[k]=(*o[:2],int(mapping[1]),int(mapping[3]))
    perm=rng.permutation(6)
    return out,perm

def generate():
    (HERE/'trials').mkdir(parents=True,exist_ok=True);manifest=[]
    for seed in range(898,904):
        prog,perm=build(seed)
        names=['R0','R1','P0','P1','P2','P3']
        xw=initial(2,prog,seed)
        prompt=('Write literal weights into a recurrent neural network to execute one instruction of a register machine at each tick. '
            'The state z contains two nonnegative integer registers R0,R1 and four one-hot program counter bits P0..P3. '
            'State vector ORDER is '+json.dumps([names[i] for i in perm])+'. '
            'One tick is z_new=ReLU(D @ z + E @ ReLU(W @ z)), ReLU componentwise. '
            'D is 6x6, W is 2x6, E is 6x2. The returned z_new has the same coordinate order. '
            'At active instruction pc j, inc(r,t) means increment register r and jump to t; dec(r,z,p) means if r=0 jump to z, '
            'otherwise subtract one from r and jump to p. halt means keep everything unchanged. All other registers retain their values. '
            'The program in pc order 0 to 3 is '+json.dumps(prog)+'. '
            'Write the cell for all register values and all four starting PCs. Return ONLY JSON object with keys D,W,E, whose values '
            'are arrays of [zero-based row,zero-based col,value] triples. Unlisted entries of these FINAL matrices are set to zero. '
            'No tools or code in the reply. Initial Xavier matrices in D,W,E order: '+json.dumps([a.tolist() for a in xw])+'\n')
        cid=f'm{seed}'
        (HERE/'trials'/f'{cid}.prompt.txt').write_text(prompt)
        manifest.append(dict(id=cid,task='machine',arm='program',seed=seed))
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def score():
    rows=[]
    for entry in json.loads((HERE/'manifest.json').read_text()):
        prog,perm=build(entry['seed']);res=dict(seed=entry['seed'])
        p=HERE/'trials'/(entry['id']+'.reply.txt')
        try:
            raw=p.read_text().strip()
            if raw.startswith(chr(96)*3):raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
            z=json.loads(raw); weights=[]
            for key,sh in zip(['D','W','E'],shapes(2,prog)):
                arr=np.zeros(sh); used=set()
                for r,c,v in z[key]:
                    assert (r,c) not in used and 0<=r<sh[0] and 0<=c<sh[1] and np.isfinite(v)
                    arr[r,c]=v;used.add((r,c))
                weights.append(arr)
            res['valid']=True
            # The outer ReLU itself clips an attempted decrement of zero.
            # Compare to both certified forms, up to hidden-gate permutation.
            canonical=list(compile_program(2,prog,entry['seed']))
            def agrees(m):
                D,W,E=m
                return any(all(np.array_equal(a,b) for a,b in zip(weights,
                    [D[perm][:,perm], W[list(order)][:,perm], E[perm][:,list(order)]]))
                    for order in itertools.permutations(range(W.shape[0])))
            covered=agrees(canonical)
            E=canonical[2].copy()
            for gate,op in enumerate(z for z in prog if z[0]=='dec'): E[op[1],gate]-=1
            lean=agrees([*canonical[:2],E])
            res['algebraic_certificate']='output_relu_clips_decrement' if lean else 'explicit_counter_restore' if covered else None
            for split,values in [('central',range(4)),('outer',range(4,11)),('mixed',[0,1,4,10,1024,2**30])]:
                x,y=one_step_pairs(2,prog,values)
                yh=forward(x[:,perm],weights)
                res[split]=dict(correct=int((yh==y[:,perm]).all(axis=1).sum()),n=len(y))
            # Recurrent replay for every initialization, including nonhalting cycles.
            x,_=one_step_pairs(2,prog,[0,1,4,10]);pred=x[:,perm];ok=np.ones(len(x),bool)
            from run import interpret,pack_state
            states=[(list(map(int,a[:2])),int(np.argmax(a[2:]))) for a in x]
            for _ in range(128):
                pred=forward(pred,weights)
                states=[interpret(c,q,prog) for c,q in states]
                target=np.array([pack_state(c,q,4) for c,q in states])[:,perm]
                ok &= (target==pred).all(axis=1)
            res['trace128']=dict(correct=int(ok.sum()),n=len(x))
        except (FileNotFoundError,AssertionError,ValueError,TypeError,KeyError,IndexError) as exc:
            res['valid']=False
            res['failure']=type(exc).__name__
        rows.append(res)
    (HERE/'scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','score'])
    if p.parse_args().action=='generate':generate()
    else:score()
