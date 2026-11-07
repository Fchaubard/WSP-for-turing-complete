"""Replay weight writers in a bounded arithmetic DSL. No model-provided Python is run."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(9213,2451,8282)
TASKS=('square','sin','selfpower')
GRAMMAR="""Write a small, straight-line neural circuit whose input wire is x and final wire is y. Return ONLY a JSON array of instructions. Syntax: ["variable","expression"] assigns an expression. ["for","i",start,stop,body] repeats its instruction array body with integer i=start,...,stop-1. Loop bounds are integer literals with at most 256 iterations. Named scalar variables may be reassigned. Available expression syntax: numeric literals, variables, parentheses, +,-,*,/,**, and relu(a). Division allowed only by a compile-time constant. Powers ** allowed only between compile-time constants. On compile-time constants ONLY, you may use sin,cos,exp,log,sqrt,factorial,abs and constant pi. No lists or subscripts inside expressions; no if/test/branch on x, input rounding, table lookup or implicit target operation. Your program fixes all coefficients and connectivity before any x is supplied. Each binary operation on a wire rounds separately to binary64. Constant expressions are evaluated as ordinary Python arithmetic/math at compile time. Return at most 4096 expanded input-dependent operations. Example syntax for a simple ramp: [["a","0.25"],["y","relu(x-a)"]]. The evaluator never executes freeform Python. Aim to minimize error in ULPs against binary64 rounding of the true function on unseen inputs. No tools, no test feedback. You may use loops to emit a large repeated circuit. """
def variants():
    for task in TASKS:
        for mode in ('affine_relu','arithmetic_relu'):
            for seed in SEEDS:
                cid=hashlib.sha256(f'{task}/{mode}/{seed}'.encode()).hexdigest()[:9]
                yield dict(id=cid,task=task,arm=mode,seed=seed)

def generate():
    (HERE/'trials').mkdir(exist_ok=True)
    for q in variants():
        rule={'square':'square x*x on [-10,10]','sin':'sine in radians on [-10,10]',
              'selfpower':'x raised to x for real x in [0,10], using the continuous extension 1 at 0'}[q['task']]
        extra=('Multiplying two input-dependent wires is FORBIDDEN. You have ordinary affine operations and ReLU only. ' if q['arm']=='affine_relu' else
               'Multiplication of two input-dependent wires is also allowed (a product neuron). ')
        prompt=GRAMMAR+extra+'Target: '+rule+'. Blind replicate label '+str(q['seed'])+'.\n'
        (HERE/'trials'/(q['id']+'.prompt.txt')).write_text(prompt)
    (HERE/'manifest.json').write_text(json.dumps(list(variants()),indent=2)+'\n')

class Compiler:
    # Values are tagged either fixed (coefficient) or input-dependent wires.
    def __init__(self,mode):
        self.mode=mode;self.ops=[];self.env={'x':('wire',0),'pi':('const',math.pi)}
        self.ticks=0
    def expression(self,n):
        if isinstance(n,ast.Expression):return self.expression(n.body)
        if isinstance(n,ast.Name):return self.env[n.id]
        if isinstance(n,ast.Constant) and type(n.value) in (float,int):return ('const',n.value)
        if isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UnaryOp,ast.USub,ast.UAdd):
            a=self.expression(n.operand)
            return self.binary('mul',('const',-1),a) if isinstance(n.op,ast.USub) else a
        if isinstance(n,ast.BinOp):
            op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div',ast.Pow:'pow'}[type(n.op)]
            return self.binary(op,self.expression(n.left),self.expression(n.right))
        if isinstance(n,ast.Call) and len(n.args)==1 and not n.keywords and isinstance(n.func,ast.Name):
            name=n.func.id;a=self.expression(n.args[0])
            if a[0]=='wire':
                if name!='relu':raise ValueError('function on input-dependent wire: '+name)
                return self.emit('relu',a,('const',0))
            functions={k:getattr(math,k) for k in ('cos','sin','exp','log','sqrt','factorial')}
            functions.update(relu=lambda t:max(t,0),abs=abs)
            if name=='factorial' and not 0<=a[1]<=350:raise ValueError('factorial range')
            return ('const',functions[name](a[1]))
        raise ValueError('outside grammar: '+type(n).__name__)
    def emit(self,op,a,b):
        if len(self.ops)>=4096:raise ValueError('operation budget')
        self.ops.append((op,a,b))
        return ('wire',len(self.ops))
    def binary(self,op,a,b):
        if a[0]==b[0]=='const':
            x,y=a[1],b[1]
            if op=='pow' and abs(y)>1100:raise ValueError('exponent range')
            if max(abs(x),abs(y))>10**700:raise ValueError('constant range')
            z={'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y,'pow':lambda:x**y}[op]()
            if isinstance(z,complex):raise ValueError('complex coefficient')
            return ('const',z)
        if op=='div':
            if b[0]!='const':raise ValueError('wire divisor')
            return self.emit('mul',a,('const',1/b[1]))
        if op=='pow':raise ValueError('wire exponentiation')
        if op=='mul' and a[0]==b[0]=='wire' and self.mode=='affine_relu':
            raise ValueError('wire product in affine network')
        return self.emit(op,a,b)
    def block(self,block,depth=0):
        if not isinstance(block,list) or depth>5:raise ValueError('block')
        for ins in block:
            self.ticks+=1
            if self.ticks>30000:raise ValueError('builder budget')
            if not isinstance(ins,list):raise ValueError('instruction')
            if len(ins)==5 and ins[0]=='for':
                _,i,lo,hi,body=ins
                if type(lo)!=int or type(hi)!=int or hi-lo>256:raise ValueError('loop bounds')
                for it in range(lo,hi):
                    self.env[i]=('const',it);self.block(body,depth+1)
            elif len(ins)==2 and isinstance(ins[0],str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',ins[0]):
                self.env[ins[0]]=self.expression(ast.parse(str(ins[1]),mode='eval'))
            else:raise ValueError('invalid instruction')
    def run(self,x):
        wires=[x];f={'relu':lambda a,b:np.maximum(a,0),'add':np.add,'sub':np.subtract,'mul':np.multiply}
        def v(pair):return wires[pair[1]] if pair[0]=='wire' else float(pair[1])
        with np.errstate(all='ignore'):
            for op,a,b in self.ops:wires.append(f[op](v(a),v(b)))
        return np.broadcast_to(v(self.env['y']),x.shape)

def samples(task):
    rng=np.random.default_rng(131490)
    x=rng.uniform(0 if task=='selfpower' else -3,3,4096)
    shell=rng.uniform(3,10,16384)
    if task!='selfpower':shell*=rng.choice([-1,1],len(shell))
    return dict(inner=x,outer=shell)

def score():
    import gmpy2 as mp
    oracle={}
    for t in TASKS:
        for spl,x in samples(t).items():
            with mp.context(precision=53):
                def ref(v):
                    z=mp.mpfr(float(v))
                    return float(mp.sin(z) if t=='sin' else z**z if t=='selfpower' else z*z)
                oracle[t,spl]=(x,np.array([ref(v) for v in x]))
    rows=[]
    for q in variants():
        row=dict(q)
        try:
            raw=(HERE/'trials'/(q['id']+'.reply.txt')).read_text().strip()
            if raw.startswith(chr(96)*3):raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
            c=Compiler(q['arm']);c.block(json.loads(raw))
            row['ops']=len(c.ops)
            for split in ['inner','outer']:
                x,y=oracle[q['task'],split]; yh=c.run(x)
                valid=np.isfinite(yh)
                error=np.abs(yh-y)
                row[split]=dict(n=len(x),exact=int((yh==y).sum()),
                    ulp1=int(((yh>=np.nextafter(y,-np.inf))&(yh<=np.nextafter(y,np.inf))).sum()),
                    finite=int(valid.sum()),mae=float(error.mean()) if valid.all() else None,
                    max_scaled=float(np.max(error/np.maximum(1,np.abs(y)))) if valid.all() else None)
            row['status']='scored'
        except Exception as exc:
            row['status']='missing' if isinstance(exc,FileNotFoundError) else 'invalid'
            row['reason']=str(exc)[:240]
        rows.append(row)
    (HERE/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print([(r['task'],r['arm'],r['status'],r.get('outer',{}).get('exact')) for r in rows])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','score'])
    if p.parse_args().action=='generate':generate()
    else:score()
