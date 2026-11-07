"""Paired specification/observations-only trials on ten tasks and three sizes."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ('square','sin','selfpower','tanh','add','subtract','sum','sort','fib','tri')
SMOOTH = TASKS[:4]
SERIES = ('fib','tri')
SEEDS = (164,317,583)
FORMULAS = dict(square='y=x squared', sin='y=sin(x), with x in radians', selfpower='y=x raised to x on x>0',
                tanh='y=tanh(x)', add='y=x1+x2', subtract='y=x1-x2',
                sum='return every inclusive prefix sum', sort='sort the four inputs into ascending order',
                fib='Fibonacci numbers, F0=0, F1=1', tri='triangular numbers, y[n]=n*(n+1)/2')
SIZES = {t:(1,4,16) for t in SMOOTH}
SIZES.update(add=(1,2,3),subtract=(1,2,3),sum=(0,1,2),sort=(4,5,6),fib=(1,2,3),tri=(2,3,4))

def reference(task,x):
    if task=='square': return x*x
    if task=='sin': return np.sin(x)
    if task=='selfpower': return x**x
    if task=='tanh': return np.tanh(x)
    if task=='add': return x[:,0]+x[:,1]
    if task=='subtract': return x[:,0]-x[:,1]
    if task=='sum': return np.cumsum(x,axis=1)
    if task=='sort': return np.sort(x,axis=1)
    if task=='tri': return np.asarray([int(n)*(int(n)+1)//2 for n in x],dtype=float)
    f=[0,1]
    for _ in range(2,int(max(x))+1): f.append(f[-1]+f[-2])
    return np.asarray([f[n] for n in x],dtype=float)

def observed(task,seed):
    rng=np.random.default_rng(seed+442)
    if task in SERIES: x=np.arange(9)
    elif task in SMOOTH:
        x=rng.uniform(.02 if task=='selfpower' else -3,3,12)
    else:
        width = 2 if task in ('add','subtract') else 4 if task=='sort' else 6
        x=rng.uniform(-3,3,(12,width))
    return x,reference(task,x)

def initial(task,size,seed):
    rng=np.random.default_rng(seed+224+size)
    def w(n,m): return rng.uniform(-np.sqrt(6/(n+m)),np.sqrt(6/(n+m)),(m,n))
    if task in SMOOTH: return np.r_[w(1,1).ravel(),0,w(1,size).ravel(),np.zeros(size),w(size,1).ravel()]
    if task in ('add','subtract'):
        return np.r_[w(min(2,size),1).ravel(),np.zeros(int(size==3))]
    if task=='sort':
        return np.concatenate([np.r_[w(4,1).ravel(),w(1,4).ravel()] for _ in range(size)])
    if size==0: return w(1,1).ravel()
    return np.r_[w(size,size).ravel(),w(1,size).ravel(),w(size,1).ravel()]

def arch(task,s):
    if task in SMOOTH: return f'scalar residual ReLU: a*x+b+sum(v*ReLU(w*x+c)), {s} hinges. Flatten [a,b, all {s} w, all {s} c, all {s} v].'
    if task in ('add','subtract'):
        return ['y=w0*x1; only one coefficient.','y=w0*x1+w1*x2. Flatten [w0,w1].','y=w0*x1+w1*x2+b. Flatten [w0,w1,b].'][s-1]
    if task=='sort':
        return (f'Input h is the 4-vector. Repeat {s} residual blocks; h <- h+v*ReLU(w dot h). '
                'Output h directly. Flatten blocks in application order, each [w0,w1,w2,w3,v0,v1,v2,v3]. No biases.')
    if task=='sum' and not s: return 'Memoryless scalar y[t]=w*x[t]. Flatten [w].'
    layout=f'd={s}; flatten A ({s} by {s}, row-major), then vector b, then vector v. '
    if task in SERIES: return layout+'h[0]=b, h[n+1]=A @ h[n], y[n]=v dot h[n].'
    return layout+'h initially zero; h <- A @ h+b*x[t]; output v dot h at every tick.'

def predict(task,size,p,x):
    if task in SMOOTH:
        a,b=p[:2]; w,c,v=p[2:].reshape(3,size)
        return a*x+b+np.maximum(x[:,None]*w+c,0)@v
    if task in ('add','subtract'):
        return x[:,:min(2,size)]@p[:min(2,size)]+(p[2] if size==3 else 0)
    if task=='sort':
        h=x.copy()
        for w,v in p.reshape(size,2,4): h=h+np.maximum(h@w,0)[:,None]*v
        return h
    if size==0: return p[0]*x
    d=size; A=p[:d*d].reshape(d,d); b=p[d*d:d*d+d]; v=p[-d:]
    if task in SERIES:
        vals=[]; h=b.copy()
        for _ in range(int(max(x))+1):
            vals.append(h@v); h=A@h
        return np.asarray(vals)[x]
    y=[];h=np.zeros((len(x),d))
    for u in x.T:
        h=h @ A.T+u[:,None]*b
        y.append(h @ v)
    return np.asarray(y).T

def inputs(task):
    if task in SERIES: return {'fit_indices':np.arange(9),'early':np.arange(9,33),'outer':np.arange(33,65)}
    rng=np.random.default_rng(912751)
    shape=(2048,) if task in SMOOTH else (2048,2 if task in ('add','subtract') else 4 if task=='sort' else 6)
    result={'inner':rng.uniform(0 if task=='selfpower' else -3,3,shape)}
    if task in SMOOTH:
        if task!='selfpower':result['left']=rng.uniform(-10,-3,shape)
        result['right']=rng.uniform(3,10,shape)
        result['outer']=np.concatenate([result[k] for k in ('left','right') if k in result])
    else: result['outer']=rng.uniform(3,10,shape)*rng.choice([-1,1],shape)
    if task=='sum':
        result['length']=rng.uniform(-3,3,(2048,32))
        result['joint']=rng.uniform(3,10,(2048,32))*rng.choice([-1,1],(2048,32))
    return result

def make_prompt(task,seed,arm):
    x,y=observed(task,seed)
    dom=('Inputs are positive reals up to 10.' if task=='selfpower' else
         'Input indices are nonnegative integers up to 64.' if task in SERIES else
         'Coordinate values are real numbers in [-10,10].')
    rule=('The target specification is '+FORMULAS[task]+'. ' if arm=='formula' else '')
    examples=json.dumps([[a,b] for a,b in zip(x.tolist(),y.tolist())],separators=(',',':'))
    lead=('You are writing neural-network coefficients, in binary64. Three architectures below are '
          'independent size variants of one unknown map. '+dom+' '+rule+
          'The only labeled observations provided are '+examples+
          '. Aim for exact outputs on unseen inputs/indices including outside the observation range. '
          'When exact realization is impossible, give your best approximation. Prefer small maximum '
          'absolute error divided by max(1,abs(target)). You may infer a rule from observations. '
          'Do not use any tools or external files. Return ONLY JSON of shape [vector0,vector1,vector2], '
          'each a complete numeric array of the stated flattened length. Do not return expressions, code or ellipses.\n')
    for j,s in enumerate(SIZES[task]):
        p=initial(task,s,seed)
        lead+=f'Variant {j} ({len(p)} coefficients): {arch(task,s)} Initial weights: '+json.dumps(p.tolist(),separators=(',',':'))+'\n'
    return lead

def cases():
    for task in TASKS:
        for seed in SEEDS:
            for arm in ('formula','examples'):
                yield task,seed,arm

if __name__=='__main__':
    import hashlib
    root=HERE/'trials'
    root.mkdir(exist_ok=True)
    manifest=[]
    for t,s,a in cases():
        case_id=hashlib.sha256(f'{t}:{s}:{a}'.encode()).hexdigest()[:10]
        prompt=make_prompt(t,s,a)
        (root/f'{case_id}.prompt.txt').write_text(prompt)
        manifest.append(dict(id=case_id,task=t,seed=s,arm=a,sizes=SIZES[t]))
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(len(manifest),'prompts')
