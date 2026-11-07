"""Benchmark native tanh and emitted affine/ReLU programs against MPFR."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import gmpy2 as mp
from circuit import tanh_graph, static_input_check

ROOT=Path(__file__).resolve().parent
URL=('https://raw.githubusercontent.com/mockingbirdnest/core-math/'
     '3ce26eb72b71ffe95784e42e280149a8c7e015d5/src/binary64/tanh/tanh.wc')
DIGEST='4fc9a5682cedd0b79e624c7cd236bc88f6965fe8a0e88ca401ca938acd293bdf'
OPCODES=('x','literal','weight','add','sub','mul','relu')
FINAL_BITS=120


def reference(x,bits):
    with mp.context(precision=bits,round=mp.RoundToNearest):
        return np.array([float(mp.tanh(mp.mpfr(v))) for v in x])


def data(payload):
    if hashlib.sha256(payload).hexdigest()!=DIGEST:
        raise ValueError('CORE-MATH corpus digest mismatch.')
    rng=np.random.default_rng(90277)
    splits={'inner':rng.uniform(-3,3,4096),'left':rng.uniform(-10,-3,4096),
            'right':rng.uniform(3,10,4096)}
    u=rng.uniform(3,10,32768)
    splits['fresh_outer']=u*rng.choice([-1.,1.],len(u))
    for name,lo,hi in [('mid_inner',0.,3.),('mid_outer',3.,10.)]:
        found=set()
        with mp.context(precision=240):
            for a in rng.uniform(lo,hi,768):
                y=float(mp.tanh(mp.mpfr(a)))
                mid=(mp.mpfr(y)+mp.mpfr(np.nextafter(y,np.inf)))/2
                site=float(mp.atanh(mid))
                for pos in [np.nextafter(site,-np.inf),site,np.nextafter(site,np.inf)]:
                    if lo<pos<hi: found.update([pos,-pos])
        splits[name]=np.array(sorted(found))
    values=set()
    for line in payload.decode().splitlines():
        try: x=float.fromhex(line)
        except ValueError: continue
        if 0<abs(x)<=10: values.update([x,-x])
    hard=np.array(sorted(values))
    splits['core_inner']=hard[np.abs(hard)<=3]
    splits['core_outer']=hard[np.abs(hard)>3]
    exps=np.array([-1074,-1022,-500,-100,-55,-54,-29,-28,-27,-26,-25,0,2,3])
    anchor=np.unique(np.r_[0.,-0.,10.,np.ldexp(np.ones(len(exps)),exps)])
    near=np.unique(np.r_[anchor,np.nextafter(anchor,0),np.nextafter(anchor,np.inf)])
    splits['anchors']=np.unique(np.r_[near,-near])
    splits['anchors']=splits['anchors'][np.abs(splits['anchors'])<=10]
    return splits


def ulps(v,y):
    def ordered(x):
        u=np.ascontiguousarray(x).view(np.uint64)
        s=np.uint64(1<<63)
        return np.where(u&s,~u+np.uint64(1),u|s)
    a,b=ordered(v),ordered(y)
    return np.maximum(a,b)-np.minimum(a,b)


def measures(value,y):
    d=ulps(value,y)
    return dict(n=len(y),exact=int((d==0).sum()),one_ulp=int((d<=1).sum()),
                worst_ulp=int(d.max()),mae=float(np.abs(value-y).mean()))


def export(g):
    table=np.zeros((len(g.nodes),3),dtype=np.int32)
    fixed=[]
    static=[]
    occurrences=0
    for i,(op,*args) in enumerate(g.nodes):
        table[i,0]=OPCODES.index(op)
        if op=='literal':
            table[i,1]=len(fixed)
            fixed.append(args[0])
        else:
            table[i,1:1+len(args)]=args
        if op=='x': is_static=False
        elif op in ('literal','weight'): is_static=True
        else:
            is_static=all(static[j] for j in args)
            if not is_static:
                occurrences+=2 if op in ('add','sub') else 1 if op=='mul' else 0
        static.append(is_static)
    np.savez_compressed(ROOT/'program.npz',ops=table,weights=g.weights(),fixed=np.array(fixed))
    return dict(**g.summary(),**static_input_check(g),unshared_affine_occurrences=occurrences)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core-file',type=Path,help='Use the pinned source locally instead of downloading')
    args=ap.parse_args()
    payload=args.core_file.read_bytes() if args.core_file else urlopen(URL,timeout=90).read()
    splits=data(payload)
    labels={}
    for k,x in splits.items():
        labels[k]=reference(x,53)
        assert np.array_equal(labels[k],reference(x,224)),k
    rows,examples,counts=[],{},{}
    def score(method,bits,ns,values):
        y=labels[ns]
        rows.append({'method':method,'bits':bits,'split':ns,**measures(values,y)})
        fail=np.flatnonzero(values!=y)
        if len(fail):
            i=fail[0]
            examples[f'{method}_{bits}_{ns}']=dict(x=splits[ns][i].hex(),output=values[i].hex(),mpfr=y[i].hex())
    for ns,x in splits.items():
        score('numpy_tanh',0,ns,np.tanh(x))
    for bits in (20,36,52,64,76,88,104,FINAL_BITS):
        for double in (False,True):
            method='two_word' if double else 'one_word'
            g=tanh_graph(bits,double)
            counts[f'{method}_{bits}']={**g.summary(),**static_input_check(g)}
            for ns,x in splits.items():
                if ns in ('fresh_outer','core_inner','core_outer','anchors') and bits not in (88,FINAL_BITS):
                    continue
                score(method,bits,ns,g.run(x,g.weights(),batch=2048))
            print(bits,method,'done',flush=True)
            if bits==FINAL_BITS and double:
                counts['summed_two_word']=export(g)
    for triple,bits in [(False,84)]+[(True,m) for m in (60,72,76,78,79,80,82,84)]:
        method='guard_three_word' if triple else 'guard_two_word'
        g=tanh_graph(bits,True,guard_round=True,triple=triple)
        counts[f'{method}_{bits}']={**g.summary(),**static_input_check(g)}
        for ns,x in splits.items():
            score(method,bits,ns,g.run(x,g.weights(),batch=2048))
        print(method,bits,'done',flush=True)
        if triple and bits==84:
            counts['deployed']=export(g)
    with (ROOT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (ROOT/'counts.json').write_text(json.dumps(counts,indent=1)+'\n')
    (ROOT/'examples.json').write_text(json.dumps(examples,indent=1)+'\n')
    info=dict(core_url=URL,core_sha256=DIGEST,numpy=np.__version__,gmpy2=mp.version(),
              mpfr=mp.mpfr_version(),bits=84,division_digits=82,words=3,rounding='significand_and_guard',format='binary64',
              domain=[-10,10],checkpoint_sha256=hashlib.sha256((ROOT/'program.npz').read_bytes()).hexdigest())
    (ROOT/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    main()
