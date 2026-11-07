"""Run scalar circuit sweeps and high-precision correctness checks."""
import csv
import platform
from pathlib import Path
import json
import sys
import numpy as np
import gmpy2 as mp
from compile import sin_graph,selfpower_graph

ROOT=Path(__file__).resolve().parent
SEED=911285
def reference(x,task,precision=53):
    with mp.context(precision=precision):
        if task=="sin":
            return np.fromiter((float(mp.sin(mp.mpfr(t))) for t in x),float)
        return np.fromiter((float(mp.mpfr(t)**mp.mpfr(t)) for t in x),float)

def ulps(pred,truth):
    def key(v):
        u=np.ascontiguousarray(v).view(np.uint64)
        sign=np.uint64(1<<63)
        return np.where(u&sign,~u+np.uint64(1),u|sign)
    a,b=key(pred),key(truth)
    return np.maximum(a,b)-np.minimum(a,b)

def samples(task):
    # Reproduce the scalar audit's exact train / evaluation draws.
    rng=np.random.default_rng(20261105)
    pos=task=="selfpower"
    data={"train":rng.uniform(0 if pos else -3,3,1024),
          "inner":rng.uniform(0 if pos else -3,3,4096)}
    if not pos: data["left"]=rng.uniform(-10,-3,4096)
    data["right"]=rng.uniform(3,10,4096)
    return data

def stress_samples(task):
    rng=np.random.default_rng(SEED)
    x=rng.uniform(3,10,32768)
    if task=="sin":
        x=x*rng.choice([-1.,1.],len(x))
        roots=[]
        with mp.context(precision=180):
            for k in (-3,-2,-1,1,2,3):
                center=float(mp.const_pi()*k)
                roots.append(center)
                for side in (np.inf,-np.inf):
                    v=center
                    for _ in range(8):
                        v=np.nextafter(v,side)
                        roots.append(v)
        return dict(fresh=x,near_roots=np.array(roots),beyond=rng.uniform(10,16,4096))
    boundaries=[]
    for k in range(-64,7):
        v=2.**k
        boundaries.extend((np.nextafter(v,-np.inf),v,np.nextafter(v,np.inf)))
    return dict(fresh=x,dyadic_edges=np.array(boundaries),
                broad=np.exp2(rng.uniform(-65,7,8192)),
                beyond=rng.uniform(10,128,4096))

def rows_for(rows,task,method,model,order,stage,dataset,params=None):
    p=model.weights() if params is None else params
    for split,(x,y,lib) in dataset.items():
        pred=model.run(x,p)
        error=ulps(pred,y)
        rows.append(dict(task=task,method=method,order=order,stage=stage,split=split,
                         **model.summary(),n=len(y),correct=int(np.sum(error==0)),
                         within1=int(np.sum(error<=1)),worst_ulp=int(error.max()),
                         max_abs=float(np.max(np.abs(pred-y))),mae=float(np.mean(np.abs(pred-y))),
                         numpy_equal=int(np.sum(pred==lib)),numpy_ref_equal=int(np.sum(lib==y))))

def labeled(inputs,task):
    out={}
    for name,x in inputs.items():
        y=reference(x,task)
        lib=np.sin(x) if task=="sin" else np.power(x,x)
        out[name]=(x,y,lib)
    return out

def save(rows,key):
    with (ROOT/key).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)

def main():
    rows=[]
    for task in ("sin","selfpower"):
        datasets=labeled(samples(task),task)
        stress=labeled(stress_samples(task),task)
        for x,y,_ in [*datasets.values(),*stress.values()]:
            assert np.array_equal(y,reference(x,task,160))
        if task=="sin":
            cases=[(d,0) for d in (9,17,25,33,41,49,57,65,73,81,89)]
        else:
            cases=[(8,6),(12,8),(16,10),(24,12),(32,16),(40,20),(48,24),(56,28)]
        for j,(a,b) in enumerate(cases):
            for double in (False,True):
                method="two_word" if double else "single_word"
                g=sin_graph(a,double) if task=="sin" else selfpower_graph(a,b,double)
                rows_for(rows,task,method,g,a,b,datasets)
                if (task=="sin" and a in (49,65,81,89)) or (task=="selfpower" and a in (32,40,48,56)):
                    rows_for(rows,task,method,g,a,b,stress)
            save(rows,"sweep.csv")
            print(task,a,b,flush=True)
        if task=="sin":
            central_fits(rows,datasets)
            save(rows,"sweep.csv")

def central_fits(rows,datasets):
    x,y,_=datasets["train"]
    for degree in (9,17,25,33,49,65):
        # Fit the odd polynomial family using central coordinates and central
        # labels only. In a well-conditioned Chebyshev feature basis the loss
        # is quadratic, so full-batch Gauss-Newton solves it in one iteration.
        scaled=x/3
        J=np.polynomial.chebyshev.chebvander(scaled,degree)[:,1::2]
        rng=np.random.default_rng(11)
        init=rng.uniform(-np.sqrt(6/(J.shape[1]+1)),np.sqrt(6/(J.shape[1]+1)),J.shape[1])
        coeff=init-np.linalg.lstsq(J,J@init-y,rcond=None)[0]
        # Second step removes cancellation in subtracting a unit-sized update.
        coeff-=np.linalg.lstsq(J,J@coeff-y,rcond=None)[0]
        cheb=np.zeros(degree+1)
        cheb[1::2]=coeff
        # Convert centrally fitted polynomial to exactly the evaluation graph used
        # by the formula writer. All fitted central coefficients are binary64.
        mono=np.polynomial.chebyshev.cheb2poly(cheb)[1::2]/(3.**np.arange(1,degree+1,2))
        g=sin_graph(degree,True,coefs=mono)
        rows_for(rows,"sin","central_gauss_newton",g,degree,0,datasets)

if __name__=="__main__":
    main()
