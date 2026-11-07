"""Input-distribution, small-output and high-precision midpoint challenges."""
from pathlib import Path
import json
from run import np,mp,reference,rows_for,labeled,save
from compile import relu_sin_graph
from relu_power import power_graph

def neighbors(c,n=3):
    out=[c]
    for d in (np.inf,-np.inf):
        v=c
        for _ in range(n):
            v=np.nextafter(v,d)
            out.append(v)
    return out

def bisect(low,high,f):
    # f is increasing and brackets zero.
    for _ in range(220):
        mid=(low+high)/2
        if f(mid)<0: low=mid
        else: high=mid
    return float((low+high)/2)

def special(task):
    rng=np.random.default_rng(45217)
    max_exp=4 if task=="sin" else 7
    with np.errstate(under="ignore"):
        random=np.ldexp(rng.uniform(.5,1,8192),rng.integers(-1073,max_exp+1,8192))
    anchor=np.unique(np.r_[0.,np.nextafter(0.,1.),np.finfo(float).tiny,np.finfo(float).tiny/2])
    roots=set()
    with mp.context(precision=220):
        two=mp.mpfr(2)
        if task=="sin":
            # On each binade, output tracks x at small arguments. A midpoint
            # between x-j*u and x-(j+1)*u is a fixed offset from x.
            for e in range(-27,-16):
                lo,hi=two**e,two**(e+1)
                ulp=two**(e-52)
                minimum=int(mp.ceil((lo-mp.sin(lo))/ulp-mp.mpfr(".5")))
                maximum=int(mp.floor((hi-mp.sin(hi))/ulp-mp.mpfr(".5")))
                if maximum<minimum:continue
                js=np.unique(np.linspace(max(0,minimum),maximum,min(64,maximum-minimum+1)).astype(int))
                for j in js:
                    off=(int(j)+mp.mpfr(".5"))*ulp
                    c=bisect(lo,hi,lambda x:x-mp.sin(x)-off)
                    roots.update(neighbors(c))
        else:
            # First 128 rounding discontinuities below 1 of x^x on (0,1/e).
            # Solve x log(x)=log(1-(j+1/2)*2^-53) at high precision.
            for j in range(128):
                cut=mp.log1p(-(j+mp.mpfr(".5"))*two**-53)
                c=bisect(two**-100,two**-36,lambda x:cut-x*mp.log(x))
                roots.update(neighbors(c))
    bounds=np.array(sorted(roots))
    if task=="sin":
        bounds=np.r_[-bounds,bounds]
        random*=rng.choice([-1.,1.],len(random))
        anchor=np.r_[-anchor,anchor]
    return dict(exponent_wide=random,tiny=anchor,near_midpoints=bounds)

def run():
    rows=[]
    errors=[]
    for task,model in [("sin",relu_sin_graph(-78)),("selfpower",power_graph(84,True))]:
        data=labeled(special(task),task)
        for split,(x,y,lib) in data.items():
            assert np.array_equal(y,reference(x,task,220))
            pred=model.run(x,model.weights())
            for idx in np.flatnonzero(pred!=y)[:4]:
                errors.append(dict(task=task,split=split,x=x[idx].hex(),
                                   output=pred[idx].hex(),mpfr=y[idx].hex()))
        method="relu_rotation_two_word" if task=="sin" else "relu_selfpower_two_word"
        rows_for(rows,task,method,model,83 if task=="sin" else 84,-78 if task=="sin" else 0,data)
    save(rows,"stress.csv")
    (Path(__file__).parent/"counterexamples.json").write_text(json.dumps(errors,indent=2)+"\n")
if __name__=="__main__":
    run()
