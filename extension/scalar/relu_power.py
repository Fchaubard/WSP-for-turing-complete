"""Positive self-power via three fixed passes of binary decisions and affine updates."""
from compile import Graph,Precision,static_input_check
from run import mp,np,labeled,samples,stress_samples,rows_for,save
from pathlib import Path
import json

def power_graph(bits,double):
    g=Graph()
    ops=Precision(g,double)
    z,o=g.literal(0),g.literal(1)
    def copy_if(keep,v,bound):
        block=(o-keep)*bound
        return (v-block).relu()-(z-v-block).relu()
    def affine_style(keep,v,bound):
        if double:return tuple(copy_if(keep,w,bound) for w in v)
        return copy_if(keep,v,bound)
    def negate(v):
        if double:return tuple(z-w for w in v)
        return z-v
    def scale_exact(v,factor):
        if double:return tuple(w*factor for w in v)
        return v*factor
    def high(v):
        return v[0] if double else v
    def compare_ge(v,pivot,gain):
        return (o-((pivot-v)*gain).relu()).relu()

    # Normalization reads x in [0,128] and writes a mantissa in [1,2] and a
    # signed exponent. Interested compiled gates do not multiply dynamic inputs.
    exps=list(range(-64,7))
    flags=[]
    for k in exps:
        flags.append(compare_ge(g.x,g.param(2.**k),g.param(2.**(53-k))))
    masks=[a-b for a,b in zip(flags,flags[1:])]+[flags[-1]]
    mant=o-flags[0]
    shift=z
    for k,flag in zip(exps,masks):
        # Mask x in its original bounded range before the exact dyadic rescaling.
        mant=mant+copy_if(flag,g.x,128)*g.param(2.**(-k))
        shift=shift+flag*g.param(k)
    with mp.context(precision=220):
        # If u=m-1 >= d, replace m by m/(1+d) and add log(1+d) to
        # the log accumulator. New u=(u-d)/(1+d). Each pass halves its bound.
        u=ops.from_wire(mant-o)
        logx=ops.multiply(ops.from_wire(shift),ops.const(mp.log(mp.mpfr(2))))
        for j in range(1,bits+1):
            delta=mp.mpfr(2)**(-j)
            pivot=g.param(float(delta))
            take=compare_ge(high(u),pivot,g.param(2.**(j+53)))
            diff=ops.add(u,ops.from_wire(z-pivot))
            updated=ops.multiply(diff,ops.const(1/(1+delta)))
            u=ops.add(affine_style(take,updated,4),affine_style(o-take,u,4))
            logarithm=ops.const(mp.log1p(delta))
            # The factor take is input-dependent, while logarithm contains only
            # stored parameters. Thus this is an ordinary scalar affine edge.
            logx=ops.add(logx,scale_exact(logarithm,take))
        # log(1+u)=u+O(u^2). This final correction doubles the asymptotic
        # accuracy of the log residual after the fixed greedy passes.
        logx=ops.add(logx,u)

        # Form y=1+x log(x) from input bits and conditional copies of log(x).
        # Starting at 2^7 supports [0,128]. All scalings are dyadic.
        remainder=g.x
        y=ops.from_wire(o)
        for k in range(7,-bits-1,-1):
            factor=g.param(2.**k)
            take=compare_ge(remainder,factor,g.param(2.**(53-k)))
            remainder=remainder-take*factor
            summand=affine_style(take,logx,64)
            y=ops.add(y,scale_exact(summand,factor))

        # Decompose positive y (min 1-e^-1). Start at exp(-1), and for each
        # selected bit multiply by a stored exp(2^k) constant. Mask the factor
        # input first so unused branches cannot overflow.
        value=ops.const(mp.exp(-1))
        for k in range(9,-bits-1,-1):
            d=mp.mpfr(2)**k
            pivot=g.param(float(d))
            take=compare_ge(high(y),pivot,g.param(2.**(53-k)))
            y=ops.add(y,ops.from_wire(z-take*pivot))
            off=affine_style(o-take,value,2.**1020)
            prev=affine_style(take,value,2.**1020)
            value=ops.add(off,ops.multiply(prev,ops.const(mp.exp(d))))
        result=ops.final(value)
        # |x log x| for x<2^-64 is too small to move 1 under correct rounding.
        copy_if(flags[0],result,2.**1020)+(o-flags[0])
    static_input_check(g)
    return g

def run():
    base=labeled(samples("selfpower"),"selfpower")
    stress=labeled(stress_samples("selfpower"),"selfpower")
    rows=[]
    counts={}
    for bits in (16,28,40,52,60,72,84):
        for double in (False,True):
            g=power_graph(bits,double)
            method="relu_selfpower_two_word" if double else "relu_selfpower_single_word"
            counts[f"{method}_{bits}"]=static_input_check(g)
            rows_for(rows,"selfpower",method,g,bits,0,base)
            if bits in (60,84):rows_for(rows,"selfpower",method,g,bits,0,stress)
        save(rows,"relu_power.csv")
        print(bits,flush=True)
    (Path(__file__).parent/"relu_power_counts.json").write_text(json.dumps(counts,indent=2)+"\n")
if __name__=="__main__":
    run()
