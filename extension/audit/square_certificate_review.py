"""Independent underflow-midpoint and exponent-stratified audit of the frozen repair."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/'synthesis'))
from experiment import Compiler


def midpoint_inputs():
    """Bracket exact *output* subnormal midpoints with neighboring input floats."""
    rng=np.random.default_rng(517003)
    sites=set()
    for exponent in range(-538,-511):
        # For x=m*2^(e-52), the square in units of 2^-1074 is m²/2^s.
        s=-2*exponent-970
        trace_min=2**104
        trace_max=(2**53-1)**2
        lo=max(0,((trace_min>>(s-1))-1)//2)
        hi=((trace_max>>(s-1))-1)//2
        if hi<lo:
            continue
        ks={lo,hi}
        for j in range(53):
            for delta in (-1,0,1):
                k=(1<<j)+delta
                if lo<=k<=hi:
                    ks.add(k)
        ks.update(map(int,rng.integers(lo,hi+1,size=160)))
        for k in ks:
            # Neighboring m values straddle the true midpoint root exactly.
            pivot=math.isqrt((2*k+1)<<(s-1))
            for m in range(pivot-2,pivot+4):
                if 2**52<=m<2**53:
                    x=math.ldexp(float(m),exponent-52)
                    sites.update((x,-x))
    return np.array(sorted(sites))


def exponent_inputs():
    rng=np.random.default_rng(41244)
    sites={0.,10.,-10.}
    for e in range(-1074,4):
        x=math.ldexp(1.,e)
        for z in [np.nextafter(x,0.),x,np.nextafter(x,np.inf)]:
            if z<=10:
                sites.update((z,-z))
        for m in rng.integers(2**52,2**53,size=16):
            z=math.ldexp(float(m),e-52)
            if z<=10:
                sites.update((z,-z))
    return np.array(sorted(sites))


def run():
    path=HERE.parent/'square_full'/'program.json'
    comp=Compiler('affine_relu')
    comp.block(json.loads(path.read_text()))
    out={}
    for key,x in [('subnormal_midpoints',midpoint_inputs()),('exponent_strata',exponent_inputs())]:
        y=np.array([float(Fraction.from_float(float(t))**2) for t in x])
        z=np.concatenate([comp.run(x[i:i+384]) for i in range(0,len(x),384)])
        assert np.isfinite(z).all() and np.array_equal(z,y),key
        assert np.array_equal(z.view(np.uint64),y.view(np.uint64)),key
        out[key]=dict(inputs=len(x),exact=int(np.sum(z==y)),min_abs=float(np.min(np.abs(x))),
                      max_abs=float(np.max(np.abs(x))))
    info={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'ops':len(comp.ops),'oracle':'float(Fraction.from_float(x)**2)','suites':out}
    print(json.dumps(info,indent=2))
    (HERE/'square_certificate_review.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    run()
