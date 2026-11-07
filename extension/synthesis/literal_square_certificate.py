"""Stress the frozen affine/ReLU writer; exact rational rounding is the oracle."""
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
import numpy as np
from experiment import Compiler
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'trials'/'3354251e2.reply.txt'

def insert(values,m):
    if not 2**52<=m<2**53:return
    for exponent in (1,2,3):
        x=math.ldexp(m,exponent-52)
        if 3<=x<=10:values.update((-x,x))

def shell_stress():
    sets={}
    # Points on both sides of every active branch boundary, plus the endpoints.
    edges=set()
    for v in (3.,4.,8.,10.):
        for direction in (-math.inf,math.inf):
            x=v
            for _ in range(9):
                if 3<=x<=10:edges.update((x,-x))
                x=math.nextafter(x,direction)
    sets['normalization_boundaries']=edges
    # Probe base-B limb / carry changes and their three adjacent significands.
    B=2**27
    limbs=set()
    for q in [2**25+i*(2**19) for i in range(65)]:
        for low in (0,1,B//2,B-2,B-1):
            for offset in (-1,0,1):
                insert(limbs,q*B+low+offset)
    sets['limb_boundaries']=limbs
    # Products at exact rounding ties (when k=53 and v2(m)=26).
    ties=set()
    for q in range(2**26+1,2**27,65536):
        m=q*2**26
        if (m*m).bit_length()==106 and (m*m)%(2**53)==2**52:insert(ties,m)
    sets['round_to_even_ties']=ties
    # Near misses to either side of a rounding midpoint via bitwise modular square roots.
    near=set()
    for k in (52,53):
        for d in range(-255,256):
            if d==0:continue
            goal=2**(k-1)+d
            roots=[goal%2]
            for bit in range(1,k):
                if not roots:break
                mask=2**(bit+1)-1
                roots=[z for r in roots for z in (r,r+2**bit) if (z*z-goal)&mask==0]
            for r in roots:
                for m in (r,r+2**k):
                    if (m*m).bit_length()==53+k:insert(near,m)
    sets['near_midpoints']=near
    return sets

if __name__=='__main__':
    raw=RAW.read_bytes()
    c=Compiler('affine_relu')
    c.block(json.loads(raw))
    results={}
    for label,items in shell_stress().items():
        x=np.array(sorted(items))
        y=np.array([float(Fraction.from_float(float(t))**2) for t in x])
        pred=c.run(x)
        hits=int(np.sum(pred==y))
        assert hits==len(x)
        results[label]=dict(n=len(x),exact=hits)
    out=dict(trial_id='3354251e2',raw_sha256=hashlib.sha256(raw).hexdigest(),ops=len(c.ops),
             oracle='float(Fraction.from_float(x)**2)',signed_shell=[3,10],suites=results)
    print(json.dumps(out,indent=2))
    (ROOT/'literal_square_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
