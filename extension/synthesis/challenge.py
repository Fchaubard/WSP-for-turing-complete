"""Blind challenge panel for the direct first responses; stream graph calls in small batches."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from experiment import Compiler,variants,HERE
sys.path.insert(0,str(HERE.parents[1]/'scalar'))
from compensated import rounding_challenges
sys.path.insert(0,str(HERE.parent/'scalar'))
from stress import special
from run import reference,ulps
from hard_sin import URL,DIGEST

def panels(core):
    rng=np.random.default_rng(292920731)
    fr=rng.uniform(3,10,32768)*rng.choice([-1,1],32768)
    mant=rng.uniform(.5,1,16384)
    sq_exp=rng.integers(-1073,5,16384); small=np.ldexp(mant,sq_exp);small=small[small<=10]
    # Include the normal to subnormal output transition and smallest rounded positive square.
    sq_edge=[]
    for k in range(-1074,7):
        v=np.sqrt(np.ldexp(1.,k))
        if v<=10:sq_edge.extend([np.nextafter(v,0),v,np.nextafter(v,np.inf)])
    sq={'midpoints':np.asarray(rounding_challenges()),'exponents':small,'square_binade_edges':np.array(sq_edge),'fresh_shell':fr}
    vals=set()
    if hashlib.sha256(core).hexdigest()!=DIGEST:raise ValueError('source digest')
    for l in core.splitlines():
        try:v=float.fromhex(l.decode().strip())
        except ValueError:continue
        if 0<abs(v)<=10:vals.add(v)
    vs=np.array(sorted(vals));sp=special('selfpower')
    sine={'core_central':vs[np.abs(vs)<=3],'core_outer':vs[np.abs(vs)>3],'fresh_shell':fr}
    spow={'near_midpoints':sp['near_midpoints'],'exponents':sp['exponent_wide'],'fresh_shell':np.abs(fr)}
    return {'square':sq,'sin':sine,'selfpower':spow}

def main(core):
    ds=panels(core); cached={}
    for task,ps in ds.items():
        for k,x in ps.items():
            cached[task,k]=(x,x*x if task=='square' else reference(x,task))
    rows=[]
    for q in variants():
        r=HERE/'trials'/(q['id']+'.reply.txt')
        if not r.exists():continue
        try:
            raw=r.read_text().strip()
            if raw.startswith(chr(96)*3):raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
            c=Compiler(q['arm']);c.block(json.loads(raw))
        except (ValueError,TypeError,KeyError,OverflowError,IndexError):continue
        for k in ds[q['task']]:
            x,y=cached[q['task'],k]
            yh=np.concatenate([c.run(x[i:i+384]) for i in range(0,len(x),384)])
            err=ulps(yh,y);miss=np.flatnonzero(yh!=y)
            rows.append(dict(**q,split=k,n=len(y),exact=int((yh==y).sum()),within1=int((err<=1).sum()),max_ulp=int(err.max()),
               first_failure_x=x[miss[0]].hex() if len(miss) else '',first_failure_output=yh[miss[0]].hex() if len(miss) else '',reference=y[miss[0]].hex() if len(miss) else ''))
    with (HERE/'challenge.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('Scored',len(rows),'challenge splits')
if __name__=='__main__':
    from urllib.request import urlopen
    ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path);r=ap.parse_args()
    main(r.core.read_bytes() if r.core else urlopen(URL,timeout=90).read())
