"""Freeze a paired central-only diagnostic and one repair prompt for each designated trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import gmpy2 as mp

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE))
from experiment import Compiler,variants

def selected():
    return [q for q in variants() if q["seed"]==9213]

def compile_reply(raw,arm):
    text=raw.strip()
    if text.startswith(chr(96)*3):
        text=text.split("\n",1)[1].rsplit(chr(96)*3,1)[0]
    c=Compiler(arm)
    c.block(json.loads(text))
    if "y" not in c.env:raise KeyError("missing output wire y")
    return c

def reference(x,task):
    with mp.context(precision=53):
        out=[]
        for t in x:
            z=mp.mpfr(float(t))
            out.append(float(z*z if task=="square" else mp.sin(z) if task=="sin" else z**z))
        return np.array(out)

def ordered(v):
    u=np.asarray(v).view(np.uint64)
    s=np.uint64(1<<63)
    return np.where(u&s,~u+np.uint64(1),u|s)

def errors(pred,target):
    a,b=ordered(pred),ordered(target)
    return np.maximum(a,b)-np.minimum(a,b)

def make(q):
    cid=q["id"]
    original=(SOURCE/"trials"/(cid+".reply.txt")).read_text()
    rule=(SOURCE/"trials"/(cid+".prompt.txt")).read_text().strip()
    # Match the original 4,096 central draws without constructing shell examples.
    x=np.random.default_rng(131490).uniform(0 if q["task"]=="selfpower" else -3,3,4096)
    y=reference(x,q["task"])
    try:
        c=compile_reply(original,q["arm"])
        pred=c.run(x)
        finite=np.isfinite(pred)
        delta=errors(pred,y)
        # Nonfinite first, then worst ULP error, earlier draw breaks ties.
        idx=sorted(range(len(x)),key=lambda i:(bool(finite[i]),-int(delta[i]),i))[:3]
        diagnostic=dict(status="valid",dynamic_ops=len(c.ops),n=len(x),
                        exact=int(np.sum(pred==y)),within_one_ulp=int(np.sum((delta<=1)&finite)),
                        finite=int(finite.sum()))
    except Exception as e:
        pred=None
        idx=list(range(3))
        diagnostic=dict(status="invalid",compiler_error=str(e)[:240])
    examples=[]
    for i in idx:
        entry=dict(x=float(x[i]).hex(),correctly_rounded_reference=float(y[i]).hex())
        if pred is not None:
            entry.update(your_output=float(pred[i]).hex(),ulp_error=int(delta[i]) if finite[i] else "nonfinite")
        examples.append(entry)
    diagnostic["three_central_examples"]=examples
    prompt=(
        "You are making one diagnostic-guided revision of your earlier circuit. The original task specification follows.\n"
        +rule+"\nEnd original specification.\nYour previous reply follows verbatim:\n"+original+
        "\nEnd previous reply.\nA scorer evaluated only the central input interval "+
        ("[0,3]" if q["task"]=="selfpower" else "[-3,3]")+
        ". Here is the entire feedback. Hexadecimal strings represent binary64 numbers. For valid outputs the three examples have the largest ULP errors (earliest input on ties). For an invalid program they are three labeled central inputs without predictions.\n"
        +json.dumps(diagnostic,indent=2)+
        "\nReturn ONE complete revised program under the original grammar and 4,096-operation limit. If your prior solution is already satisfactory, you may return it unchanged. No tools, data queries, or outside feedback. Return only the JSON instructions.\n")
    base=HERE/"trials"/cid
    prompt_path=base.with_suffix(".prompt.txt")
    if prompt_path.exists() and prompt_path.read_text()!=prompt:
        raise ValueError("Frozen repair prompt differs from reconstruction")
    prompt_path.write_text(prompt)
    base.with_suffix(".feedback.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    return cid,diagnostic["status"]

def manifest():
    (HERE/"trials").mkdir(exist_ok=True)
    (HERE/"manifest.json").write_text(json.dumps(selected(),indent=2)+"\n")

if __name__=="__main__":
    manifest()
    for q in selected():
        if (SOURCE/"trials"/(q["id"]+".reply.txt")).exists():
            print(*make(q))
