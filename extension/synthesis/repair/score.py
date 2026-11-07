"""Replay first and revised circuits after completion. Outer predictions are never sent to the model."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from feedback import HERE,SOURCE,selected,compile_reply,reference,errors
from experiment import samples

def oracle(task):
    splits=samples(task)
    fresh=np.random.default_rng(640317)
    shell=fresh.uniform(3,10,32768)
    if task!="selfpower":shell*=fresh.choice([-1,1],size=len(shell))
    splits["fresh_outer"]=shell
    splits["fresh_inner"]=np.random.default_rng(385591).uniform(0 if task=="selfpower" else -3,3,4096)
    return {name:(x,reference(x,task)) for name,x in splits.items()}

def metrics(yhat,y):
    finite=np.isfinite(yhat)
    delta=errors(yhat,y)
    return dict(n=y.size,correct=int(np.sum(yhat==y)),within1=int(np.sum((delta<=1)&finite)),
                finite=int(finite.sum()),worst_ulp=int(np.max(delta)) if finite.all() else None,
                mae=float(np.mean(np.abs(yhat-y))) if finite.all() else None)

def score():
    oracles={}
    rows=[]
    paired=[]
    metadata=[]
    for q in selected():
        cid=q["id"]
        if not (HERE/"trials"/(cid+".meta.json")).exists() and not (HERE/"trials"/(cid+".failure.json")).exists():
            continue
        if q["task"] not in oracles:oracles[q["task"]]=oracle(q["task"])
        exact_flags={}
        for stage,folder in (("original",SOURCE),("repair",HERE)):
            base=folder/"trials"/cid
            info=dict(id=cid,task=q["task"],arm=q["arm"],stage=stage)
            try:
                reply=base.with_suffix(".reply.txt").read_text()
                api=json.loads(base.with_suffix(".meta.json").read_text())
                info.update(seconds=api["elapsed_seconds"],events=api["completed_items"],
                            sha256=hashlib.sha256(reply.encode()).hexdigest())
                c=compile_reply(reply,q["arm"])
                info.update(status="valid",ops=len(c.ops))
                for split,(x,y) in oracles[q["task"]].items():
                    output=c.run(x)
                    exact_flags[stage,split]=output==y
                    r=dict(id=cid,task=q["task"],arm=q["arm"],stage=stage,split=split)
                    rows.append(dict(**r,**metrics(output,y)))
            except Exception as exc:
                if not base.with_suffix(".reply.txt").exists():
                    info.update(status="request_failed")
                else:info.update(status="invalid",reason=str(exc)[:240])
            metadata.append(info)
        for split in oracles[q["task"]]:
            if ("original",split) in exact_flags and ("repair",split) in exact_flags:
                before=exact_flags["original",split]
                after=exact_flags["repair",split]
                paired.append(dict(id=cid,task=q["task"],arm=q["arm"],split=split,
                                   n=len(before),gained=int(np.sum(~before & after)),
                                   lost=int(np.sum(before & ~after)),
                                   both=int(np.sum(before & after)),neither=int(np.sum(~before & ~after))))
    if paired:
        with (HERE/"paired.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=paired[0])
            writer.writeheader()
            writer.writerows(paired)
    (HERE/"status.json").write_text(json.dumps(metadata,indent=2)+"\n")
    if rows:
        with (HERE/"results.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print("Scored complete pairs",len(metadata)//2)

if __name__=="__main__":score()
