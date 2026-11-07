"""Score all first proposals only after the global freeze, with raw and cost records."""
import csv
import hashlib
import json
import math
import secrets
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from statistics import median
import time

import numpy as np
from network import Network

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'extension/writer'))
from design import TASKS, SERIES, SMOOTH, observed, inputs, reference
sys.path.insert(0,str(ROOT))
from error_metrics import rmse


def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(p):
    raw=p.read_text().strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(raw)
def write(name,rows):
    with (HERE/name).open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def data(task,seed,entropy):
    sets={};sets['train']=observed(task,seed)
    old=inputs(task)
    center=np.arange(17,33) if task in SERIES else old['inner']
    sets['dev']=(center,reference(task,center));sets['outer']=(old['outer'],reference(task,old['outer']))
    if task in SERIES:return sets
    rng=np.random.default_rng(np.random.SeedSequence([entropy,TASKS.index(task)]))
    n=2048;dim={'add':2,'subtract':2,'sort':4,'sum':6}.get(task)
    shape=(n,dim) if dim else n
    center=rng.uniform(0 if task=='selfpower' else -3,3,shape)
    outer=rng.uniform(3,10,shape)
    if task!='selfpower':outer*=rng.choice([-1,1],size=shape)
    sets['confirmation_dev']=(center,reference(task,center))
    sets['confirmation_outer']=(outer,reference(task,outer))
    if task=='sum':sets['length32']=(old['length'],reference(task,old['length']))
    return sets


def evaluate_case(case,entropy):
    task,seed,uid=case['task'],case['seed'],case['id'];base=HERE/'trials'/uid
    header=dict(task=task,seed=seed,id=uid,mode=case['mode'],method='architecture_wsp_examples')
    score=[];cost=[]
    try:
        reply=parse(base.with_suffix('.reply.txt'))
        model=Network(reply,case['mode'])
        info=dict(**header,status='valid',family=reply.get('family',''),nodes=model.nodes,coefficients=model.coefficients,
                  coefficient_bytes=8*model.coefficients,serialized_model_bytes=model.serialized_bytes,
                  inferred_rule=reply.get('inferred_rule',''),construction_summary=reply.get('construction_summary',''),
                  certificate=reply.get('certificate',''),error='')
        for split,(x,y) in data(task,seed,entropy).items():
            try:
                prediction,costs=model.predict(x,y.shape[1:])
                finite=bool(np.isfinite(prediction).all())
                if not finite:raise ValueError('Nonfinite prediction')
                equal=(prediction==y).reshape(len(x),-1).all(axis=1)
                norm=(prediction-y)/np.maximum(1,np.abs(y))
                scaled=rmse(norm,np.zeros_like(norm))
                score.append(dict(**header,split=split,status='valid',n_examples=len(x),n_scalars=y.size,
                                  exact_count=int(equal.sum()),exact_fraction=float(equal.mean()),raw_rmse=rmse(prediction,y),
                                  scaled_rmse=scaled,error=''))
                cost.append(dict(**header,split=split,n_examples=len(x),coefficients=model.coefficients,
                                 coefficient_bytes=model.coefficients*8,serialized_model_bytes=model.serialized_bytes,
                                 nodes=model.nodes,
                                 flops_min=min(c.arithmetic for c in costs),flops_median=median(c.arithmetic for c in costs),flops_max=max(c.arithmetic for c in costs),
                                 comparisons_min=min(c.comparisons for c in costs),comparisons_median=median(c.comparisons for c in costs),comparisons_max=max(c.comparisons for c in costs),
                                 transcendental_min=min(c.transcendental for c in costs),transcendental_median=median(c.transcendental for c in costs),transcendental_max=max(c.transcendental for c in costs),
                                 workspace_bytes_min=8*min(c.peak_scalars for c in costs),workspace_bytes_median=8*median(c.peak_scalars for c in costs),workspace_bytes_max=8*max(c.peak_scalars for c in costs)))
            except Exception as e:
                info['status']='evaluation_failure';info['error']+=split+': '+str(e)+'; '
                score.append(dict(**header,split=split,status='failure',n_examples=len(x),n_scalars=y.size,exact_count=0,
                                  exact_fraction=0.,raw_rmse=float('inf'),scaled_rmse=float('inf'),error=str(e)))
    except Exception as e:
        info=dict(**header,status='invalid',family='',nodes='',coefficients='',coefficient_bytes='',serialized_model_bytes='',inferred_rule='',construction_summary='',certificate='',error=str(e))
        for split,(x,y) in data(task,seed,entropy).items():
            score.append(dict(**header,split=split,status='failure',n_examples=len(x),n_scalars=y.size,exact_count=0,
                              exact_fraction=0.,raw_rmse=float('inf'),scaled_rmse=float('inf'),error=str(e)))
    return info,score,cost


def main():
    freeze=load(HERE/'FROZEN.json')
    assert digest(HERE/'PROTOCOL.md')==freeze['protocol'] and digest(HERE/'PRECOMMIT.json')==freeze['precommit']
    for n,h in freeze['trials'].items():assert digest(HERE/'trials'/n)==h
    path=HERE/'confirmation_seed.json'
    if not path.exists():path.write_text(json.dumps(dict(entropy=secrets.randbits(128),created_after_freeze=time.time()))+'\n')
    entropy=load(path)['entropy'];cases=load(HERE/'manifest.json')
    records=[];scores=[];costs=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        pending=[pool.submit(evaluate_case,c,entropy) for c in cases]
        for i,f in enumerate(as_completed(pending),1):
            a,b,c=f.result();records.append(a);scores+=b;costs+=c
            print(i,a['task'],a['seed'],a['status'],flush=True)
    records.sort(key=lambda r:(TASKS.index(r['task']),r['seed']))
    scores.sort(key=lambda r:(TASKS.index(r['task']),r['seed'],r['split']))
    costs.sort(key=lambda r:(TASKS.index(r['task']),r['seed'],r['split']))
    write('models.csv',records);write('scores.csv',scores);write('costs.csv',costs)
    for name in freeze['trials']:assert digest(HERE/'trials'/name)==freeze['trials'][name]
    stats=dict(freeze_sha256=digest(HERE/'FROZEN.json'),models=len(records),status=dict(Counter(r['status'] for r in records)),
               outer_exact_sets=sum(r['exact_count']==r['n_examples'] for r in scores if r['split']=='outer'),
               result_sha256={n:digest(HERE/n) for n in ['models.csv','scores.csv','costs.csv']})
    (HERE/'audit.json').write_text(json.dumps(stats,indent=1)+'\n');print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
