"""Post-freeze numerical evaluation. No model calls or training are possible here."""
import csv
import hashlib
import json
import secrets
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COLLECT=ROOT/'focus/iterative'
sys.path[:0]=[str(ROOT/'extension/writer'),str(ROOT/'focus/arithmetic'),str(ROOT)]
from design import SERIES, SMOOTH, TASKS, inputs, initial, observed, predict, reference
from error_metrics import rmse
from fixtures import forward as arithmetic_forward
from compare import parse

PROTOCOL='75c8a10bc407ddd2808e3a7101a82fcda21fe45b42c8d28ecd48cad31cc88638'
CONTROLS=('examples','formula','adam_bfgs','ls_xavier','gd','gd_lbfgs','adam_lbfgs','lbfgs_xavier')
SPLITS=('train','dev','outer')
SCHEMA=('protocol','task','size','params','seed','method','split','split_description',
        'n_examples','n_scalars','exact_count','exact_fraction','raw_rmse','train_loss','split_loss')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name,records):
    records=list(records)
    if not records: raise ValueError(name+' has no rows')
    fields=list(dict.fromkeys(key for row in records for key in row))
    with (HERE/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(records)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestring():
    return datetime.now(timezone.utc).isoformat()


def frozen_input():
    marker=COLLECT/'FROZEN.json'
    assert marker.exists(), 'Evaluation must wait for the global freeze.'
    assert sha(COLLECT/'PROTOCOL.md')==PROTOCOL
    baseline=json.loads((HERE/'one_shot_snapshot.json').read_text())
    assert all(sha(ROOT/file)==digest for file,digest in baseline.items())
    return json.loads(marker.read_text())


def source(task):
    return 'arithmetic' if task in ('add','subtract') else 'objective'


def original_weights():
    registry={}
    for folder in ('arithmetic','objective'):
        root=ROOT/'focus'/folder
        for case in json.loads((root/'manifest.json').read_text()):
            f=root/'trials'/(case['id']+'.reply.txt')
            raw=json.loads(f.read_bytes()) if folder=='arithmetic' else parse(f.read_text())
            assert len(raw)==len(case['sizes'])
            for size,values in zip(case['sizes'],raw):
                task,seed=case['task'],case['seed']
                p=np.asarray(values,dtype=float)
                n=4*size+1 if folder=='arithmetic' else len(initial(task,size,seed))
                assert p.shape==(n,) and np.isfinite(p).all()
                registry[task,size,seed,case['arm']]=p
    assert len(registry)==174
    return registry


def raw_prediction(task,size,theta,x):
    with np.errstate(over='ignore',invalid='ignore',under='ignore',divide='ignore'):
        return arithmetic_forward(size,theta,x)[0] if source(task)=='arithmetic' else predict(task,size,theta,x)


def measures(task,size,theta,x,y):
    pred=raw_prediction(task,size,theta,x)
    with np.errstate(over='ignore',invalid='ignore',under='ignore',divide='ignore'):
        residual=(pred-y)/np.maximum(1.,np.abs(y))
        loss=float(.5*np.mean(residual**2))
        error=rmse(pred,y)
    eq=pred==y
    hits=eq if eq.ndim==1 else eq.all(axis=1)
    return dict(n_examples=len(x),n_scalars=y.size,exact_count=int(hits.sum()),
                exact_fraction=float(hits.mean()),raw_rmse=error,split_loss=loss)


@cache
def feedback(task,seed):
    if task in SERIES:
        x=np.arange(9,17)
    else:
        rng=np.random.default_rng(7810000+10000*TASKS.index(task)+seed)
        dimension={'add':2,'subtract':2,'sort':4,'sum':6}.get(task)
        shape=(256,dimension) if dimension else 256
        x=rng.uniform(0 if task=='selfpower' else -3,3,shape)
    return x,reference(task,x)


def confirmation_seed():
    frozen_input()
    path=HERE/'confirmation_seed.json'
    if not path.exists():
        record=dict(host_random_seed=secrets.randbits(128),utc=timestring(),
                    global_freeze_sha256=sha(COLLECT/'FROZEN.json'))
        with path.open('x') as stream: json.dump(record,stream,indent=2);stream.write('\n')
    record=json.loads(path.read_text())
    assert record['global_freeze_sha256']==sha(COLLECT/'FROZEN.json')
    return record['host_random_seed']


@cache
def confirmation(task,entropy):
    if task in SERIES: return {}
    rng=np.random.default_rng(np.random.SeedSequence([entropy,TASKS.index(task)]))
    n=4096
    dim={'add':2,'subtract':2,'sort':4,'sum':6}.get(task)
    shape=(n,dim) if dim else n
    center=rng.uniform(0 if task=='selfpower' else -3,3,shape)
    if dim:
        outer=rng.uniform(3,10,shape)*rng.choice([-1,1],shape)
    elif task=='selfpower': outer=rng.uniform(3,10,n)
    else: outer=np.concatenate((rng.uniform(-10,-3,n),rng.uniform(3,10,n)))
    return {name:(x,reference(task,x)) for name,x in
            [('confirmation_central',center),('confirmation_outer',outer)]}


@cache
def datasets(task,seed,entropy):
    fixed=inputs(task)
    dev=np.arange(17,33) if task in SERIES else fixed['inner']
    sets={'train':observed(task,seed),'feedback_dev':feedback(task,seed),
          'dev':(dev,reference(task,dev)),'outer':(fixed['outer'],reference(task,fixed['outer']))}
    if task in SERIES: sets['legacy_early_9_32']=(fixed['early'],reference(task,fixed['early']))
    if entropy is not None: sets.update(confirmation(task,entropy))
    return sets


def description(task,split):
    return {'train':'observed','dev':('untouched indices 17–32' if task in SERIES else 'untouched original central test'),
            'outer':('indices 33–64' if task in SERIES else 'original outer held-out'),
            'feedback_dev':('selection indices 9–16' if task in SERIES else '256 new central selection examples'),
            'legacy_early_9_32':'original early indices 9–32, overlapping feedback 9–16',
            'confirmation_central':'fresh post-freeze central confirmation',
            'confirmation_outer':'fresh post-freeze outer confirmation'}[split]


def score_vector(task,size,seed,method,theta,entropy,allowed=None):
    sets=datasets(task,seed,entropy)
    train=measures(task,size,theta,*sets['train'])
    rows=[]
    for name,(x,y) in sets.items():
        if allowed and name not in allowed: continue
        metrics=train if name=='train' else measures(task,size,theta,x,y)
        row=dict(protocol='objective-matched iterative audit',task=task,size=size,params=len(theta),
                 seed=seed,method=method,split=name,split_description=description(task,name),
                 **metrics,train_loss=train['split_loss'])
        rows.append({k:row[k] for k in SCHEMA})
    return rows


def control_weights(wsp):
    by_case=defaultdict(dict)
    for (task,size,seed,arm),p in wsp.items(): by_case[task,size,seed][arm]=p
    with np.load(ROOT/'extension/optimizer/states.npz') as old, np.load(ROOT/'focus/arithmetic/fits.npz') as arithmetic, np.load(ROOT/'focus/lbfgs/weights.npz') as gradients:
        for (task,size,seed),group in by_case.items():
            for method in ('gd','gd_lbfgs','adam_lbfgs','lbfgs_xavier'):
                group[method]=gradients[f'{task}_{size}_{seed}_{method}']
            for method in ('adam_bfgs','ls_xavier'):
                if source(task)=='arithmetic':
                    key='trf_xavier' if method=='ls_xavier' else method
                    group[method]=arithmetic[f'{task}_{seed}_{size}_{key}']
                else: group[method]=old[f'{task}_{size}_{seed}_{method}']
            assert set(group)==set(CONTROLS)
    return by_case


def compare_original(all_rows,canonical):
    checks=0
    for r in all_rows:
        task,size,seed,method,split=(r[k] for k in ('task','size','seed','method','split'))
        if task in SERIES:
            if split=='dev': continue
            if split=='legacy_early_9_32': split='dev'
        if split not in SPLITS: continue
        b=canonical[task,size,seed,method,split]
        assert r['n_examples']==int(b['n_examples']) and r['n_scalars']==int(b['n_scalars'])
        assert r['exact_count']==int(b['exact_count'])
        for field in ['raw_rmse','train_loss']:
            assert np.isclose(r[field],float(b[field]),rtol=4e-13,atol=1e-30),(r,b,field)
        checks+=1
    return checks
