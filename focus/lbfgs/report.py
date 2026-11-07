"""Independently check archived controls, then tabulate paired optimizer outcomes."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARMS=('gd','gd_lbfgs','adam_lbfgs','lbfgs_xavier')
SPLITS=('train','dev','outer')
TOL=1e-10


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, records):
    with (HERE/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0])
        writer.writeheader();writer.writerows(records)


def key(row):
    return row['task'],int(row['size']),int(row['seed'])


def central(x,y):
    return math.isclose(float(x),float(y),rel_tol=2e-12,abs_tol=1e-28)


def win_counts(values):
    return dict(lower=sum(x < -TOL for x in values),higher=sum(x > TOL for x in values),
                tied=sum(abs(x)<=TOL for x in values))


def main():
    fits=read(HERE/'scores.csv')
    source=read(ROOT/'focus/endpoints/endpoints.csv')
    old=read(ROOT/'extension/optimizer/results.csv')
    cert=read(ROOT/'focus/refinement/certified_optima.csv')
    norms={key(r):float(r['real_optimum']) for r in cert}
    base={(r['method'],key(r),r['split']):r for r in source}
    own={(r['method'],key(r),r['split']):r for r in fits}
    keys=sorted({key(r) for r in fits})
    assert len(keys)==87 and len(norms)==12 and len(own)==87*6*3
    for k in keys:
        for split in SPLITS:
            r=own['xavier',k,split]
            ref=base['xavier',k,split]
            assert r['exact_count']==ref['exact_count'] and central(r['raw_rmse'],ref['raw_rmse'])
            assert central(r['train_loss'],ref['train_loss'])
    # Sixty-nine shared Adam checkpoints have independent stored score/fit records.
    old_adam={key(r):r for r in old if r['method']=='adam800'}
    n_old=0
    for k in keys:
        if k[0] in ('add','subtract'): continue
        n_old+=1
        row=old_adam[k]
        for split in SPLITS:
            r=own['adam800',k,split]
            assert central(r['exact_fraction'],row[split+'_exact'])
            assert central(r['train_loss'],row['train_loss'])
    assert n_old==69
    # Per-task medians never mix raw errors from different target units.
    pooled=defaultdict(list)
    for r in source:
        if r['method'] in ('examples','formula','adam_bfgs','ls_xavier'):
            pooled[r['task'],r['method'],r['split']].append(r)
    for r in fits:
        pooled[r['task'],r['method'],r['split']].append(r)
    by_task=[]
    for (task,method,split),group in sorted(pooled.items()):
        by_task.append(dict(task=task,method=method,split=split,settings=len(group),
                    perfect_settings=sum(float(r['exact_fraction'])==1. for r in group),
                    median_exact=median(float(r['exact_fraction']) for r in group),
                    median_rmse=median(float(r['raw_rmse']) for r in group),
                    median_train_loss=median(float(r['train_loss']) for r in group)))
    write('by_task.csv',by_task)
    pair_rows=[]
    summaries={}
    diags=read(HERE/'diagnostics.csv')
    for method in ('xavier','adam800',*ARMS):
        tr=[own[method,k,'train'] for k in keys]
        summary=dict(n=87,train_near_zero=sum(float(r['train_loss'])<=TOL for r in tr),
                     median_train_loss=median(float(r['train_loss']) for r in tr),
                     near_certified=sum(abs(float(own[method,k,'train']['train_loss'])-v)<=TOL for k,v in norms.items()),
                     n_certified=12,perfect={})
        for split in SPLITS:
            summary['perfect'][split]=sum(float(own[method,k,split]['exact_fraction'])==1. for k in keys)
        summaries[method]=summary
        if method not in ARMS:continue
        diag=[r for r in diags if r['method']==method]
        summary['joint_evals']=sum(int(r['joint_evals']) for r in diag)
        summary['stage_selection']=dict(Counter(r['selected_stage'] for r in diag))
        for competitor in ('examples','formula','adam_bfgs','ls_xavier'):
            group=[]
            for k in keys:
                one=own[method,k,'outer']; other=base[competitor,k,'outer']
                row=dict(task=k[0],size=k[1],seed=k[2],solver=method,comparator=competitor,
                         train_loss_delta=float(one['train_loss'])-float(other['train_loss']),
                         outer_rmse_delta=float(one['raw_rmse'])-float(other['raw_rmse']),
                         outer_exact_delta=float(one['exact_fraction'])-float(other['exact_fraction']))
                pair_rows.append(row)
                group.append(row)
            summary['vs_'+competitor]={v:win_counts([r[v] for r in group])
                                      for v in ('train_loss_delta','outer_rmse_delta','outer_exact_delta')}
    write('paired.csv',pair_rows)
    report=dict(tolerance=TOL,methods=summaries,verified=dict(xavier_score_rows=261,original_adam_score_rows=207,
               arithmetic_replayed_adam_full_bfgs_matches=18))
    (HERE/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
