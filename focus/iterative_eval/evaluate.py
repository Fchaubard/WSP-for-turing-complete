"""Independent outer scoring begins only after the collector has frozen all choices."""
import hashlib
import json
from collections import Counter,defaultdict
from statistics import median

import numpy as np

from measure import (
    COLLECT, CONTROLS, HERE, PROTOCOL, ROOT, SERIES, SPLITS, TASKS,
    compare_original, confirmation, confirmation_seed, control_weights, datasets,
    frozen_input, original_weights, read_csv, score_vector, sha, timestring, write_csv,
)
from check_selection import audit as replay_selection

TOL=1e-10


def split_fingerprints(seed):
    rows=[]
    for task in TASKS:
        for init in (164,317,583):
            sets=datasets(task,init,seed)
            previous={}
            for name,(x,y) in sets.items():
                if name=='legacy_early_9_32': continue
                entries={np.ascontiguousarray(a).tobytes() for a in x}
                for other,other_entries in previous.items():
                    assert not entries.intersection(other_entries),(task,init,name,other)
                previous[name]=entries
                rows.append(dict(task=task,seed=init,split=name,n=len(x),scalars=y.size,
                        input_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                        label_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()))
    return rows


def comparator_index():
    rows=read_csv(ROOT/'focus/endpoints/endpoints.csv')+read_csv(ROOT/'focus/lbfgs/scores.csv')
    return {(r['task'],int(r['size']),int(r['seed']),r['method'],r['split']):r for r in rows
            if r['method'] in CONTROLS}


def summarize(final,baselines,stops,calls):
    group=defaultdict(list)
    for r in final+baselines: group[r['method'],r['split']].append(r)
    overview=[]
    by_size=[]
    by_task=[]
    method_sets={}
    for (m,s),rows in group.items():
        overview.append(dict(method=m,split=s,settings=len(rows),
                             perfect=sum(r['exact_count']==r['n_examples'] for r in rows),
                             median_train_loss=median(r['train_loss'] for r in rows)))
        method_sets.setdefault(m,{})[s]={field:overview[-1][field] for field in ('settings','perfect')}
        for task in sorted({r['task'] for r in rows}):
            task_rows=[r for r in rows if r['task']==task]
            buckets=[(task_rows,{},by_task)]
            for size in sorted({r['size'] for r in task_rows}):
                buckets.append(([r for r in task_rows if r['size']==size],{'size':size},by_size))
            for sub,extra,dest in buckets:
                dest.append(dict(task=task,**extra,method=m,split=s,settings=len(sub),
                    perfect=sum(r['exact_count']==r['n_examples'] for r in sub),
                    median_exact_fraction=median(r['exact_fraction'] for r in sub),
                    median_raw_rmse=median(r['raw_rmse'] for r in sub),
                    median_split_loss=median(r['split_loss'] for r in sub)))
    write_csv('by_task.csv',by_task);write_csv('by_architecture.csv',by_size)
    write_csv('overview.csv',overview)
    b={(r['task'],r['size'],r['seed'],r['method'],r['split']):r for r in baselines}
    pairs=[]
    aggregate=defaultdict(list)
    for a in final:
        task,size,seed,method,s=(a[k] for k in ('task','size','seed','method','split'))
        own_arm=method.removeprefix('iter_')
        for comparator in (own_arm,*CONTROLS[2:]):
            old=b[task,size,seed,comparator,s]
            assert a['n_examples']==old['n_examples'] and a['n_scalars']==old['n_scalars']
            row=dict(task=task,size=size,seed=seed,method=method,comparator=comparator,split=s,
                     n_examples=a['n_examples'],n_scalars=a['n_scalars'],
                     prior_exact=old['exact_count'],iterative_exact=a['exact_count'],
                     prior_rmse=old['raw_rmse'],iterative_rmse=a['raw_rmse'],
                     prior_loss=old['split_loss'],iterative_loss=a['split_loss'])
            row['delta_exact_fraction']=a['exact_fraction']-old['exact_fraction']
            for metric in ('rmse','loss'):row['delta_'+metric]=row['iterative_'+metric]-row['prior_'+metric]
            pairs.append(row); aggregate[method,comparator,s].append(row)
    write_csv('paired.csv',pairs)
    by_delta=[]
    for (m,c,s),rows in aggregate.items():
        for metric in ('exact_fraction','rmse','loss'):
            for tol in ((0.,TOL) if metric=='loss' else (0.,)):
                values=[r['delta_'+metric] for r in rows]
                indeterminate=sum(np.isnan(v) for v in values)
                by_delta.append(dict(method=m,comparator=c,split=s,metric=metric,
                              tolerance=tol,settings=len(rows),higher=sum(v>tol for v in values),
                              lower=sum(v < -tol for v in values),tie=sum(abs(v)<=tol for v in values),
                              indeterminate_delta=indeterminate))
    write_csv('paired_summary.csv',by_delta)
    # Cross-split analysis compares each revision to its own immutable first response.
    own_pairs=defaultdict(dict)
    for row in pairs:
        if row['comparator']==row['method'].removeprefix('iter_'):
            own_pairs[row['task'],row['size'],row['seed'],row['method']][row['split']]=row
    changes=[]
    for (task,size,seed,m),group in own_pairs.items():
        row=dict(task=task,size=size,seed=seed,method=m)
        for s,item in group.items():
            for metric in ('loss','rmse','exact_fraction'):
                row[s+'_delta_'+metric]=item['delta_'+metric]
        changes.append(row)
    write_csv('cross_split_changes.csv',changes)
    certified={ (r['task'],int(r['size']),int(r['seed'])):float(r['real_optimum'])
                for r in read_csv(ROOT/'focus/refinement/certified_optima.csv') }
    assert len(certified)==12
    cert_rows=[]
    for r in final+baselines:
        key=(r['task'],r['size'],r['seed'])
        if r['split']=='train' and key in certified:
            gap=r['train_loss']-certified[key]
            cert_rows.append(dict(task=key[0],size=key[1],seed=key[2],method=r['method'],
                                  train_loss=r['train_loss'],real_optimum=certified[key],
                                  additive_gap=gap,within_1e_10=abs(gap)<=TOL))
    write_csv('certified_optima.csv',cert_rows)
    names=('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')
    return dict(methods=method_sets,near_certified_1e_10={
                    m:sum(r['within_1e_10'] for r in cert_rows if r['method']==m)
                    for m in method_sets},revision_calls=dict(total=len(calls),completed=sum(c['status']=='completed' for c in calls),
                        recorded_tool_events=sum(c['recorded_tool_events'] or 0 for c in calls),
                        requests_without_token_or_tool_metadata=sum(c['status']!='completed' for c in calls),
                        sum_request_seconds=sum(c['elapsed_seconds'] for c in calls),
                        **{n:sum(c.get(n,0) for c in calls) for n in names}),
                all_stop_reasons=dict(Counter(r['reason'] for r in stops)),
                initial_freezes=sum(r['attempts']==0 for r in stops),
                arms={a:dict(stops=dict(Counter(r['reason'] for r in stops if r['arm']==a)),
                        improved_arrays=sum(r['accepted']>0 for r in stops if r['arm']==a),
                        accepted_proposals=sum(r['accepted'] for r in stops if r['arm']==a),
                        attempted_proposals=sum(r['attempts'] for r in stops if r['arm']==a))
                        for a in ('examples','formula')})


def model_costs(revision_calls):
    originals=[]
    for folder in ('objective','arithmetic'):
        base=ROOT/'focus'/folder
        for case in json.loads((base/'manifest.json').read_text()):
            meta=json.loads((base/'trials'/(case['id']+'.meta.json')).read_text())
            assert meta['completed_items']=={'agent_message':1} and len(meta['usage'])==1
            originals.append(dict(request=case['id'],arm=case['arm'],task=case['task'],seed=case['seed'],
                elapsed_seconds=meta['elapsed_seconds'],status='completed',**meta['usage'][0]))
    assert len(originals)==60
    write_csv('original_calls.csv',originals)
    output=[]
    for name,records in [('original_reused',originals),('revisions',revision_calls),('total',originals+revision_calls)]:
        for arm in ('formula','examples'):
            block=[x for x in records if x['arm']==arm]
            output.append(dict(stage=name,arm=arm,requests=len(block),completed=sum(x['status']=='completed' for x in block),
                summed_request_seconds=sum(x['elapsed_seconds'] for x in block),
                median_request_seconds=median(x['elapsed_seconds'] for x in block),
                **{k:sum(x.get(k,0) for x in block) for k in
                ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}))
    write_csv('model_costs.csv',output)


def main():
    marker=frozen_input()
    start=timestring()
    raw=original_weights()
    finals,attempts,stops,calls=replay_selection(marker,raw)
    entropy=confirmation_seed()
    write_csv('splits.csv',split_fingerprints(entropy))
    heldout={}
    for task in TASKS:
        for split,(x,y) in confirmation(task,entropy).items():
            heldout[f'{task}_{split}_x']=x
            heldout[f'{task}_{split}_y']=y
    assert len(heldout)==32
    np.savez_compressed(HERE/'confirmation_data.npz',**heldout)
    # Scoring frozen weights. No information is sent to the collector.
    baselines=[]
    for (task,size,seed),weights in control_weights(raw).items():
        for method,p in weights.items():
            baselines.extend(score_vector(task,size,seed,method,p,entropy))
    matches=compare_original(baselines,comparator_index())
    assert matches==87*len(CONTROLS)*3
    final=[]
    for task,size,seed,arm,p in finals:
        final.extend(score_vector(task,size,seed,'iter_'+arm,p,entropy))
    assert len([r for r in final if r['split'] in SPLITS])==174*3
    for target,rows in [('endpoints.csv',final),('comparators.csv',baselines)]:
        write_csv(target,(r for r in rows if r['split'] in SPLITS))
    write_csv('extended.csv',final+baselines)
    stages=[]
    for r in baselines:
        if r['method'] in ('formula','examples') and not r['split'].startswith('confirmation'):
            stages.append(dict(**r,round=0,stage='round_zero',request='',accepted=''))
    for step in attempts:
        for label in ('proposal','selected'):
            p=step[label]
            if p is None or (label=='proposal' and not step['valid']): continue
            for r in score_vector(step['task'],step['size'],step['seed'],step['arm'],p,None):
                stages.append(dict(**r,round=step['round'],stage=label,request=step['request'],accepted=step['accepted']))
    write_csv('stage_scores.csv',stages)
    write_csv('calls.csv',calls);write_csv('stopping.csv',stops)
    model_costs(calls)
    write_csv('proposals.csv',[{k:v for k,v in x.items() if k not in ('proposal','selected')} for x in attempts])
    report=summarize(final,baselines,stops,calls)
    (HERE/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # Freeze and original source files are checked again after all computation.
    assert frozen_input()==marker
    report=dict(started_after_freeze_utc=start,completed_utc=timestring(),global_freeze_sha256=sha(COLLECT/'FROZEN.json'),
                original_files_unchanged=len(json.loads((HERE/'one_shot_snapshot.json').read_text())),
                original_score_rows_verified=matches,original_wsp_rows_verified=174*3,complete_trajectories=len(marker['trajectories']),
                protocol_sha256=PROTOCOL,prompt_shell_sha256='7d9745e8e75b6f5000be9e94aba8eca1db83c647c03b1d8150e6d2babf814b95',
                audited_requests=len(calls),audited_proposals=len(attempts),audited_final_arrays=len(finals),
                invalid_proposals=sum(not a['valid'] for a in attempts),accepted_proposals=sum(a['accepted'] for a in attempts),
                result_sha256={f.name:sha(f) for f in sorted(HERE.glob('*.csv'))},
                summary_sha256=sha(HERE/'summary.json'),evaluation_plan_sha256=sha(HERE/'PLAN.md'),
                confirmation_seed_sha256=sha(HERE/'confirmation_seed.json'),
                confirmation_data_sha256=sha(HERE/'confirmation_data.npz'))
    (HERE/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
