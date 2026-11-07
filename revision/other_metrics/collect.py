"""Export secondary WSP, BP, and solver controls in the same raw units."""
import csv
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from error_metrics import rmse


def read_json(name):
    return json.loads((ROOT/name).read_text())


def read_csv(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))


def collect():
    rows=[]
    def add(study,task,family,method,setting,seed,split,unit,correct,examples,scalars,error):
        assert 0 <= correct <= examples and examples > 0 and error >= 0
        rows.append(dict(study=study,task=task,family=family,method=method,setting=setting,seed=seed,
                         split=split,exact_unit=unit,exact_count=correct,examples=examples,
                         scalar_outputs=scalars,rmse=error))

    def scores(study,task,family,method,setting,seed,splits):
        for split,measures in splits.items():
            n,length=measures['n'],measures['length']
            scalar=n*length
            if task in ('fib','tri'):
                unit,examples,correct='index',scalar,round(measures['exact']*scalar)
            else:
                unit,examples,correct='whole list' if task=='sort' else 'whole stream',n,round(measures['exact_all']*n)
            add(study,task,family,method,setting,seed,split,unit,correct,examples,scalar,measures['rmse'])

    for row in read_csv('frontier/short.csv'):
        h=int(row['H'])
        for split,num in [('fit',h),('next',1)]:
            add('fixed_support_impulse','sum','WSP','input_vector_solve','d='+row['d']+';H='+str(h),row['seed'],split,
                'impulse coefficient',int(row[split+'_exact']),num,num,float(row[split+'_rmse']))
    for row in read_csv('frontier/trajectory.csv'):
        t=int(row['t'])
        add('edit_support_trajectory','sum','WSP' if row['method']!='omit_one' else 'ablation',row['method'],
            'd='+row['d']+';T='+str(t),row['seed'],row['split'],'whole stream',round(64*float(row['whole_exact'])),
            64,64*t,float(row['rmse']))
    for row in read_csv('frontier/gradient.csv'):
        add('adaptive_gradient','sum','BP','adaptive_unit_step','d=1;T=64',row['seed'],'shell','whole stream',
            int(row['exact_sequences']),int(row['streams']),int(row['streams'])*int(row['ticks']),float(row['rmse']))

    for row in read_csv('revision/other_metrics/sparse_paired.csv'):
        add('fixed_support_writer','leaky_sum',row['method'],row['condition'],'case='+row['case']+';d='+row['d'],
            '',row['split'],row['unit'],int(row['exact_count']),int(row['example_count']),int(row['scalar_count']),float(row['rmse']))

    for row in read_json('sequence/tanh_recurrence.json'):
        family='BP' if row['method']=='bp' else 'WSP'
        key='d='+str(row['width'])
        if family=='WSP':
            key+=';exponent='+str(row['gain_exponent'])
        scores('scaled_tanh','sum',family,'tanh_cell',key,row.get('seed',''),row['scores'])

    for row in read_json('sequence/gauge_control.json'):
        scores('state_rescaling','sum','BP',row['kind'],'d=1',row['seed'],row['scores'])
    syn=read_json('sequence/synthesis.json')
    for task in ['sum','fib','tri','sort']:
        scores('structural_solver',task,'structured solver','example_only','fitted structure','',syn[task]['scores'])
    for gain,measures in syn['gauge'].items():
        scores('state_rescaling','sum','WSP','algebraic_scale','gain='+gain,'',measures)
    for task,row in read_json('sequence/signed_indices.json').items():
        for method,family,entry in [('direct','WSP',row['direct']),('synthesized','structured solver',row['synthesized'])]+[
                ('teacher_forced','BP',z) for z in row['gradient_teacher_forced']]:
            splits={split:v['scores'] for split,v in entry['splits'].items()}
            scores('signed_indices',task,family,method,'d='+str(row['width']),entry.get('seed',''),splits)

    robust=read_json('sequence/robust_sort.json')
    scores('robust_sort','sort','WSP','bounded_selector','shared_50',101,robust['scores'])
    for key,m in robust['pairs'].items():
        add('robust_sort_pairs','sort','WSP','bounded_selector','shared_50',101,key,'ordered pair',
            m['pairs']-m['numeric_mismatches'],m['pairs'],2*m['pairs'],m['rmse'])
    ctrl=read_json('sequence/robust_control.json')
    for entry in ctrl['bp']:
        scores('robust_sort','sort','BP','shared_comparator','shared_50',entry['seed'],entry['scores'])
    scores('robust_sort','sort','ablation','gain_from_training_gap','shared_50','',ctrl['gain_ablation']['scores'])
    signed=read_csv('scalar/signed_selfpower.csv')
    for key,family in [('full_spec','WSP'),('central_min_norm','least squares'),('central_bp','BP')]:
        for split in ['inner','negative_shell','positive_shell']:
            q=[v for v in signed if v['method']==key and v['split']==split]
            target=np.array([float.fromhex(v['reference_hex']) for v in q]);out=np.array([float.fromhex(v['output_hex']) for v in q])
            add('signed_selfpower','selfpower',family,key,'fixed hats / readout',11,split,'index',
                int(np.sum(target==out)),len(q),len(q),rmse(out,target))
    return rows


if __name__=='__main__':
    rows=collect()
    with (HERE/'secondary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows),'secondary rows')
