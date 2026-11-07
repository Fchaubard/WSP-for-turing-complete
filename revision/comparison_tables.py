"""Build tables by pooling scalar errors first, taking medians across starts second."""
import csv,json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(path):
    with (root/path).open() as f:return list(csv.DictReader(f))
def fmt(x):
    if x==0:return '0'
    if .001<=abs(x)<1000:return f'{x:.3g}'
    a,b=f'{x:.2e}'.split('e')
    return '$'+a+r'\times10^{'+str(int(b))+'}$'
def write(name,rows):
    (root/'paper'/name).write_text('\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule'+'\n')
names=dict(square='Square',sin='Sine',selfpower='Positive self-power',tanh='Hyperbolic tangent',add='Addition',subtract='Subtraction',sum='Running sum',sort='Sort four',fib='Fibonacci',tri='Triangular')
paired=read('extension/writer/table.csv')
write('scaled_rows.tex',[[names[t],*[fmt(float(r['median_scaled'])) for r in paired if r['task']==t]] for t in names])
seq=json.loads((root/'sequence/results.json').read_text())['results'];rows=[]
for t,size,slots,inner,outer in [('sum','1',3,'ID_T8','OOD_T32'),('fib','2',8,'train_indices_0_8','ood_indices_33_64'),('tri','3',15,'train_indices_0_8','ood_indices_33_64'),('sort','g5',15,'ID_T4','OOD_T4')]:
    for split,label in [(inner,'Central'),(outer,'Outer')]:
        row=[names[t],slots,label]
        for method in ['program','bp']:
            group=[v['scores'][split] for v in seq if v['task']==t and v['size']==size and v['method']==method]
            assert len(group)==5
            metric='exact' if t in ['fib','tri'] else 'exact_all'
            row += [f"{100*statistics.median(z[metric] for z in group):.2f}",fmt(statistics.median(z['rmse'] for z in group))]
        rows.append(row)
write('legacy_sequence_rows.tex',rows)
shallow=read('scalar/shallow.csv');rows=[]
for t in ['square','sin','tanh','selfpower']:
    row=[names[t]]
    for method in ['write_local','write_global','adam_lbfgs']:
        q=[r for r in shallow if r['task']==t and r['size']=='63' and r['method']==method and r['split'] in ('left','right')]
        errors=[];absolute=[]
        for seed in {r['seed'] for r in q}:
            group=[r for r in q if r['seed']==seed]; n=sum(int(r['n']) for r in group)
            errors.append(math.sqrt(sum(float(r['rmse'])**2*int(r['n']) for r in group)/n))
            absolute.append(sum(float(r['mae'])*int(r['n']) for r in group)/n)
            assert all(float(r['exact'])==0 for r in group)
        row.extend([fmt(statistics.median(absolute)),fmt(statistics.median(errors))])
    rows.append(row)
write('legacy_scalar_rows.tex',rows)
sparse=read('revision/other_metrics/sparse_paired.csv');short=json.loads((root/'writer/short_scores.json').read_text());rows=[]
for i,pair in enumerate(['(1,1)','(1,1)','(1,1)','(-1/2,2)','(1/2,-1)','(-1,1/2)']):
    rec=short[i];row=['$'+pair+'$',rec['d'],rec['H']]
    for method in ['WSP','BP']:
        q=next(r for r in sparse if r['method']==method and r['case']==str(i+7) and r['split']=='fit_impulse')
        row.extend([q['exact_count']+'/'+q['example_count'],fmt(float(q['rmse']))])
    row += [f"{rec['next_abs']:.3f}",rec['d']+1]
    rows.append(row)
write('sparse_rows.tex',rows)
