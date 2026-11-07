"""Publish the new eight-method comparison, broader controls, and raw-data-derived paper statements."""
from collections import Counter, defaultdict
from statistics import median
from pathlib import Path
import csv, json, math
ROOT=Path(__file__).resolve().parents[1]; EV=ROOT/'focus/iterative_eval'; OUT=ROOT/'output'; PAPER=ROOT/'paper'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def dump(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def fixture(r):return r['task'],int(r['size']),int(r['seed'])
names={'examples':'One-shot WSP: Examples only','iter_examples':'Iterative WSP: Examples only',
       'formula':'One-shot WSP: Formula and examples','iter_formula':'Iterative WSP: Formula and examples',
       'gd':'Gradient descent (backpropagation)','gd_lbfgs':'Gradient descent followed by L-BFGS',
       'adam_bfgs':'Adam followed by full BFGS','ls_xavier':'Trust-region least squares',
       'adam_lbfgs':'Adam followed by L-BFGS','lbfgs_xavier':'L-BFGS directly from Xavier',
       'pool_gd':'Gradient descent: full feedback pool','pool_gd_lbfgs':'Gradient descent followed by L-BFGS: full feedback pool'}
def main():
    audit=json.loads((EV/'audit.json').read_text())
    assert audit['audited_final_arrays']==174 and audit['original_wsp_rows_verified']==522
    ext=read(EV/'extended.csv');base={ (r['task'],r['split']):r for r in ext }
    for r in read(ROOT/'focus/feedback_gradient/scores.csv'):
        if r['split']=='feedback': r['split']='feedback_dev'
        other=base[r['task'],r['split']]
        r.update(n_scalars=other['n_scalars'],split_description=other['split_description'],protocol='all original training and all feedback labels: pooled gradient control')
        ext.append(r)
    fields=('task','size','params','seed','method','method_name','split','split_description','n_examples','n_scalars','exact_count','exact_fraction','raw_rmse','train_loss','split_loss','protocol')
    published=[{k:names[r['method']] if k=='method_name' else r[k] for k in fields} for r in ext]
    dump(OUT/'iterative-wsp-endpoints.csv',published)
    idx=defaultdict(dict); group=defaultdict(list)
    for r in ext:
        idx[r['method'],r['split']][fixture(r)]=r
        if r['split'] in ('train','dev','outer'):group[r['task'],int(r['size']),r['split'],r['method']].append(r)
    sized=[]
    for (t,size,s,m),rows in sorted(group.items()):
        assert len(rows)==3
        sized.append(dict(task=t,size=size,params=rows[0]['params'],method=m,method_name=names[m],split=s,n_starts=3,n_examples_per_start=int(rows[0]['n_examples']),
                          median_exact_count=median([int(r['exact_count']) for r in rows]),median_raw_rmse=median([float(r['raw_rmse']) for r in rows])))
    dump(OUT/'iterative-wsp-by-size.csv',sized)
    def compare(m,n,s='outer',metric='raw_rmse'):
        d,e=idx[m,s],idx[n,s];assert d.keys()==e.keys() and len(d) in (69,87)
        v=[float(r[metric])-float(e[k][metric]) for k,r in d.items()];assert all(map(math.isfinite,v))
        return sum(z<0 for z in v),sum(z==0 for z in v),sum(z>0 for z in v)
    def perfect(m):
        rows=idx[m,'outer'];assert len(rows)==87
        return sum(int(r['exact_count'])==int(r['n_examples']) for r in rows.values())
    macros={}
    certs={fixture(r):float(r['real_optimum']) for r in read(ROOT/'focus/refinement/certified_optima.csv')};assert len(certs)==12
    for arm,k in [('examples','E'),('formula','F')]:
        im='iter_'+arm
        compared=compare(im,arm)
        for a,v in zip(('iterBetter','iterTie','iterWorse'),compared):macros[a+k]=v
        macros['oldPerfect'+k]=perfect(arm);macros['iterPerfect'+k]=perfect(im)
        for m,q in [('gd','vsGD'),('gd_lbfgs','vsGL')]:macros[q+k]=compare(im,m)[0]
    for m,name in [('iter_examples','certIterE'),('iter_formula','certIterF'),('gd','certGD'),('gd_lbfgs','certGL')]:
        macros[name]=sum(float(idx[m,'train'][k]['train_loss'])<=opt+1e-10 for k,opt in certs.items())
    for m,n in [('gd','perfectGD'),('gd_lbfgs','perfectGL'),('pool_gd','perfectPoolGD'),('pool_gd_lbfgs','perfectPoolGL')]:macros[n]=perfect(m)
    for m,n in [('pool_gd','vsPoolGD'),('pool_gd_lbfgs','vsPoolGL')]:macros[n]=compare('iter_examples',m)[0]
    s=json.loads((EV/'summary.json').read_text()); cs=s['all_stop_reasons']
    macros.update(revCalls=s['revision_calls']['total'],zeroStops=s['initial_freezes'],exactStops=cs.get('exact_train_dev',0),patienceStops=cs.get('two_nonimprovements',0),budgetStops=cs.get('round_limit',0))
    calls=read(EV/'calls.csv')
    macros.update(latencySec=round(median(float(r['elapsed_seconds']) for r in calls),1),outputTokens=format(s['revision_calls']['output_tokens'],','))
    assert sum(macros[k] for k in ('exactStops','patienceStops','budgetStops'))==174
    (PAPER/'focus_iteration_macros.tex').write_text('% Computed from frozen evaluation files by revision/iteration_release.py.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in macros.items()))
    msg=[]
    for arm in ['examples','formula']:
        result=compare('iter_'+arm,arm,'confirmation_outer')
        msg.append('%d lower, %d tied and %d higher'%result)
    (PAPER/'focus_iteration_confirmation.tex').write_text(r'\paragraph{Independent test draw.} After the weight selection was frozen, we generated a second set of central and outer inputs using a new random seed. On the 69 settings with sampled inputs, the iterative-versus-one-shot outer RMSE comparison was '+msg[0]+' with examples only, and '+msg[1]+' with the formula supplied. The 18 integer settings have fixed target indices and are not repeated as independent draws. The companion data include the scores from both draws.\n')
    from iteration_stages import main as stage_report
    stage_report()
    from iteration_symmetry import main as parity_report
    parity_report()
    print(json.dumps(macros,indent=1))
if __name__=='__main__': main()
