"""Aggregate the fixed, comparable architectures; do not maximize over seeds or sizes."""
import csv, json, statistics
from pathlib import Path
here=Path(__file__).resolve().parent
sizes=dict(square=16,sin=16,selfpower=16,tanh=16,add=2,subtract=2,sum=1,sort=5,fib=2,tri=3)
labels={'square':r'$x^2$','sin':r'$\sin x$','selfpower':r'$x^x$, positive','tanh':r'$\tanh x$','add':'Addition','subtract':'Subtraction','sum':'Running sum','sort':'Sort four','fib':'Fibonacci','tri':'Triangular'}
rows=list(csv.DictReader((here/'scores.csv').open()))
out=[];tex=[]
def sci(x):
    if x==0:return '0'
    if .001<=x<100:return f'{x:.3g}'
    a,b=f'{x:.1e}'.split('e');return '$'+a+r'\times 10^{'+str(int(b))+'}$'
for t,k in sizes.items():
    line=[labels[t]]
    for m in ['formula','examples','bp']:
        q=[r for r in rows if r['task']==t and int(r['size'])==k and r['arm']==m and r['split']=='outer']
        assert len(q)==3
        exact=[float(r['exact']) for r in q]
        err=statistics.median(float(r['mean_scaled']) for r in q)
        if len(line)==1:line.append(q[0]['slots'])
        out.append(dict(task=t,size=k,slots=q[0]['slots'],arm=m,perfect=sum(x==1 for x in exact),seeds=len(q),median_exact=statistics.median(exact),median_rmse=statistics.median(float(r["rmse"]) for r in q),median_scaled=err,n_each=int(q[0]['n'])))
        line.extend([f'{100*statistics.median(exact):.1f}',sci(out[-1]["median_rmse"])])
    tex.append(' & '.join(line)+r'\\')
with (here/'table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
(here.parents[1]/'paper'/'writer_rows.tex').write_text('\n'.join(tex)+'\n'+r'\bottomrule'+'\n')
