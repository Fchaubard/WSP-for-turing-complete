"""Render the topology-free response table. No ranking or filtering of replicates."""
from pathlib import Path
import json
here=Path(__file__).parent
rows=json.loads((here/'results.json').read_text())
p=here.parents[1]/'paper'/'synthesis_rows.tex'
def cell(r,k):
    if r['status']!='scored':return r['status']
    return str(r['ops']) if k=='ops' else f"{r['outer']['exact']:,}"
out=[]
for key,name in [('square','Square'),('sin','Sine'),('selfpower','Positive self-power')]:
    for arm,m in [('affine_relu','Affine/ReLU'),('arithmetic_relu','With product')]:
        q=[r for r in rows if r['task']==key and r['arm']==arm]
        out.append(' & '.join([name,m,*[cell(r,'exact') for r in q],', '.join(cell(r,'ops') for r in q)])+r'\\')
p.write_text('\n'.join(out)+'\n'+r'\bottomrule'+'\n')
