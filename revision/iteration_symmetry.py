"""Exact parity audit on frozen shallow scalar WSP networks, without any new proposals."""
import csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'focus/coefficients'))
from audit import symmetry
def main():
    loc=ROOT/'focus/iterative'
    assert (ROOT/'focus/iterative_eval/audit.json').is_file()
    rows=[];perfect=Counter();counts=Counter()
    for fp in sorted((loc/'trajectories').glob('*.json')):
        saved=json.loads(fp.read_text()); task=saved['task']
        if task not in ('sin','square','tanh'):continue
        parity=1 if task=='square' else -1
        for width,state in saved['states'].items():
            for method,key in [('one-shot','initial'),('iterative','weights')]:
                defect,where=symmetry(state[key],int(width),parity)
                rows.append(dict(task=task,width=width,seed=saved['seed'],condition=saved['arm'],method=method,parity='even' if parity==1 else 'odd',is_exact=int(defect==0),max_real_defect=str(defect),witness_x=str(where)))
                perfect[task,method]+=defect==0;counts[task,method]+=1
    assert len(rows)==108 and all(v==18 for v in counts.values())
    with (ROOT/'output/iterative-wsp-symmetry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    assert perfect['sin','one-shot']==0
    statement=('Among the 18 sine networks across both information conditions, the exact symmetry test passed for none of the one-shot networks and '+str(perfect['sin','iterative'])+' of the iteratively selected networks. ')
    statement+='For square, '+str(perfect['square','one-shot'])+' one-shot and '+str(perfect['square','iterative'])+' iterative networks were exactly even; for hyperbolic tangent, '+str(perfect['tanh','one-shot'])+' and '+str(perfect['tanh','iterative'])+' were exactly odd (out of 18 each). These statements concern the real function represented by the stored weights over $[-10,10]$.\n'
    (ROOT/'paper/focus_iteration_symmetry.tex').write_text(statement)
if __name__=='__main__':main()
