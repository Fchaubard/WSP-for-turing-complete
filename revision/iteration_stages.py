"""Reconstruct all fixed-width cohorts after each opportunity; carry early stops forward."""
from collections import Counter, defaultdict
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]; EV=HERE/'focus/iterative_eval'
def read(name):
    with (EV/name).open(newline='') as f:return list(csv.DictReader(f))
def key(r):return r['task'],int(r['size']),int(r['seed'])
def main():
    assert (EV/'audit.json').is_file()
    records=read('stage_scores.csv'); attempts=read('proposals.csv')
    history=defaultdict(dict)
    for r in records:
        if r['stage'] not in ('round_zero','selected') or r['split']!='outer':continue
        history[r['method'],key(r)][int(r['round'])]=r
    finals={(r['method'],key(r)):r for r in read('endpoints.csv') if r['split']=='outer'}
    rows=[]
    for arm in ('examples','formula'):
        subset={k:v for (a,k),v in history.items() if a==arm};assert len(subset)==87
        for stage in range(4):
            counts=Counter()
            for k,h in subset.items():
                r=h[max(i for i in h if i<=stage)]
                if stage==3:
                    assert float(r['raw_rmse'])==float(finals['iter_'+arm,k]['raw_rmse'])
                    assert r['exact_count']==finals['iter_'+arm,k]['exact_count']
                initial=float(h[0]['raw_rmse']); current=float(r['raw_rmse'])
                counts['lower' if current<initial else 'higher' if current>initial else 'tie']+=1
                counts['exact']+=int(r['exact_count']==r['n_examples'])
            proposed=[r for r in attempts if r['arm']==arm and int(r['round'])==stage]
            rows.append(dict(arm=arm,round=stage,attempted=len(proposed),accepted=sum(r['accepted']=='True' for r in proposed),perfect=counts['exact'],lower=counts['lower'],tie=counts['tie'],higher=counts['higher']))
    with (HERE/'output/iterative-wsp-rounds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    tex=[r'\begin{table}[htbp]\centering\small',r'\caption{What changed with each opportunity to revise? Each information condition keeps all 87 settings throughout. Attempted and accepted count individual size-specific proposals. After a size stops, its last retained weights carry forward. The last four columns are retrospective extrapolation scores; these scores were never shown to Astra. Exact counts settings in which every test prediction matched. Lower, equal and higher compare the retained outer RMSE with that setting\textquotesingle s one-shot outer RMSE.}\label{tab:iterrounds}',r'\begin{tabular}{@{}lrrrrrrr@{}}\toprule',r'Information & Round & Attempted & Accepted & Exact & Lower & Equal & Higher\\\midrule']
    for r in rows:
        i=r['round'];label=('Examples only' if r['arm']=='examples' else 'Formula and examples') if i==0 else ''
        if i==0 and r['arm']=='formula':tex.append(r'\midrule')
        tex.append(' & '.join([label,str(i)]+[str(r[k]) if i else '--' for k in ['attempted','accepted']]+[str(r[k]) for k in ['perfect','lower','tie','higher']])+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    (HERE/'paper/focus_iteration_stages.tex').write_text('\n'.join(tex)+'\n')
if __name__=='__main__':main()
