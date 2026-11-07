"""Publication learning curves from immutable WSP interaction records.

No fitting, evaluation queries, model calls or synthetic iteration points.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, LogFormatterSciNotation, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'focus/iterative/trajectories'
ARMS = ('examples', 'formula')
SPLITS = ('train', 'dev', 'outer')
TASKS = {'square':'Square', 'sin':'Sine', 'selfpower':'Positive self-power',
         'tanh':'Hyperbolic tangent', 'add':'Addition', 'subtract':'Subtraction',
         'sum':'Running sum', 'sort':'Sort four', 'fib':'Fibonacci', 'tri':'Triangular'}
COLORS = {'examples':'#246B82', 'formula':'#BD6B43'}
LABELS = {'examples':'Examples only', 'formula':'Formula and examples'}
INK = '#213642'
GRID = '#DDE5E8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.4,'axes.titlesize':10,
    'axes.labelsize':8.4, 'xtick.labelsize':8,'ytick.labelsize':8,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#9EAEB6',
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def write_csv(name, rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metric(row):
    return {'scaled_loss':float(row['split_loss']), 'scaled_rmse':math.sqrt(2*float(row['split_loss'])),
            'raw_rmse':float(row['raw_rmse']), 'exact_count':int(row['exact_count']),
            'n_examples':int(row['n_examples'])}


def history_metric(row):
    return {'scaled_loss':row['loss'], 'scaled_rmse':math.sqrt(2*row['loss']),
            'raw_rmse':row['rmse'], 'exact_count':row['exact'], 'n_examples':row['n']}


def check_close(a,b):
    # Independent cached scores can differ by binary64 reduction rounding.
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-300),(a,b)


def build_records():
    staged=read_csv(ROOT/'focus/iterative_eval/stage_scores.csv')
    cache={(r['task'],int(r['size']),int(r['seed']),r['method'],int(r['round']),r['stage'],r['split']):r for r in staged}
    final={(r['task'],int(r['size']),int(r['seed']),r['method'],r['split']):r for r in read_csv(ROOT/'focus/iterative_eval/extended.csv')}
    rows=[];proposals=[];decisions=[];source_hashes={};verified=0;dev_changes=[]
    for path in sorted(SOURCE.glob('*.json')):
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        d=json.loads(path.read_text())
        for size,s in d['states'].items():
            size=int(size);key=(d['task'],size,d['seed'],d['arm'])
            common=dict(task=d['task'],size=size,seed=d['seed'],arm=d['arm'])
            state={split:history_metric(s['initial_'+split]) for split in ('train','dev')}
            state['outer']=metric(cache[key+(0,'round_zero','outer')])
            source_round=0
            by_round={h['round']:h for h in s['history']}
            prior_dev=state['dev']['scaled_loss']
            for r in range(4):
                h=by_round.get(r)
                if h:
                    assert r>0
                    state.update({split:history_metric(h['incumbent_'+split]) for split in ('train','dev')})
                    state['outer']=metric(cache[key+(r,'selected','outer')])
                    if h['accepted']:source_round=r
                    decisions.append(dict(**common,round=r,valid=h['valid'],accepted=h['accepted'],request=h['request_id']))
                    assert state['dev']['scaled_loss'] <= prior_dev,(key,r)
                    for split in ('train','dev'):
                        c=metric(cache[key+(r,'selected','feedback_dev' if split=='dev' else split)])
                        for m in ('scaled_loss','raw_rmse'):check_close(state[split][m],c[m]);verified+=1
                    for split in SPLITS:
                        pm=None
                        if h['valid']:
                            pm=history_metric(h['proposal_'+split]) if split!='outer' else metric(cache[key+(r,'proposal',split)])
                        proposals.append(dict(**common,round=r,split=split,valid=h['valid'],accepted=h['accepted'],
                            request=h['request_id'],**(pm or {k:'' for k in state[split]})))
                else:
                    assert r==0 or r>s['attempts'],(key,r,'unexpected missing history')
                for split in SPLITS:
                    rows.append(dict(**common,round=r,split=split,
                        attempted=bool(h),accepted_this_round=bool(h and h['accepted']),
                        source_round=source_round,carried_after_stop=bool(r>0 and h is None),
                        stop_reason=s['stop'],**state[split]))
                dev_changes.append(state['dev']['scaled_loss']-prior_dev)
                prior_dev=state['dev']['scaled_loss']
            for split in SPLITS:
                c=metric(final[key[:3]+('iter_'+key[3],'feedback_dev' if split=='dev' else split)])
                for m in ('scaled_loss','raw_rmse'):check_close(state[split][m],c[m]);verified+=1
    assert len(rows)==174*4*3
    assert len(decisions)==332 and sum(r['accepted'] for r in decisions)==217
    assert sum(not r['valid'] for r in decisions)==1
    assert len({(r['task'],r['size'],r['seed']) for r in rows if r['arm']=='examples'})==87
    assert all(d<=0 for d in dev_changes)
    write_csv('selected_rounds.csv',rows);write_csv('proposal_rounds.csv',proposals)
    write_csv('decisions.csv',decisions)
    summary=[]
    for arm in ARMS:
        for r in range(4):
            for split in SPLITS:
                group=[x for x in rows if x['arm']==arm and x['round']==r and x['split']==split]
                assert len(group)==87
                summary.append(dict(arm=arm,round=r,split=split,settings=len(group),
                    median_scaled_rmse=float(np.median([x['scaled_rmse'] for x in group])),
                    median_raw_rmse=float(np.median([x['raw_rmse'] for x in group])),
                    exact_settings=sum(x['exact_count']==x['n_examples'] for x in group),
                    attempted_settings=sum(x['attempted'] for x in group),
                    accepted_settings=sum(x['accepted_this_round'] for x in group),
                    carried_after_stop_settings=sum(x['carried_after_stop'] for x in group)))
    write_csv('round_summary.csv',summary)
    per_task=[]
    for arm in ARMS:
        for task in TASKS:
            for r in range(4):
                for split in SPLITS:
                    group=[x for x in rows if x['arm']==arm and x['task']==task and x['round']==r and x['split']==split]
                    per_task.append(dict(arm=arm,task=task,round=r,split=split,settings=len(group),
                        median_raw_rmse=float(np.median([x['raw_rmse'] for x in group])),
                        mean_raw_rmse=float(np.mean([x['raw_rmse'] for x in group])),
                        median_scaled_rmse=float(np.median([x['scaled_rmse'] for x in group]))))
    write_csv('task_round_summary.csv',per_task)
    audit=dict(trajectories=174,conditions=2,settings_per_condition=87,rounds=[0,1,2,3],
        selected_rows=len(rows),proposal_decisions=len(decisions),accepted_proposals=217,invalid_proposals=1,
        feedback_and_final_numeric_cells_crosschecked=verified,development_loss_increases=0,
        source_hashes=source_hashes,
        sources={'stages_sha256':hashlib.sha256((ROOT/'focus/iterative_eval/stage_scores.csv').read_bytes()).hexdigest(),
                 'endpoints_sha256':hashlib.sha256((ROOT/'focus/iterative_eval/extended.csv').read_bytes()).hexdigest()},
        aggregation='Medians over all87 fixed settings in each information condition. Stopped incumbents are retained.',
        exact_initial_train_and_development_per_arm=30,main_zero_clipping='none; all displayed aggregate medians are positive')
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return rows,proposals,summary,per_task


def style(ax):
    for name in ('top','right'):ax.spines[name].set_visible(False)
    ax.grid(axis='y',which='major',color=GRID,linewidth=.65,zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([0,1,2,3],['0\nOne-shot','1','2','3'])
    ax.tick_params(axis='both',length=3)
    ax.set_xlim(-.12,3.18)
    ax.set_xlabel('Revision opportunity')


def save(fig,name):
    fig.savefig(HERE/(name+'.pdf'),metadata={'Title':name.replace('_',' ')})
    fig.savefig(HERE/(name+'.png'),dpi=220)
    plt.close(fig)


def main_figure(summary):
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.15),gridspec_kw={'wspace':.34})
    fig.subplots_adjust(left=.085,right=.985,top=.77,bottom=.23)
    panels=[('train','a  Training fit'),('dev','b  Development feedback'),
            ('outer','c  Extrapolation test')]
    for j,(split,title) in enumerate(panels):
        ax=axes[j];style(ax);ax.set_title(title,loc='left',fontweight='bold',fontsize=9,pad=11)
        for arm in ARMS:
            vals=[next(float(x['median_scaled_rmse']) for x in summary if x['arm']==arm and x['split']==split and int(x['round'])==r) for r in range(4)]
            assert min(vals)>0
            ax.plot(range(4),vals,color=COLORS[arm],lw=1.8,marker='o' if arm=='examples' else 'D',ms=3.8,
                    markerfacecolor='white' if arm=='formula' else COLORS[arm],label=LABELS[arm],zorder=3)
        ax.set_yscale('log');ax.set_xlabel('')
        ax.yaxis.set_minor_locator(NullLocator())
        if split=='train':
            ax.set_ylabel('Median scaled RMSE')
            ax.set_ylim(1e-16,.02);ax.set_yticks([1e-15,1e-11,1e-7,1e-3])
        elif split=='dev':
            ax.set_ylim(2e-5,.08);ax.set_yticks([1e-4,1e-3,1e-2])
        else:
            ax.set_ylim(.003,.5);ax.set_yticks([.005,.02,.1,.4])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.985),ncol=2,frameon=False,
               handlelength=2.5,columnspacing=2.5)
    fig.text(.55,.035,'Revision opportunity (0 = one-shot WSP)',ha='center',fontsize=8.3,color=INK)
    save(fig,'learning_main')


def task_figure(per_task):
    fig,axes=plt.subplots(5,2,figsize=(6.9,8.5))
    fig.subplots_adjust(left=.12,right=.98,top=.9,bottom=.075,wspace=.36,hspace=.63)
    for ax,(task,name) in zip(axes.flat,TASKS.items()):
        style(ax);ax.set_title(name,loc='left',fontweight='bold',pad=5)
        values=[]
        for arm in ARMS:
            for split in ('train','dev'):
                vals=[next(x['mean_raw_rmse'] for x in per_task if x['arm']==arm and x['task']==task and x['split']==split and x['round']==r) for r in range(4)]
                values+=vals
                ax.plot(range(4),vals,color=COLORS[arm],ls='--' if split=='train' else '-',lw=1.3,
                    marker='o' if split=='dev' else 's',ms=2.8,mfc='white' if split=='train' else COLORS[arm],zorder=3)
        # No synthetic epsilon: an exact-zero panel uses a linear scale.
        nonzero=[v for v in values if v>0]
        if nonzero and len(nonzero)==len(values):
            ax.set_yscale('log')
            lo,hi=min(nonzero)*.65,max(nonzero)*1.6
            ax.set_ylim(lo,hi)
            candidates=[m*10.**e for e in range(math.floor(math.log10(lo))-1,math.ceil(math.log10(hi))+1)
                        for m in (1,2,5) if lo<=m*10.**e<=hi]
            if len(candidates)>4:
                candidates=[candidates[i] for i in np.linspace(0,len(candidates)-1,4).round().astype(int)]
            ax.set_yticks(candidates)
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
            ax.yaxis.set_minor_locator(NullLocator())
        elif not nonzero:
            ax.set_ylim(-.08,.3);ax.set_yticks([0]);ax.text(.5,.68,'All selected errors are zero',ha='center',transform=ax.transAxes,fontsize=8,color='#526875')
        else:
            raise AssertionError('Mixed exact-zero and positive means need explicit zero treatment')
        ax.set_ylabel('Mean raw RMSE',fontsize=7.7)
        ax.set_xticks([0,1,2,3], ['0','1','2','3'])
        ax.set_xlabel('')
        ax.tick_params(labelsize=7.5)
    handles=[Line2D([0],[0],color=COLORS[a],ls='--' if s=='train' else '-',marker='s' if s=='train' else 'o',ms=3,lw=1.5,
                    mfc='white' if s=='train' else COLORS[a],label=LABELS[a]+' · '+('training' if s=='train' else 'development')) for a in ARMS for s in ('train','dev')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.988),ncol=2,frameon=False,fontsize=8)
    fig.text(.55,.018,'Revision opportunity (0 = one-shot WSP)',ha='center',fontsize=8,color=INK)
    save(fig,'learning_tasks')


def proposal_figure(rows,proposals):
    fig,axes=plt.subplots(2,2,figsize=(6.9,4.9))
    fig.subplots_adjust(left=.135,right=.98,top=.84,bottom=.14,wspace=.41,hspace=.6)
    for k,(task,size) in enumerate([('square',4),('sort',4)]):
        for j,split in enumerate(('train','dev')):
            ax=axes[k,j];style(ax)
            title=TASKS[task]+(' · 4 hidden units' if task=='square' else ' · 4 residual blocks')
            ax.set_title(title+'\n'+('Training examples' if split=='train' else 'Development feedback'),loc='left',fontweight='bold',fontsize=9,pad=6)
            rr=[x for x in rows if x['task']==task and x['size']==size and x['arm']=='examples' and x['seed']==164 and x['split']==split]
            rr=sorted(rr,key=lambda x:x['round'])
            ax.plot([x['round'] for x in rr],[x['scaled_rmse'] for x in rr],color=COLORS['examples'],marker='o',ms=4,lw=1.6,zorder=3)
            pp=[x for x in proposals if x['task']==task and x['size']==size and x['arm']=='examples' and x['seed']==164 and x['split']==split and x['valid']]
            for x in pp:
                retained=next(r['scaled_rmse'] for r in rr if r['round']==x['round'])
                if not x['accepted']:
                    ax.plot([x['round'],x['round']],[retained,x['scaled_rmse']],color=COLORS['formula'],lw=.8,zorder=2)
                    ax.scatter(x['round'],x['scaled_rmse'],color=COLORS['formula'],marker='x',s=30,linewidths=1.3,zorder=4)
                else:
                    ax.scatter(x['round'],x['scaled_rmse'],facecolors='white',edgecolors=COLORS['examples'],marker='o',s=42,linewidths=1.1,zorder=4)
            ax.set_yscale('log');ax.set_ylabel('Scaled RMSE')
            ax.set_xlabel('')
            ax.set_xticks([0,1,2,3],['0','1','2','3'])
    fig.legend(handles=[Line2D([0],[0],color=COLORS['examples'],marker='o',ms=4,lw=1.5,label='Selected weights'),
                        Line2D([0],[0],color=COLORS['examples'],mfc='white',marker='o',ms=5,lw=0,label='Accepted proposal'),
                        Line2D([0],[0],color=COLORS['formula'],marker='x',ms=5,lw=0,label='Rejected proposal')],
               loc='upper center',bbox_to_anchor=(.55,.985),ncol=3,frameon=False,fontsize=8)
    fig.text(.56,.025,'Revision opportunity (0 = one-shot WSP)',ha='center',fontsize=8,color=INK)
    save(fig,'learning_proposals')


def tex():
    (HERE/'learning_main.tex').write_text(r'''\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{learning_main.pdf}
\caption{\textbf{Development gains do not guarantee extrapolation gains.} Each point is the median scaled root mean squared error (RMSE) of selected weights across the same 87 settings in one information condition. Round 0 is one-shot WSP; rounds 1--3 are revision opportunities, with stopped runs retaining their last weights. Development data supply feedback and select revisions. Extrapolation scores were computed only after selection was frozen and never supplied feedback or selected weights. All vertical axes are logarithmic with different ranges. Thirty settings per condition stay exact on all three splits; their zeros remain in the medians. Lines join recorded summaries, not continuous optimization steps.}\label{fig:wsp-learning}
\end{figure}
''')
    (HERE/'learning_appendix.tex').write_text(r'''\subsection{Recorded iterative WSP learning curves}\label{app:learning-curves}
The plots reconstruct all 174 trajectories from their original interaction records: 87 task, size and start settings in each information condition. Each selected training and development score agrees with the separately archived evaluator, and each final state agrees with the published endpoint. A rejected proposal leaves the selected weights unchanged. After early stopping, later opportunities display that retained state; they are not additional model calls. No new fitting or test-guided selection was performed for these plots. The data archive includes every selected state, all proposal decisions, raw and scaled errors, and exact counts.

Figure~\ref{fig:wsp-learning} uses the same scaled error as the main endpoint summary: divide each output error by the larger of one and the target magnitude before computing RMSE. It aggregates by the median over settings, not by first averaging within each task. No uncertainty band is drawn; these are descriptive summaries of the fixed cases. The median training error need not decrease because acceptance prioritizes development loss. The examples-only median development error falls from $0.0214$ to $5.89\times10^{-5}$, while the median training error rises from $9.03\times10^{-16}$ to $1.06\times10^{-3}$. Formula-and-examples medians change from $0.0248$ to $5.89\times10^{-5}$ on development and from $6.62\times10^{-16}$ to $1.32\times10^{-3}$ on training. The maximum of three attempted rounds does not establish convergence.

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{learning_tasks.pdf}
\caption{\textbf{Training and development curves for all ten tasks.} Each point is the mean raw RMSE over that task's tested sizes and three starts, using selected weights and carrying stopped states forward. Solid lines show development feedback; dashed lines show training. Colors distinguish examples only from formula and examples. Each task has its own logarithmic vertical range. Running sum is exact throughout and uses a linear axis at zero. Unlike the main learning curves, these within-task summaries use means, preserving the contribution of low-capacity networks that can be hidden by a zero median. The sorting panel reports errors across all output coordinates. Development data are reused for feedback and selection.}\label{fig:learning-tasks}
\end{figure}

\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{learning_proposals.pdf}
\caption{\textbf{A proposal is not necessarily the selected network.} Two illustrative examples-only trajectories at seed 164 show every valid proposal, the selected state, and training versus development scores. The square network has four hidden units; the sorting network has four residual blocks. Open circles mark accepted proposals and crosses mark rejected proposals. Vertical axes are logarithmic with separate ranges. Rejection leaves the selected curve flat. Selection minimizes scaled development loss first, using training loss only to break a tie. These examples explain selection behavior; Figure~\ref{fig:learning-tasks} covers the full task set.}\label{fig:learning-proposals}
\end{figure}
''')


def main():
    rows,proposals,summary,per_task=build_records()
    main_figure(summary);task_figure(per_task);proposal_figure(rows,proposals);tex()
    print(json.dumps({'selected_rows':len(rows),'proposal_rows':len(proposals),'figures':3,'main_size_inches':[7.1,3.15]}))

if __name__=='__main__':main()
