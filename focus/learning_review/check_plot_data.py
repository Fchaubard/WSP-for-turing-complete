"""Compare figure tables with independently reconstructed saved trajectories."""
from pathlib import Path
from collections import defaultdict
import json,csv,math,statistics
R=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent

def read(p):
 with (R/p).open() as f:return list(csv.DictReader(f))
def close(a,b):
 a,b=float(a),float(b)
 return a==b or math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-300)
def key(r):return (r['task'],int(r['size']),int(r['seed']),r['arm'],int(r['round']),r['split'])
independent={key(r):r for r in read('focus/learning_review/independent_retained.csv')}
actual=read('focus/learning_plots/selected_rounds.csv')
stages={(r['task'],int(r['size']),int(r['seed']),r['method'],int(r['round']),r['stage'],r['split']):r for r in read('focus/iterative_eval/stage_scores.csv')}
metricchecks=0
for r in actual:
 k=key(r)
 if r['split']!='outer':
  e=independent[k]
  for x,y in [('scaled_loss','loss'),('scaled_rmse','scaled_rmse'),('raw_rmse','raw_rmse'),('exact_count','exact'),('n_examples','n')]:
   assert close(r[x],e[y]),(k,x,r[x],e[y]);metricchecks+=1
  assert r['attempted']==e['attempted']
  assert r['accepted_this_round']==e['accepted']
 else:
  source=int(r['source_round']);stage='round_zero' if source==0 else 'selected'
  e=stages[k[:4]+(source,stage,'outer')]
  for x,y in [('scaled_loss','split_loss'),('raw_rmse','raw_rmse'),('exact_count','exact_count'),('n_examples','n_examples')]:
   assert close(r[x],e[y]),(k,x,r[x],e[y]);metricchecks+=1
  assert close(r['scaled_rmse'],math.sqrt(2*float(e['split_loss'])));metricchecks+=1
assert len(actual)==2088
props={key(r):r for r in read('focus/learning_review/independent_proposals.csv')}
proposalchecks=0;invalid=0
for r in read('focus/learning_plots/proposal_rounds.csv'):
 k=key(r)
 if r['valid']=='False':
  invalid+=1;assert all(r[x]=='' for x in ['raw_rmse','scaled_loss','exact_count']);continue
 if r['split']!='outer':
  e=props[k]
  for x,y in [('raw_rmse','raw_rmse'),('scaled_loss','loss'),('exact_count','exact'),('n_examples','n')]:
   assert close(r[x],e[y]);proposalchecks+=1
 else:
  e=stages[k[:5]+('proposal','outer')]
  for x,y in [('raw_rmse','raw_rmse'),('scaled_loss','split_loss'),('exact_count','exact_count'),('n_examples','n_examples')]:
   assert close(r[x],e[y]);proposalchecks+=1
assert invalid==3
summarychecks=0
for filename,include_task in [('round_summary.csv',False),('task_round_summary.csv',True)]:
 for r in read('focus/learning_plots/'+filename):
  block=[x for x in actual if x['arm']==r['arm'] and x['round']==r['round'] and x['split']==r['split'] and (not include_task or x['task']==r['task'])]
  assert len(block)==int(r['settings'])
  for metric in ['raw_rmse','scaled_rmse']:
   assert close(statistics.median(float(x[metric]) for x in block),r['median_'+metric]);summarychecks+=1
   if 'mean_'+metric in r:
    assert close(statistics.mean(float(x[metric]) for x in block),r['mean_'+metric]);summarychecks+=1
  if not include_task:
   assert sum(x['exact_count']==x['n_examples'] for x in block)==int(r['exact_settings'])
   for col,target in [('attempted','attempted_settings'),('accepted_this_round','accepted_settings'),('carried_after_stop','carried_after_stop_settings')]:
    assert sum(x[col]=='True' for x in block)==int(r[target])
report=dict(selected_rows_checked=len(actual),selected_metric_cells_checked=metricchecks,valid_proposal_metric_cells_checked=proposalchecks,invalid_split_records_retained=invalid,summary_metric_cells_checked=summarychecks,matched_original_trajectories_and_frozen_evaluation=True,main_aggregate_medians_include_all_zeros=True)
(H/'learning_data_review.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
