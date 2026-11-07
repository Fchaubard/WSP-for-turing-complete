"""Independent validation of saved iterative plotting records; never modifies inputs."""
from pathlib import Path
import csv
import json
import math
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def read(path):
 with path.open() as f:return list(csv.DictReader(f))

def close(a,b):
 a,b=float(a),float(b)
 return a==b or abs(a-b)<=1e-12*max(abs(a),abs(b))

def main():
 records=[]; proposals=[]; trajectories=[]
 for path in sorted((ROOT/'focus/iterative/trajectories').glob('*.json')):
  t=json.loads(path.read_text())
  for size,state in t['states'].items():
   key=dict(task=t['task'],size=int(size),seed=int(t['seed']),arm=t['arm'])
   history={int(h['round']):h for h in state['history']}
   oldtrain=state['initial_train'];olddev=state['initial_dev']
   for r in range(4):
    if r in history:
     h=history[r]
     assert close(h['before_train']['loss'],oldtrain['loss'])
     assert close(h['before_dev']['loss'],olddev['loss'])
     if h['valid']:
      expected=(h['proposal_dev']['loss'],h['proposal_train']['loss'])<(olddev['loss'],oldtrain['loss'])
      assert h['accepted']==expected,(key,r)
      for split in ('train','dev'):
       p=h['proposal_'+split]
       proposals.append(dict(**key,round=r,split=split,accepted=h['accepted'],loss=p['loss'],raw_rmse=p['rmse'],exact=p['exact'],n=p['n']))
     assert h['incumbent_dev']['loss']<=olddev['loss'],(key,r)
     oldtrain=h['incumbent_train'];olddev=h['incumbent_dev']
    for split,metric in [('train',oldtrain),('dev',olddev)]:
     records.append(dict(**key,round=r,split=split,loss=metric['loss'],scaled_rmse=math.sqrt(2*metric['loss']),raw_rmse=metric['rmse'],exact=metric['exact'],n=metric['n'],attempted=(r in history),accepted=history.get(r,{}).get('accepted',False)))
   trajectories.append((key,state))
 assert len(trajectories)==174
 assert len(records)==1392
 assert Counter(r['arm'] for r in records)==dict(examples=696,formula=696)
 endpoints={(r['task'],int(r['size']),int(r['seed']),r['method'].removeprefix('iter_'),r['split']):r for r in read(ROOT/'focus/iterative_eval/extended.csv') if r['method'].startswith('iter_')}
 checked=0
 for r in records:
  if r['round']!=3:continue
  split='feedback_dev' if r['split']=='dev' else 'train'
  e=endpoints[r['task'],r['size'],r['seed'],r['arm'],split]
  assert close(r['raw_rmse'],e['raw_rmse']),(r,e)
  assert close(r['loss'],e['split_loss']),(r,e)
  assert r['exact']==int(e['exact_count'])
  checked+=1
 for name,rows in [('independent_retained.csv',records),('independent_proposals.csv',proposals)]:
  with (HERE/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=dict(trajectories=174,arms=2,settings_per_arm=87,retained_split_round_records=len(records),proposal_split_records=len(proposals),final_split_endpoints_checked=checked,all_dev_losses_nonincreasing=True,all_acceptance_decisions_reproduced=True)
 (HERE/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(summary)

if __name__=='__main__':main()
