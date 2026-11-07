"""Cross-check archived headline numbers and replay every literal-model response."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'writer'))
from probe import score as infinite_score
from short_probe import score_one as finite_score
from task_probe import score as score_task
import numpy as np

def json_file(name):
    return json.loads((ROOT/name).read_text())

def csv_file(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))

def vector(i):
    raw = (ROOT/f'writer/case{i}.reply.txt').read_text().strip()
    if raw.startswith(chr(96)*3):
        raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
    return np.array(json.loads(raw))

def main():
    for i in range(6):
        all_h = infinite_score(i)
        assert all_h['row_certificate'] and all_h['edits'] == all_h['d'] + 1
        assert all(all_h[split]['complete_exact']==512 for split in ['dev','shell'])
        finite = finite_score(i)
        assert finite['edits']==finite['H'] and finite['fit_max']<1e-9 and finite['next_abs']>1e-2
    saved = json_file('writer/task_scores.json')
    for i, old in enumerate(saved):
        trial = score_task(i,vector(i+13))
        assert all(np.isclose(trial[s]['exact'],old[s]['exact'],atol=.005) for s in ['in','shell'])
    rational = json_file('frontier/general.json')
    early = [r for r in rational['horizon_sweep'] if r['imposed_horizon']<r['width']+r['order']]
    assert len(rational['results'])==75 and len(rational['horizon_sweep'])==480 and len(early)==405
    assert all(r['input_edits']+r['recurrent_edits']==r['predicted'] for r in rational['horizon_sweep'])
    assert all(r['first_mismatch_index']==r['imposed_horizon'] for r in early)
    assert all(r['verified_impulses']==32 for r in rational['results'])
    rs = [r for r in csv_file('scalar/compensated.csv') if r['method']=='write_corrected' and r['size']=='48']
    assert {r['split']:round(int(r['n'])*float(r['exact'])) for r in rs if r['split'] in ('stress_shell','midpoint_shell','in')} == {'stress_shell':131072,'midpoint_shell':40,'in':4096}
    orig = json_file('sequence/results.json')['results']
    low = next(r for r in orig if r['method']=='program' and r['task']=='sort' and r['size']=='g5')
    assert round(2048*low['scores']['OOD_T4']['exact_all'])==1105
    robust = json_file('sequence/robust_sort.json')
    assert sum(r['pairs'] for r in robust['pairs'].values())==769636
    assert all(r['numeric_mismatches']==0 for r in robust['pairs'].values())
    assert len(csv_file('frontier/gradient.csv'))==256
    assert all(r['exact_sequences']=='512' for r in csv_file('frontier/gradient.csv'))
    print('Verified 21 archived model proposals, 480 rational prefixes, 75 global certificates,')
    print('square stress tests, sorting audits, and 256 exact recurrent gradient controls.')

if __name__=='__main__':
    main()
