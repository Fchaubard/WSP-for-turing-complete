"""Audit the published denominators and pairing across the independently archived studies."""
from collections import Counter
import csv,json
from pathlib import Path
base=Path(__file__).resolve().parent
def j(path):return json.loads((base/path).read_text())
def cs(path):
    with (base/path).open() as f:return list(csv.DictReader(f))
def assert_n_returned(path,n):
    rs=j(path/'manifest.json');assert len(rs)==n
    for r in rs:
        q=base/path/'trials'/r['id']
        assert q.with_suffix('.prompt.txt').is_file() and q.with_suffix('.reply.txt').is_file()
        m=json.loads(q.with_suffix('.meta.json').read_text());assert m['requested_model']=='gpt-6-astra' and set(m['completed_items'])=={'agent_message'}
for directory,n in [('writer',60),('synthesis',18),('counters/literal',6)]:
    assert_n_returned(Path(directory),n)
assert len(cs('writer/status.csv'))==180 and {z['status'] for z in cs('writer/status.csv')}=={'valid'}
assert len(j('writer/bp.json'))==90
o=cs('optimizer/results.csv');assert len(o)==945
groups=Counter(r['method'] for r in o)
assert groups['adam_bfgs']==groups['ls_xavier']==groups['ls_examples']==groups['ls_formula']==90
summ=j('optimizer/summary.json')
s={r['method']:r for r in summ['methods']}
assert [s[m]['hinge_certified_near'] for m in ('adam800','adam_bfgs','ls_xavier','ls_formula','ls_examples')]==[0,1,5,7,9]
assert sum(z['trajectories']['outer']['runs'] for z in j('counters/results.json')['programs'])==161
assert sum(z['trajectories']['outer']['ticks'] for z in j('counters/results.json')['programs'])==18330
assert all(z['algebraic_certificate']=='output_relu_clips_decrement' for z in j('counters/literal/scores.json'))
assert Counter(z['status'] for z in j('synthesis/results.json'))=={'scored':16,'invalid':2}
assert len(j('synthesis/repair/manifest.json'))==6
assert len(cs('convex/scores.csv'))==2016
for task in ('sin','selfpower'):
    r=next(q for q in cs('scalar/'+('relu.csv' if task=='sin' else 'relu_power.csv')) if q['split']=='fresh' and q['method']==('relu_rotation_two_word' if task=='sin' else 'relu_selfpower_two_word') and int(q['order'])==(83 if task=='sin' else 84))
    assert int(r['correct'])==int(r['n'])==32768
print('Verified 180 paired proposals, 90 paired fits, 945 optimizer records, 2,016 frozen-feature measurements, six certified machine proposals, 18 original free-topology replies and six revision pairs.')

assert j('square_full/results.json')['engineered']['operations']==3352
assert j('audit/square_certificate_review.json')['ops']==3352
