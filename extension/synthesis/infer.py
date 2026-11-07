"""Execute an archived original or repaired neural graph; no target functions at inference."""
import argparse,json
import numpy as np
from experiment import Compiler,HERE
def load(uid,stage):
    trial=next((r for r in json.loads((HERE/'manifest.json').read_text()) if r['id']==uid),None)
    if not trial:raise ValueError('Unknown trial ID')
    folder=(HERE/'repair') if stage=='repair' else HERE
    raw=(folder/'trials'/(uid+'.reply.txt')).read_text().strip()
    if raw.startswith(chr(96)*3):raw=raw.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
    g=Compiler(trial['arm']);g.block(json.loads(raw));return g
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=float);p.add_argument('--id',default='3354251e2');p.add_argument('--stage',choices=['original','repair'],default='original')
    a=p.parse_args();g=load(a.id,a.stage);print(g.run(np.array(a.inputs)))
