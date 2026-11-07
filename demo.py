"""Run the archived binary64 circuits. Try: python demo.py sort 7 -3 9 2"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scalar'),str(ROOT/'sequence')]
from benchmark import hinge, native_tanh
from compensated import predict as dense_predict
from robust_sort import evaluate as compare
from run import odd_even_network
from signed_indices import forward as clock_forward

TASKS = ('square','xpowx','sin','tanh','rolling_sum','sort','add','subtract',
         'fibonacci','triangular')


class Circuits:
    def __init__(self, folder=ROOT):
        folder = Path(folder)
        manifest = json.loads((folder/'circuits.json').read_text())
        path = folder/manifest['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Checkpoint SHA-256 differs from circuits.json')
        with np.load(path,allow_pickle=False) as f:
            self.weights = dict(f)
        self.dense = {}
        for name in ['square','integer_power']:
            values = self.weights[name+'_values']
            offset,layers = 0,[]
            for a,b in self.weights[name+'_shapes'].astype(int):
                end = offset+a*b
                layers.append((values[offset:end].reshape(a,b),values[end:end+b]))
                offset = end+b
            assert offset==len(values)
            self.dense[name] = layers
        p = self.weights
        self.sort_graph = []
        for i,(lo,hi) in enumerate(zip(p['sort_ptr'],p['sort_ptr'][1:])):
            edges = list(zip(p['sort_src'][lo:hi],p['sort_coef'][lo:hi]))
            self.sort_graph.append((edges,p['sort_bias'][i],p['sort_relu'][i]))

    def _sort(self,x):
        if len(x)<2:
            return x.copy()
        # Pad with the upper endpoint. Discard the same number of largest
        # elements after sorting; original copies of +10 are preserved as values.
        n = 1<<(len(x)-1).bit_length()
        state = np.full(n,10.)
        state[:len(x)] = x
        # Group independent comparisons for vectorized inference. Dependency
        # levels preserve every wire's order from the original Batcher schedule.
        last = [-1]*n
        levels = []
        for i,j in odd_even_network(n):
            level = max(last[i],last[j])+1
            if level==len(levels):
                levels.append([])
            levels[level].append((i,j))
            last[i]=last[j]=level
        for pairs in levels:
            left,right = np.array(pairs).T
            state[left],state[right] = compare(self.sort_graph,state[left],state[right])
        return state[:len(x)]

    def predict(self, task, values):
        """Return a one-dimensional float64 array. Scalar tasks accept batches."""
        x = np.atleast_1d(np.asarray(values,dtype=np.float64))
        if task not in TASKS or x.ndim!=1 or not np.all(np.isfinite(x)):
            raise ValueError('Expected a known task and a one-dimensional finite input.')
        if task not in ('rolling_sum','sort') and len(x)==0:
            raise ValueError('This task needs at least one input.')
        if task in ('fibonacci','triangular'):
            if np.any(x!=np.trunc(x)) or np.any(np.abs(x)>64):
                raise ValueError('The signed clock accepts integer indices from -64 to 64.')
            weights = [self.weights[task+'_'+k] for k in ('forward','backward','initial','readout')]
            return np.array([clock_forward(weights,int(n)) for n in x])
        if np.any(np.abs(x)>10):
            raise ValueError('The demo uses the declared envelope [-10,10].')
        p = self.weights
        if task=='square':
            return dense_predict(self.dense['square'],x)
        if task=='sin':
            return hinge(hinge(x,p['sin_fold']),p['sin_spline'])
        if task=='tanh':
            return native_tanh(x,p['tanh'])
        if task=='xpowx':
            positive = x>0
            if np.any(x[~positive]!=np.trunc(x[~positive])):
                raise ValueError('Negative noninteger binary64 inputs have no real self-power.')
            out = np.empty_like(x)
            if np.any(positive):
                out[positive] = hinge(x[positive],p['selfpower_spline'])
            if np.any(~positive):
                out[~positive] = dense_predict(self.dense['integer_power'],x[~positive])
            return out
        if task in ('add','subtract'):
            if len(x)!=2:
                raise ValueError('Supply exactly two operands.')
            return np.array([x @ p[task]])
        if task=='rolling_sum':
            A,b,v = p[task]
            h,out = 0.,np.empty_like(x)
            for i,xi in enumerate(x):
                h = A*h+b*xi
                out[i] = v*h
            return out
        return self._sort(x)


def verify(model):
    # Reference functions below are used only in this explicit audit.
    for t in TASKS:
        assert t in json.loads((ROOT/'circuits.json').read_text())['tasks']
    for task,x,ref in [('square',[-7.5,6.25],[56.25,39.0625]),('add',[4.75,-8.],[ -3.25]),
                       ('subtract',[4.75,-8.],[12.75]),('rolling_sum',[7,-9,4],[7,-2,2]),
                       ('fibonacci',[-10,-3,0,10],[-55,2,0,55]),
                       ('triangular',[-10,-3,0,10],[45,3,0,55]),
                       ('xpowx',[-3,0],[-1/27,1]),('tanh',[-7,4],np.tanh([-7,4]))]:
        np.testing.assert_array_equal(model.predict(task,x),ref)
    rng = np.random.default_rng(23)
    for n in (0,1,2,3,4,5,7,8,13,16,37,127):
        x = rng.uniform(-10,10,n)
        if n>3:
            x[:4] = [10.,10.,-10.,np.nextafter(0.,1.)]
        np.testing.assert_array_equal(model.predict('sort',x),np.sort(x))
    x = np.array([-9.,-3.,0.,6.,10.])
    assert np.max(np.abs(model.predict('sin',x)-np.sin(x)))<3e-4
    x = np.array([.1,1.5,4.3,8.1,10.])
    assert np.max(np.abs(model.predict('xpowx',x)/np.power(x,x)-1))<.0102
    x = float.fromhex('0x1.0bb639c98c0b5p+2')
    got = float(model.predict('square',[x])[0])
    target = x*x
    if got != np.nextafter(target,np.inf):
        raise RuntimeError(f'Archived one-ULP failure did not reproduce: {got.hex()}, target {target.hex()}')
    print('All ten checkpoints passed the demo checks. Reproduced the square counterexample.')
    print(f'x = {x.hex()}, circuit = {got.hex()}, binary64 square = {target.hex()}')


def number(arg):
    return float.fromhex(arg) if arg.lstrip('+-').lower().startswith('0x') else float(arg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('task',choices=TASKS+('verify',))
    ap.add_argument('values',nargs=argparse.REMAINDER,help='decimal or hexadecimal floats')
    args = ap.parse_args()
    model = Circuits()
    if args.task=='verify':
        if args.values:
            ap.error('verify takes no inputs')
        verify(model)
        return
    try:
        values = [number(s) for s in args.values if s!='--']
        out = model.predict(args.task,values)
    except ValueError as exc:
        ap.error(str(exc))
    scalar = len(out)==1 and args.task not in ('sort','rolling_sum')
    print(json.dumps(float(out[0]) if scalar else out.tolist(),allow_nan=False))


if __name__=='__main__':
    main()
