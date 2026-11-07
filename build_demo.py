"""Reconstruct the ten-task demo checkpoint from the experiment compilers."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scalar'),str(ROOT/'sequence')]
import benchmark as scalar
from compensated import compile_square
from selfpower import grid, center_relative


def dense(arrays,name,layers):
    arrays[name+'_shapes'] = np.array([w.shape for w,_ in layers],dtype=np.int16)
    arrays[name+'_values'] = np.concatenate([a.ravel() for layer in layers for a in layer])


def main():
    arrays = {}
    dense(arrays,'square',compile_square(48))
    x = scalar.data('sin')['train']
    fold = scalar.initial_hinge(6,11)
    fold[:2] = [-1,-3*np.pi]
    fold[2:] = scalar.pack([np.ones(6),-(np.arange(-3,3)+.5)*np.pi,2*(-1.)**np.arange(6)])
    z = scalar.hinge(x,fold)
    knots = np.linspace(-np.pi/2,np.pi/2,65)[1:-1]
    design = np.column_stack([z,np.ones(len(z)),np.maximum(0,z[:,None]-knots)])
    theta = np.linalg.lstsq(design,np.sin(x),rcond=None)[0]
    arrays['sin_fold'] = fold
    arrays['sin_spline'] = scalar.pack([theta[:2],np.ones(63),-knots,theta[2:]])
    knots = grid(63,'relative')
    y = np.power(knots,knots)
    slopes = np.diff(y)/np.diff(knots)
    scale = center_relative(knots,y,slopes)
    arrays['selfpower_spline'] = scalar.pack([np.array([scale*slopes[0],scale*y[0]]),
                                              np.ones(63),-knots[1:-1],scale*np.diff(slopes)])
    # The signed-integer-only circuit is the exact full-spec lookup, loaded
    # from the archived circuit outputs. Its positive branch is optional.
    with (ROOT/'scalar'/'signed_selfpower.csv').open() as f:
        entries = [row for row in csv.DictReader(f) if row['method']=='full_spec']
    ordinates = np.array([float.fromhex(row['output_hex']) for row in entries])
    w2 = np.zeros((23,21))
    idx = np.arange(21)
    for off,c in enumerate([1.,-2.,1.]):
        w2[idx+off,idx] = c
    dense(arrays,'integer_power',[(np.ones((1,23)),-np.arange(-11,12,dtype=float)),
                                 (w2,np.zeros(21)),(ordinates[:,None],np.zeros(1))])
    arrays['tanh'] = np.array([1.,0.,1.,0.])
    arrays['add'],arrays['subtract'] = np.array([1.,1.]),np.array([1.,-1.])
    rows = json.loads((ROOT/'sequence'/'results.json').read_text())['results']
    arrays['rolling_sum'] = np.array(next(row['final'] for row in rows if
        (row['task'],row['method'],row['size']) == ('sum','program','1')))
    signed = json.loads((ROOT/'sequence'/'signed_indices.json').read_text())
    for task,key in [('fibonacci','fib'),('triangular','tri')]:
        for parameter,value in zip(['forward','backward','initial','readout'],
                                   signed[key]['direct']['final']):
            arrays[f'{task}_{parameter}'] = np.array(value)
    graph = json.loads((ROOT/'sequence'/'robust_sort.json').read_text())['graph']
    arrays['sort_ptr'] = np.cumsum([0]+[len(n['edges']) for n in graph],dtype=np.int16)
    arrays['sort_src'] = np.array([i for n in graph for i,_ in n['edges']],dtype=np.int16)
    arrays['sort_coef'] = np.array([c for n in graph for _,c in n['edges']])
    arrays['sort_bias'] = np.array([n['bias'] for n in graph])
    arrays['sort_relu'] = np.array([n['relu'] for n in graph])
    path = ROOT/'circuits.npz'
    np.savez_compressed(path,**arrays)
    tasks = {
        'square':'48 compensated dyadic-tent stages, 9,779 dense scalar slots',
        'sin':'6 phase-folding hinges, 63 readout hinges fit on central labels',
        'tanh':'one native NumPy tanh neuron with identity affine coefficients',
        'xpowx':'63 relative-curvature hinges, full-envelope balanced relative readout; negative integers use the saved 572-slot signed-grid lookup',
        'add':'two-weight linear readout', 'subtract':'two-weight linear readout',
        'rolling_sum':'one-state linear recurrent prefix sum',
        'sort':'50-slot, 13-ReLU binary64 comparator shared across a Batcher schedule',
        'fibonacci':'signed unit clock, two-state reversible linear recurrence',
        'triangular':'signed unit clock, three-state reversible linear recurrence'}
    metadata = {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'dtype':'binary64, except integer graph indices and shapes',
                'tasks':tasks, 'arrays':{k:list(v.shape) for k,v in arrays.items()}}
    (ROOT/'circuits.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Built {path.name} ({path.stat().st_size:,} bytes).')


if __name__=='__main__':
    main()
