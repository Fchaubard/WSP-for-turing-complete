"""Compile and audit a binary64-exact comparator as an affine/ReLU DAG.

An edge is a pair (source-node index, coefficient). An evaluator executes the
listed multiply/add order with separate float64 ufuncs, round-to-nearest and
gradual underflow. Fixed negative and positive unit edges are counted.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from error_metrics import rmse

import numpy as np

from run import Sort, data, metrics, odd_even_network, targets, xavier

HERE = Path(__file__).resolve().parent


def template():
    nodes = []
    def emit(edges, bias=0., relu=False):
        nodes.append((edges,bias,relu))
        return 1+len(nodes)  # a and b occupy slots 0 and 1
    # Slots for K and M are shared across all applications of the comparison.
    a,b = 0,1
    r = emit([(a,1.),(b,-1.)],relu=True)
    u = emit([(r,'K')])
    excess = emit([(u,1.)],bias=-1.,relu=True)
    v = emit([(u,1.),(excess,-1.)])
    missed_big = emit([(u,-1.)],bias=1.,relu=True)
    big = emit([(missed_big,-1.)],bias=1.)
    missed_small = emit([(v,'-K')],bias=1.,relu=True)
    small = emit([(missed_small,-1.)],bias=1.)
    keep = emit([(big,-1.),(small,-1.)],bias=1.,relu=True)
    swap = emit([(keep,-1.)],bias=1.)
    shift_swap = emit([(swap,'M')])
    shift_keep = emit([(keep,'M')])
    def gate(x, off_shift):
        pos = emit([(x,1.),(off_shift,-1.)],relu=True)
        neg = emit([(x,-1.),(off_shift,-1.)],relu=True)
        return emit([(pos,1.),(neg,-1.)])
    a_low = gate(a,shift_swap)
    b_low = gate(b,shift_keep)
    a_high = gate(a,shift_keep)
    b_high = gate(b,shift_swap)
    emit([(a_low,1.),(b_low,1.)])
    emit([(a_high,1.),(b_high,1.)])
    return nodes


def compile_numeric(nodes, seed=101):
    rng = np.random.default_rng(seed)
    # Xavier uniform on each affine node's input row, structural bias slots at 0.
    slots = []
    for edges,bias,_ in nodes:
        slots.extend(xavier(rng,len(edges),1)[0])
        if bias != 0:
            slots.append(0.)
    slots = np.asarray(slots)
    initial = {'seed':seed,'values':slots.tolist(),
               'sha256':hashlib.sha256(slots.tobytes()).hexdigest()}
    lookup = {'K':2.**537,'-K':-(2.**537),'M':10.}
    pos,graph = 0,[]
    for edges,bias,relu in nodes:
        row = []
        for source,coef in edges:
            slots[pos] = lookup[coef] if isinstance(coef,str) else coef
            row.append((source,slots[pos]))
            pos += 1
        if bias != 0:
            slots[pos] = bias
            bias = slots[pos]
            pos += 1
        graph.append((row,bias,relu))
    assert pos == len(slots)
    return graph,initial,{'values':slots.tolist(),'K':lookup['K'],'M':lookup['M']}


def evaluate(graph,a,b):
    values = [a,b]
    for edges,bias,relu in graph:
        z = bias
        for src, coeff in edges:
            z = np.add(z,np.multiply(values[src],coeff))
        values.append(np.maximum(0.,z) if relu else z)
    return values[-2], values[-1]


def sort(graph,x):
    state = x.copy()
    for i,j in odd_even_network(x.shape[1]):
        lo,hi = evaluate(graph,state[:,i],state[:,j])
        state[:,i],state[:,j] = lo,hi
    return state


def pair_checks(graph):
    tiny = np.nextafter(0.,1.)
    assert tiny == float.fromhex('0x0.0000000000001p-1022')
    assert tiny * 2.**537 * 2.**537 == 1.
    # Positive/negative subnormals, adjacent binade boundaries, signed zero,
    # opposite signs, and a range of separations around both clamp transitions.
    pos = [tiny,2*tiny,3*tiny,np.finfo(float).tiny,10.]
    exps = [-1073,-1060,-1022,-1000,-539,-538,-537,-536,-535,-485,-484,-483,
            -54,-53,-52,-51,-1,0,1,2,3]
    for k in exps:
        val = 2.**k
        pos += [np.nextafter(val,0.),val,np.nextafter(val,np.inf)]
    pos = np.unique(pos)
    lattice = np.concatenate((pos,-pos,[0.,-0.]))
    aa,bb = np.meshgrid(lattice,lattice)
    edge_a,edge_b = aa.flatten(),bb.flatten()
    rng = np.random.default_rng(64001)
    raw = rng.bit_generator.random_raw(1_000_000).view(np.float64)
    with np.errstate(invalid='ignore'):
        raw = raw[np.isfinite(raw) & (np.abs(raw) <= 10.)]
    a,b = np.split(raw[:2*(raw.size//2)],2)
    # Adjacent values in both orders throughout the range, including tiny gaps.
    up = np.nextafter(a,np.inf)
    ok = up <= 10.
    probes = {'edge_cartesian':(edge_a,edge_b),'random_raw_bits':(a,b),
              'adjacent_ascending':(a[ok],up[ok]),'adjacent_descending':(up[ok],a[ok])}
    result = {}
    for name,(left,right) in probes.items():
        lo,hi = evaluate(graph,left,right)
        want_lo, want_hi = np.minimum(left,right),np.maximum(left,right)
        assert np.all(lo == want_lo) and np.all(hi == want_hi), name
        for pred,want in ((lo,want_lo),(hi,want_hi)):
            nz = want != 0
            assert np.array_equal(pred[nz].view('uint64'),want[nz].view('uint64'))
        result[name] = {'pairs':len(left),'numeric_mismatches':0,'nonzero_bit_mismatches':0,
                        'rmse':rmse([lo,hi], [want_lo,want_hi])}
    return result


def main():
    raw = template()
    graph,initial,final = compile_numeric(raw)
    cost = dict(nodes=len(raw),relu=sum(relu for _,_,relu in raw),
                edge_slots=sum(len(e) for e,_,_ in raw),
                nonzero_bias_slots=sum(b!=0 for _,b,_ in raw),tied_scale_slots=2)
    assert cost == dict(nodes=26,relu=13,edge_slots=43,nonzero_bias_slots=7,tied_scale_slots=2)
    summary = {}
    for t in [4,8]:
        for domain in ['ID','OOD']:
            for dyadic in [False, True]:
                x = data(2048,t,domain,7001+t+(domain=='OOD')+101*dyadic,dyadic)
                yh = sort(graph,x)
                y = targets('sort',x)
                stats = metrics(yh,y)
                assert stats['bit_fraction'] == 1.
                summary[f'{domain}_T{t}'+('_dyadic' if dyadic else '')] = stats
    results = dict(cost=cost,initial=initial,final=final,scores=summary,pairs=pair_checks(graph),
                   graph=[dict(edges=e,bias=b,relu=r) for e,b,r in graph])
    (HERE/'robust_sort.json').write_text(json.dumps(results,indent=1,allow_nan=False)+'\n')
    print(cost,results['pairs'])


if __name__ == '__main__':
    main()
