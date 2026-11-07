"""Finite, label-only weight synthesis using the baseline training examples.

The hypothesis classes are task-specific. No held-out values determine parameter
choice. Sorting also searches discrete wiring, unlike the fixed-topology BP fit.
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from run import Recurrence, Sort, data, metrics, pack, targets, tests

HERE = Path(__file__).resolve().parent


def solve_unique(matrix, rhs):
    """Exact Gaussian elimination. Return None for a singular square system."""
    d = len(rhs)
    rows = [[Fraction(int(v)) for v in row]+[Fraction(int(y))] for row,y in zip(matrix,rhs)]
    for i in range(d):
        pivot = next((j for j in range(i,d) if rows[j][i]),None)
        if pivot is None:
            return None
        rows[i],rows[pivot] = rows[pivot],rows[i]
        s = rows[i][i]
        rows[i] = [a/s for a in rows[i]]
        for j in range(d):
            if j != i:
                s = rows[j][i]
                rows[j] = [a-s*b for a,b in zip(rows[j],rows[i])]
    return [row[-1] for row in rows]


def recurrence_from_labels(y):
    prefix = [int(v) for v in y]
    for d in range(1,len(prefix)//2+1):
        matrix = [prefix[i:i+d] for i in range(d)]
        coeff = solve_unique(matrix,prefix[d:2*d])
        if coeff is not None and all(sum(a*b for a,b in zip(coeff,prefix[i:i+d]))
                                     == prefix[i+d] for i in range(len(prefix)-d)):
            return d,coeff
    raise ValueError('No nonsingular leading Hankel system explains the prefix')


def score(model,p,task):
    return {name:metrics(model.forward(p,x)[0][:,seg],targets(task,x)[:,seg])
            for name,x,seg in tests(task,model)}


def solve_autonomous(task):
    template = np.zeros((1,9))
    labels = targets(task,template)[0]
    d,c = recurrence_from_labels(labels)
    model = Recurrence(task,d)
    p = model.init(101)
    A, h, v = model.parts(p)
    A[:] = np.eye(d,k=1)
    A[-1] = list(map(float,c))
    h[:] = labels[:d]
    v[:] = np.eye(d)[0]
    return {'width':d,'coefficients':list(map(str,c)),'final':p.tolist(),
            'scores':score(model,p,task)}


def solve_sum():
    x = data(256,8,'ID',3330)
    y = targets('sum',x)
    model = Recurrence('sum',1)
    valid = []
    p = model.init(101)
    for coeff in itertools.product([-1.,0.,1.],repeat=3):
        p[:] = coeff
        if np.array_equal(model.forward(p,x)[0],y):
            valid.append(coeff)
    p[:] = valid[0]
    return {'candidates':27, 'solutions':valid, 'final':p.tolist(),
            'scores':score(model,p,'sum')}


class LearnedNetwork(Sort):
    def __init__(self,pairs):
        super().__init__(gates=len(pairs),tied=False)
        self.pairs = pairs

    def forward(self, p, x):
        state = x.copy()
        for (l,r),(q,u,v) in zip(self.pairs,p.reshape(-1,3)):
            h = np.maximum(0,q*(state[:,l]-state[:,r]))
            state[:,l] -= u*h
            state[:,r] += v*h
        return state, None


def solve_sort():
    x = data(256,4,'ID',3330)
    y = targets('sort',x)
    inputs = np.argsort(np.argsort(x,axis=1),axis=1)
    outputs = np.argsort(np.argsort(y,axis=1),axis=1)
    labeled = np.unique(np.concatenate((inputs,outputs),axis=1),axis=0)
    x, y = labeled[:,:4], labeled[:,4:]
    pairs = list(itertools.combinations(range(4),2))
    # Process all networks at each budget. Training traces are integer rankings.
    frontier = [((),x)]
    tested = []
    while frontier:
        solutions = [idx for idx,state in frontier if np.array_equal(state,y)]
        tested.append(len(frontier))
        if solutions:
            break
        assert len(tested) < 7, 'No sorting certificate found by 6 comparisons'
        new = []
        for idx,state in frontier:
            for j,(l,r) in enumerate(pairs):
                child = state.copy()
                child[:,l] = np.minimum(state[:,l],state[:,r])
                child[:,r] = np.maximum(state[:,l],state[:,r])
                new.append((idx+(j,),child))
        frontier = new
    selected = [pairs[j] for j in solutions[0]]
    model = LearnedNetwork(selected)
    initial = model.init(101)
    p = model.burn(initial)
    return {'train_order_types':len(labeled),'candidates_by_depth':tested,
            'solutions_at_first_feasible_depth':len(solutions),'pairs':selected,
            'final':p.tolist(),'scores':score(model,p,'sort')}


def gauge():
    model = Recurrence('sum',1)
    result = {}
    for gain in [.125,.5,1,2,8,1.5,3,np.sqrt(2),np.pi,.1]:
        p = pack(np.array([1]),np.array([gain]),np.array([1/gain]))
        result[str(gain)] = score(model,p,'sum')
    return result


if __name__ == '__main__':
    results = {'sum':solve_sum(),'fib':solve_autonomous('fib'),'tri':solve_autonomous('tri'),
               'sort':solve_sort(),'gauge':gauge()}
    (HERE/'synthesis.json').write_text(json.dumps(results,indent=1,allow_nan=False)+'\n')
    for task,res in results.items():
        if task != 'gauge':
            print(task,{k:v for k,v in res.items() if k!='scores'})
            print({s:round(v['exact_all'],6) for s,v in res['scores'].items()})
        else:
            print('gauge OOD length-32 exact:',
                  {c:round(v['OOD_T32']['exact_all'],6) for c,v in res.items()})
