"""Plain end-to-end SGD with four labeled sequences in a one-state accumulator."""
import csv
from pathlib import Path
import numpy as np
from run import draw
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from error_metrics import rmse


def fit(a,b):
    # Squared loss is 1/2 (prediction-label)^2, all steps have learning rate one.
    # b receives two length-one examples. a does not occur in length-one outputs.
    for _ in range(2):
        b -= b - sum([1.])
    # Adapt x2 to make the derivative dh2/db = a+x2 vanish. Backpropagate through h1.
    for _ in range(2):
        x2 = -a
        h1 = b
        pred = a*h1 + b*x2
        delta = pred - sum([1.,x2])
        da = delta*h1
        db = delta*(a+x2)
        a -= da
        b -= db
    return a,b

if __name__=='__main__':
    out=[]
    rng=np.random.default_rng(6412)
    x=rng.uniform(3,10,(64,512))*rng.choice([-1,1],(64,512))
    target=np.cumsum(x,axis=0)
    for seed in range(256):
        A,B=draw(1,seed)
        a,b=fit(float(A[0,0]),float(B[0]))
        h=np.zeros(x.shape[1])
        correct=np.ones(len(h),dtype=bool)
        prediction = np.empty_like(x)
        for t,u in enumerate(x):
            h=a*h+b*u
            prediction[t] = h
            correct &= h==target[t]
        out.append(dict(seed=seed,a=a,b=b,exact_sequences=int(sum(correct)),
                        rmse=rmse(prediction, target), streams=512, ticks=64))
    with (Path(__file__).parent/'gradient.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=out[0].keys()); wr.writeheader(); wr.writerows(out)
    print(sum(r['exact_sequences']==512 for r in out),len(out))
