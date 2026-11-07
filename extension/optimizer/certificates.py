"""Exact-arithmetic certificates for fixed observed-data losses on several architectures."""
from fractions import Fraction as Q
import numpy as np


def solve(A, b):
    """Small rational linear system by Gaussian elimination."""
    n = len(b)
    M = [list(row) + [yi] for row, yi in zip(A, b)]
    for j in range(n):
        k = next((i for i in range(j,n) if M[i][j]), None)
        if k is None:
            return None
        M[k], M[j] = M[j], M[k]
        d = M[j][j]; M[j] = [v/d for v in M[j]]
        for i in range(n):
            if i != j:
                d=M[i][j]
                M[i] = [a-d*b for a,b in zip(M[i],M[j])]
    return [v[-1] for v in M]


def wls(X, y):
    s=[max(abs(v),1) for v in y]
    Z=[[v/scale for v in row] for row,scale in zip(X,s)]
    u=[v/scale for v,scale in zip(y,s)]
    cols=list(zip(*Z))
    A=[[sum(a*b for a,b in zip(l,r)) for r in cols] for l in cols]
    b=[sum(a*t for a,t in zip(l,u)) for l in cols]
    p=solve(A,b)
    if p is None: return None
    loss=sum((sum(v*a for a,v in zip(p,row))-t)**2 for row,t in zip(Z,u))/(2*len(y))
    return loss,p


def line(x,y):
    return wls([[a,Q(1)] for a in x],y)


def one_hinge(x,y):
    x,y=zip(*sorted(zip(map(Q,x),map(Q,y))))
    linear, coeff = line(x,y)
    candidates=[(linear,[*coeff,Q(1),Q(0),Q(0)],"affine")]
    for t in x[1:-1]:
        loss,(a,b,v)=wls([[u,Q(1),max(Q(0),u-t)] for u in x], y)
        candidates.append((loss,[a,b,Q(1),-t,v],"observed_knot"))
    for k in range(2,len(x)-1):
        cost_left,(a,b)=line(x[:k],y[:k]);cost_right,(c,d)=line(x[k:],y[k:])
        if a==c: continue
        t=(b-d)/(c-a)
        if x[k-1] < t < x[k]:
            loss=(cost_left*k+cost_right*(len(x)-k))/len(x)
            candidates.append((loss,[a,b,Q(1),-t,c-a],"interior_intersection"))
    return min(candidates,key=lambda z:z[0])


def certified(task,size,x,y,smooth):
    """Return real optimum as Q, one feasible binary64 candidate, and certificate kind."""
    if task in smooth and size==1: return one_hinge(x,y)
    if task in smooth and size>=len(x)-2:
        indices=np.argsort(x); xx=np.array(x)[indices]; yy=np.array(y)[indices]
        slopes=np.diff(yy)/np.diff(xx)
        h=len(x)-2; p=np.zeros(2+3*size);p[:2]=[slopes[0], yy[0]-slopes[0]*xx[0]]
        p[2:2+h]=1; p[2+size:2+size+h]=-xx[1:-1];p[2+2*size:2+2*size+h]=np.diff(slopes)
        return Q(0),p,"interpolates_in_real_arithmetic"
    if task in ("add","subtract") or (task=="sum" and size==0):
        xx=(np.reshape(x,(-1,1)) if task=="sum" else x[:,:min(size,2)])
        if size==3: xx=np.column_stack((xx,np.ones(len(x))))
        val,p=wls([[Q(z) for z in row] for row in xx],[Q(z) for z in np.ravel(y)])
        return val,p,"weighted_linear_projection"
    return None
