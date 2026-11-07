"""Fix scalar hidden features and solve the precise training objective globally."""
from pathlib import Path
import csv, json, time
import numpy as np
from scipy.linalg import svd
HERE=Path(__file__).parent
TARGETS={'square':lambda x:x*x,'sin':np.sin,'selfpower':lambda x:x**x,'tanh':np.tanh}
def data(r,lo,hi,n):return r.uniform(lo,hi,n)
def run():
    rows=[];start=time.monotonic()
    for task,f in TARGETS.items():
        pos=task=='selfpower'
        for seed in [164,317,583]:
            rng=np.random.default_rng(66113+seed)
            sets={'central_train':data(rng,0 if pos else -3,3,1024),'envelope_train':data(rng,0 if pos else -10,10,1024),
                  'inner':data(rng,0 if pos else -3,3,4096),'outer':data(rng,3,10,4096)}
            if not pos:sets['outer']=np.r_[-data(rng,3,10,4096),sets['outer']]
            for width in [0,1,3,7,15,31,63]:
                for basis in ['xavier','central_knots','envelope_knots']:
                    r=np.random.default_rng(2204+seed+width)
                    w=r.uniform(-np.sqrt(6/(width+1)),np.sqrt(6/(width+1)),width);b=np.zeros(width)
                    head=np.r_[r.uniform(-np.sqrt(3),np.sqrt(3)),0.,r.uniform(-np.sqrt(6/(width+1)),np.sqrt(6/(width+1)),width)]
                    if basis!='xavier':
                        w[:]=1;extent=3 if basis=='central_knots' else 10
                        b=-np.linspace(0 if pos else -extent,extent,width+2)[1:-1]
                    X={k:np.column_stack((x,np.ones(len(x)),np.maximum(x[:,None]*w+b,0.))) for k,x in sets.items()}
                    for split in ['central_train','envelope_train']:
                        y=f(sets[split]);scale=np.maximum(1,np.abs(y));M=X[split]/scale[:,None]; labels=y/scale
                        u,s,vt=svd(M,full_matrices=False,lapack_driver='gesdd',check_finite=False)
                        good=s>s[0]*(max(M.shape)*np.finfo(float).eps)
                        # Retain the original head in the nullspace (minimum change from start).
                        residual=M@head-labels
                        direct=head-vt[good].T@((u[:,good].T@residual)/s[good])
                        # One preconditioned backprop step; the SVD preconditioner uses input features and training loss weights.
                        jac_back=M.T@residual
                        prec=head-vt[good].T@((vt[good]@jac_back)/s[good]**2)
                        # Plain full-batch GD at 1/L, evaluated via its exact spectral iteration on the Gram matrix.
                        # Stable expression for 5000 analytical updates; this is a trajectory calculation, not 5000 timed matmuls.
                        progress=-np.expm1(5000*np.log1p(-np.minimum(.9999999999999999,(s[good]/s[0])**2)))
                        gd=head-vt[good].T@(progress*(u[:,good].T@residual)/s[good])
                        for method,h in [('xavier_head',head),('gd_5000_spectral',gd),('preconditioned_bp',prec),('svd',direct)]:
                            rv=M@h-labels; loss=.5*float(np.mean(rv**2))
                            row=dict(task=task,seed=seed,width=width,hidden=basis,training=split,method=method,
                                feature_rank=int(good.sum()),outputs=width+2,condition=float(s[0]/s[good][-1]),loss=loss,
                                normal_residual=float(np.max(np.abs(M.T@rv))/len(y)))
                            for k in ['inner','outer']:
                                yh=X[k]@h;yt=f(sets[k]);err=np.abs(yh-yt)/np.maximum(1,np.abs(yt))
                                row[k+'_exact']=float(np.mean(yh==yt));row[k+'_scaled']=float(err.mean())
                                row[k+'_rmse']=float(np.sqrt(np.mean((yh-yt)**2)))
                            rows.append(row)
    with (HERE/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(len(rows),round(time.monotonic()-start,2),'seconds')
if __name__=='__main__':run()
