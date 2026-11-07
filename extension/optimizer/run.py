"""Paired solver diagnostics on the exact ninety writer fixtures; no shell-based selection."""
import csv
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.optimize import least_squares,minimize

HERE=Path(__file__).resolve().parent
WRITER=HERE.parent/'writer'
sys.path.insert(0,str(WRITER))
from design import TASKS,SIZES,SEEDS,SMOOTH,SERIES,observed,initial,predict,reference,inputs
from compare import objective, parse
from certificates import certified


def jacobian(task,d,p,x):
    n=len(x); P=len(p)
    if task in SMOOTH:
        w,c,v=p[2:].reshape(3,d);z=x[:,None]*w+c;active=z>0
        return np.column_stack((x,np.ones(n),active*v*x[:,None],active*v,np.maximum(0,z)))
    if task in ('add','subtract'):
        J=x[:,:min(2,d)]
        return np.column_stack((J,np.ones(n))) if d==3 else J
    if task=='sort':
        h=x.copy();dh=np.zeros((n,4,P))
        for j,(w,v) in enumerate(p.reshape(d,2,4)):
            raw=h@w; g=np.maximum(raw,0.)
            dg=np.einsum('i,nip->np',w,dh)
            dg[:,8*j:8*j+4]+=h; dg*= (raw>0)[:,None]
            dh+=v[None,:,None]*dg[:,None,:]
            for k in range(4):dh[:,k,8*j+4+k]+=g
            h=h+g[:,None]*v
        return dh.reshape(-1,P)
    if d==0:return x.reshape(-1,1)
    A=p[:d*d].reshape(d,d);b=p[d*d:d*d+d];v=p[-d:]
    if task in SERIES:
        xx=np.zeros((1,max(x)));h=b[None,:].copy();h0=True
    else:
        xx=x;h=np.zeros((n,d));h0=False
    dh=np.zeros((len(h),d,P));rows=[]
    def readout():
        out=np.einsum('j,njp->np',v,dh)
        out[:,-d:]+=h
        return out
    if h0:
        for i in range(d):dh[:,i,d*d+i]=1.
        rows.append(readout())
    for u in xx.T:
        upd=np.einsum('ij,njp->nip',A,dh)
        for i in range(d):
            upd[:,i,i*d:(i+1)*d]+=h
            if not h0:upd[:,i,d*d+i]+=u
        h=h@A.T+u[:,None]*b
        dh=upd;rows.append(readout())
    arr=np.stack(rows,axis=1)
    return arr[0,x] if h0 else arr.reshape(-1,P)


def methods(task,d,seed,x,y,p0,archived,proposals,weights):
    fun=objective(task,d,p0,x,y)
    runs={}
    def insert(name,p,fev=0,jev=0,gn=0,reason=''):
        runs[name]=(p.copy(),int(fev),int(jev),int(gn),str(reason))
    insert('xavier',p0)
    for name,val in proposals.items():
        if val is not None:insert(name,val)
    insert('bp_archived',archived)
    # Replicate exactly the archived 800-step Adam -> BFGS schedule, preserving the intermediate.
    p=p0.copy();m=np.zeros_like(p);v=np.zeros_like(p)
    for t in range(1,801):
        _,g=fun(p);g*=min(1.,5./max(np.linalg.norm(g),1e-250))
        m=.9*m+.1*g; v=.999*v+.001*g**2
        p-=.02*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    insert('adam800',p,800,800)
    bfgs=minimize(fun,p,jac=True,method='BFGS',options={'maxiter':1000,'gtol':1e-14})
    pb=bfgs.x if bfgs.fun <= fun(p)[0] else p
    insert('adam_bfgs',pb,800+bfgs.nfev,800+bfgs.njev,reason=bfgs.message)
    s=np.maximum(1,np.abs(y)).ravel()
    def res(p):return (predict(task,d,p,x)-y).ravel()/s
    def jac(p):return jacobian(task,d,p,x)/s[:,None]
    # Check full forward-sensitivity Jacobian against independent reverse mode and a centered JVP.
    r=res(p0);J=jac(p0)
    assert np.max(np.abs(J.T@r/len(r)-fun(p0)[1])/np.maximum(1,np.abs(fun(p0)[1])))<1e-8
    direction=np.random.default_rng(seed+1201).normal(size=p.shape);direction/=np.linalg.norm(direction)
    approx=(res(p0+1e-6*direction)-res(p0-1e-6*direction))/(2e-6)
    assert np.max(np.abs(J@direction-approx)/np.maximum(1,np.abs(approx)))<2e-6,(task,d)
    for start in ('xavier','adam_bfgs','examples','formula'):
        if start not in runs:continue
        start_p=runs[start][0];p=start_p.copy()
        cnt={'f':0,'j':0}
        def counted(p):
            cnt['f']+=1
            return res(p)
        def counted_jac(p):
            cnt['j']+=1
            return jac(p)
        r=counted(p);cost=r@r;iterations=0;reason='initial zero'
        if cost>0:
            fit=least_squares(counted,p,jac=counted_jac,method='trf',max_nfev=250,
                              xtol=4e-15,ftol=4e-15,gtol=4e-15)
            if fit.fun@fit.fun <= cost: p,r,cost=fit.x,fit.fun,fit.fun@fit.fun
            reason=fit.message
            # Preserve the best measured loss. A final Gauss--Newton search addresses tiny residuals.
            for _ in range(10):
                if cost==0:break
                j=counted_jac(p)
                delta=np.linalg.lstsq(j,-r,rcond=1e-12)[0]
                oldcost=cost
                for alpha in [1.,.5,.25,.125,.0625,.03125,.015625]:
                    cand=p+alpha*delta
                    if np.array_equal(cand,p):continue
                    rnew=counted(cand);cnew=rnew@rnew
                    if cnew < cost:
                        p,r,cost=cand,rnew,cnew
                        break
                if cost>=oldcost:break
                iterations+=1
        insert('ls_'+start,p,cnt['f'],cnt['j'],iterations,reason=reason)
    cert=certified(task,d,x,y,SMOOTH)
    if cert is not None:
        val,cp,kind=cert
        insert('certificate_write',np.asarray(cp,dtype=float))
        cert_record=dict(real_loss=float(val),kind=kind)
    else:cert_record={}
    for name,(p,*_) in runs.items():
        weights[f'{task}_{d}_{seed}_{name}']=p
    return runs,cert_record


def score(p,task,size,x,y):
    with np.errstate(over='ignore',invalid='ignore'):
        yh=predict(task,size,p,x); diff=yh-y; e=np.abs(diff)/np.maximum(1,np.abs(y))
        eq=yh==y
        if eq.ndim==2:eq=eq.all(axis=1)
        return dict(exact=float(np.mean(eq)),mae=float(np.mean(np.abs(diff))),
                    mean_scaled=float(np.mean(e)),half_mse=float(.5*np.mean(e**2)))


def main():
    props={}
    invalid=[]
    for case in json.loads((WRITER/'manifest.json').read_text()):
        vals=parse((WRITER/'trials'/(case['id']+'.reply.txt')).read_text())
        for k,d in enumerate(case['sizes']):
            task,seed=case['task'],case['seed'];p0=initial(task,d,seed);arm=case['arm']
            try:
                p=np.asarray(vals[k],dtype=float)
                good=p.shape==p0.shape and np.isfinite(p).all()
            except (TypeError,ValueError,IndexError):good=False
            if not good:invalid.append((task,seed,d,arm))
            props[(task,seed,d,arm)]=p if good else None
    bases={(r['task'],r['seed'],r['size']):np.asarray(r['weights']) for r in json.loads((WRITER/'bp.json').read_text())}
    rows=[];weights={}
    for task in TASKS:
        dataset=inputs(task)
        dev_name='inner' if 'inner' in dataset else 'early'
        devx=dataset[dev_name];devy=reference(task,devx)
        ox=dataset['outer'];oy=reference(task,ox)
        for seed in SEEDS:
            x,y=observed(task,seed)
            for d in SIZES[task]:
                p0=initial(task,d,seed)
                pp={a:props[(task,seed,d,a)] for a in ['examples','formula']}
                runs,cert=methods(task,d,seed,x,y,p0,bases[(task,seed,d)],pp,weights)
                cm=[]
                for name,(p,fev,jev,polish,msg) in runs.items():
                    tr=score(p,task,d,x,y);de=score(p,task,d,devx,devy);outer=score(p,task,d,ox,oy)
                    cm.append(dict(task=task,size=d,seed=seed,method=name,n=len(x),params=len(p),
                                 train_loss=tr['half_mse'],train_exact=tr['exact'],
                                 dev_split=dev_name,dev_exact=de['exact'],dev_scaled=de['mean_scaled'],
                                 outer_exact=outer['exact'],outer_mae=outer['mae'],outer_scaled=outer['mean_scaled'],
                                 nfev=fev,njev=jev,polish_steps=polish,termination=msg,
                                 real_optimum=cert.get('real_loss'),certificate=cert.get('kind','')))
                best=min(v['train_loss'] for v in cm)
                for row in cm:
                    row['best_measured']=best;row['gap_to_best']=row['train_loss']-best
                rows.extend(cm)
        print(task,len(rows),'rows',flush=True)
        with (HERE/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        np.savez_compressed(HERE/'states.npz',**weights)
    (HERE/'invalid.json').write_text(json.dumps(invalid)+'\n')


if __name__=='__main__':
    main()
