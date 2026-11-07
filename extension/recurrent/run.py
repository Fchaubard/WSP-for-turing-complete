"""Passive, dev-selected BPTT versus programmed tanh RNNs and GRUs."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
TASKS = ('sum','fib','tri','sort')
CELLS = ('rnn','gru')
SEEDS = (101,202,303)


def setup():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)


def lengths(task):
    return (2,3,4) if task=='sort' else (4,6,8)


def dataset(task,T,N,domain,seed,canonical=False):
    rng = np.random.default_rng([seed,T,TASKS.index(task),int(domain=='outer')])
    shape = (N,T) if task in ('sum','sort') else (N,2)
    if domain=='outer':
        x = rng.uniform(np.nextafter(3.,10.),10.,shape)*rng.choice([-1.,1.],shape)
    else:
        x = rng.uniform(-3.,3.,shape)
    if task == 'sum':
        x,y = x[:,:,None], np.cumsum(x,axis=1)
    elif task == 'sort':
        y = np.sort(x,axis=1)
        ticks = np.zeros((N,2*T,2))
        ticks[:,:T,0] = x
        ticks[:,T:,1] = 1
        x = ticks
    else:
        if canonical:
            x[:] = [0.,1.]
        y = np.empty((N,T))
        first,second = x[:,0].copy(),x[:,1].copy()
        for t in range(T):
            y[:,t] = first
            if task=='fib':
                first,second = second,first+second
            else:
                first,second = first+second,second+1
    return torch.from_numpy(x),torch.from_numpy(y)


class Network(nn.Module):
    def __init__(self,task,cell,d):
        super().__init__()
        self.task,self.cell,self.d = task,cell,d
        self.encoder = nn.Linear(2,d) if task in ('fib','tri') else None
        cell_type = nn.RNN if cell=='rnn' else nn.GRU
        self.recurrent = cell_type(2 if task=='sort' else 1,d,batch_first=True)
        self.readout = nn.Linear(d,1)

    def reset_xavier(self, seed):
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name,p in self.named_parameters():
                if 'bias' in name:
                    p.zero_()
                elif self.cell=='gru' and name.startswith('recurrent'):
                    for gate in p.chunk(3,dim=0):
                        nn.init.xavier_uniform_(gate,generator=generator)
                else:
                    nn.init.xavier_uniform_(p,generator=generator)

    def forward(self,x,T):
        if self.encoder is not None:
            h = torch.tanh(self.encoder(x))
            ticks = x.new_zeros((len(x),T-1,1))
            tail,_ = self.recurrent(ticks,h.unsqueeze(0))
            states = torch.cat([h[:,None,:],tail],dim=1)
        else:
            states,_ = self.recurrent(x)
            if self.task=='sort':
                states = states[:,T:,:]
        return self.readout(states).squeeze(-1)

    def burn(self):
        """Install a bounded-horizon program. Returns False if no construction."""
        if self.task=='sort' or (self.encoder is not None and self.d<2):
            return False
        gain = 2.**(-52 if self.task=='fib' else -40)
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            rnn = self.recurrent
            if self.cell=='gru':
                # PyTorch GRU gate order is reset, update, candidate. The large
                # logits give exactly r=1 and z=0 under this binary64 kernel.
                rnn.bias_ih_l0[:self.d] = 1000.
                rnn.bias_ih_l0[self.d:2*self.d] = -1000.
                start = 2*self.d
            else:
                start = 0
            A = rnn.weight_hh_l0[start:]
            if self.task=='sum':
                A[0,0] = 1
                rnn.weight_ih_l0[start,0] = gain
            else:
                self.encoder.weight[:2,:] = gain*torch.eye(2)
                if self.task=='fib':
                    A[:2,:2] = torch.tensor([[0.,1.],[1.,1.]])
                else:
                    A[:2,:2] = torch.tensor([[1.,1.],[0.,1.]])
                    rnn.bias_ih_l0[start+1] = gain
            self.readout.weight[0,0] = 1/gain
        return True


def scores(yh,y):
    yh,y = yh.detach().numpy(),y.numpy()
    err = np.abs(yh-y)
    exact = (yh==y)
    one = (yh>=np.nextafter(y,-np.inf)) & (yh<=np.nextafter(y,np.inf))
    scaled = err<=8*np.finfo(float).eps*np.maximum(1.,np.abs(y))
    rmse = np.sqrt(np.mean(err**2))
    return dict(examples=len(y),steps=y.shape[1],exact=float(exact.mean()),
                exact_all=float(exact.all(1).mean()),one_ulp=float(one.mean()),
                one_ulp_all=float(one.all(1).mean()),eps8_all=float(scaled.all(1).mean()),
                rmse=float(rmse),nrmse=float(rmse/np.sqrt(np.mean(y*y))),
                max_abs=float(err.max()),max_scaled=float((err/np.maximum(1.,np.abs(y))).max()))


def evaluate(model):
    task = model.task
    short = lengths(task)[-1]
    long = 8 if task=='sort' else 32
    result = {}
    with torch.no_grad():
        for domain,T,split in [('inner',short,'ID'),('outer',short,'range'),
                               ('inner',long,'length'),('outer',long,'joint')]:
            x,y = dataset(task,T,1024,domain,7777)
            result[split] = scores(model(x,T),y)
        if task in ('fib','tri'):
            x,y = dataset(task,65,1,'inner',0,canonical=True)
            out = model(x,65)
            for i,j in [(0,8),(8,33),(33,65)]:
                result[f'canonical_{i}_{j-1}'] = scores(out[:,i:j],y[:,i:j])
    return result


def parameter_arrays(model):
    return {k:v.detach().cpu().numpy().copy() for k,v in model.named_parameters()}


def restore(model,values):
    with torch.no_grad():
        for key,p in model.named_parameters():
            p.copy_(torch.from_numpy(values[key]))


def fit(model,seed,steps,polish):
    T_values = lengths(model.task)
    train = {t:dataset(model.task,t,512,'inner',3701) for t in T_values}
    dev = {t:dataset(model.task,t,160,'inner',4807) for t in T_values}
    # Estimate per-time RMS from labels on the train bank only. Loss gives each
    # length and time equal weight even when recursive targets grow with time.
    scales = {t:y.square().mean(0).sqrt().clamp_min(1.) for t,(_,y) in train.items()}
    def loss_at(data,t,idx=None):
        x,y = data[t]
        if idx is not None:
            x,y=x[idx],y[idx]
        return ((model(x,t)-y)/scales[t]).square().mean()
    def full(bank):
        return sum(loss_at(bank,t) for t in T_values)/len(T_values)
    with torch.no_grad():
        best = float(full(dev))
    best_state,best_at = parameter_arrays(model),0
    gen = np.random.default_rng(seed+808)
    opt = torch.optim.Adam(model.parameters(),lr=.01)
    for step in range(1,steps+1):
        if step==1+3*steps//4:
            opt.param_groups[0]['lr'] = .003
        t = int(gen.choice(T_values))
        idx = torch.from_numpy(gen.integers(0,512,64))
        opt.zero_grad(set_to_none=True)
        loss_at(train,t,idx).backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if step%50==0 or step==steps:
            with torch.no_grad():
                val = float(full(dev))
            if val<best:
                best,best_state,best_at = val,parameter_arrays(model),step
    restore(model,best_state)
    best_adam = best
    evaluations = 0
    if polish:
        lb = torch.optim.LBFGS(model.parameters(),lr=.7,max_iter=polish,history_size=10,
                               tolerance_grad=1e-13,tolerance_change=1e-16,
                               line_search_fn='strong_wolfe')
        def closure():
            nonlocal evaluations
            evaluations += 1
            lb.zero_grad(set_to_none=True)
            loss = full(train)
            loss.backward()
            return loss
        lb.step(closure)
        with torch.no_grad():
            val = float(full(dev))
        if val < best:
            best,best_state,best_at = val,parameter_arrays(model),'lbfgs'
    restore(model,best_state)
    with torch.no_grad():
        train_loss = float(full(train))
    return dict(best_validation_loss=best,best_adam_validation=best_adam,best_stage=best_at,
                selected_train_loss=train_loss,polish_closures=evaluations)


def one(job):
    task,cell,d,seed,steps,polish,cache = job
    stem = f'{task}_{cell}_{d}_{seed}'
    out = Path(cache)/(stem+'.json')
    weights = Path(cache)/(stem+'.npz')
    if out.exists() and weights.exists():
        return json.loads(out.read_text())
    setup()
    model = Network(task,cell,d)
    model.reset_xavier(seed)
    initial = parameter_arrays(model)
    digest = hashlib.sha256(b''.join(a.tobytes() for a in initial.values())).hexdigest()
    base = dict(task=task,cell=cell,width=d,seed=seed,params=sum(p.size for p in initial.values()),
                initial_sha256=digest)
    arrays = {'init_'+k:v for k,v in initial.items()}
    rows = []
    if seed==SEEDS[0] and model.burn():
        rows.append(dict(base,method='program',metrics=evaluate(model)))
        arrays.update({'program_'+k:v for k,v in parameter_arrays(model).items()})
    restore(model,initial)
    start = time.perf_counter()
    info = fit(model,seed,steps,polish)
    seconds = time.perf_counter()-start
    rows.append(dict(base,method='bp',fit=info,fit_seconds=seconds,metrics=evaluate(model)))
    arrays.update({'bp_'+k:v for k,v in parameter_arrays(model).items()})
    np.savez_compressed(weights,**arrays)
    out.write_text(json.dumps(rows,allow_nan=False))
    print(stem,'joint NRMSE',format(rows[-1]['metrics']['joint']['nrmse'],'.3g'),flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--polish',type=int,default=35)
    ap.add_argument('--cache',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT)
    args = ap.parse_args()
    setup()
    args.output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='recurrent-wsp-') as tmp:
        cache = args.cache or Path(tmp)
        cache.mkdir(parents=True,exist_ok=True)
        # The cache key includes the optimizer budget, preventing stale resumes.
        cache = cache/f's{args.steps}_p{args.polish}'
        cache.mkdir(exist_ok=True)
        jobs = [(task,cell,d,seed,args.steps,args.polish,str(cache))
                for task in TASKS for d in ([1,2,4,8,16] if task=='sort' else [1,2,4,8])
                for cell,seed in itertools.product(CELLS,SEEDS)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            nested = list(ex.map(one,jobs,chunksize=1))
        rows = [r for section in nested for r in section]
        archives = {}
        for task,cell,d,seed,*_ in jobs:
            key = f'{task}_{cell}_{d}_{seed}'
            with np.load(cache/(key+'.npz'),allow_pickle=False) as f:
                archives.update({key+'/'+k:f[k] for k in f.files})
        np.savez_compressed(args.output/'weights.npz',**archives)
    settings = dict(steps=args.steps,polish=args.polish,seeds=SEEDS,examples_per_train_length=512,
                    examples_per_dev_length=160,examples_per_test_split=1024,
                    python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                    machine=platform.platform(),jobs=args.jobs,seconds=time.perf_counter()-start)
    (args.output/'results.json').write_text(json.dumps(dict(settings=settings,rows=rows),indent=1,
                                                allow_nan=False)+'\n')


if __name__=='__main__':
    main()
