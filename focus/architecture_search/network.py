"""Safe binary64 neural graph execution with explicit scalar-operation and liveness costs."""
from collections import Counter
from dataclasses import dataclass
import json
import math

import numpy as np

MAX_NODES=64
MAX_COEFFICIENTS=4096
MAX_LIVE=65536
UNARY={'relu','tanh','sigmoid','reshape','take','transpose','softmax','affine','conv1d'}
BINARY={'add','multiply','matmul'}


@dataclass
class Cost:
    arithmetic: int=0
    comparisons: int=0
    transcendental: int=0
    peak_scalars: int=0

    def absorb(self,other,extra=0):
        self.arithmetic+=other.arithmetic
        self.comparisons+=other.comparisons
        self.transcendental+=other.transcendental
        self.peak_scalars=max(self.peak_scalars,other.peak_scalars+extra)


def numeric_array(value):
    def valid(x):
        if type(x) in (int,float):return math.isfinite(x)
        return isinstance(x,list) and len(x)>0 and all(valid(v) for v in x)
    if not valid(value):raise ValueError('Coefficient values must be finite numeric arrays')
    result=np.asarray(value,dtype=np.float64)
    if result.ndim not in (1,2,3) or result.size>MAX_COEFFICIENTS:raise ValueError('Coefficient array shape exceeds limits')
    return result


def sources(node):
    op=node['op']
    if op=='constant':return []
    if op in UNARY:return [node['source']]
    if op in BINARY:return [node['left'],node['right']]
    if op=='concat':return node['sources']
    raise ValueError('Unsupported operator '+str(op))


def checked_ints(xs,positive=False):
    if not isinstance(xs,list) or not xs or any(type(v)!=int or (positive and v<=0) for v in xs):
        raise ValueError('Expected static integer list')
    return xs


class Graph:
    def __init__(self,nodes,output,external):
        if not isinstance(nodes,list) or len(nodes)>MAX_NODES or not isinstance(output,str):raise ValueError('Malformed graph')
        self.nodes=[];self.coefficients=0;self.output=output
        known=set(external)
        for raw in nodes:
            if not isinstance(raw,dict):raise ValueError('Malformed node')
            n=dict(raw);name=n.get('name');op=n.get('op')
            if not isinstance(name,str) or name in known:raise ValueError('Node names must be unique and distinct from external inputs')
            refs=sources(n)
            if not isinstance(refs,list) or any(not isinstance(r,str) or r not in known for r in refs):raise ValueError('Nonexistent or forward node reference')
            fields={'name','op'}
            if op=='constant':fields|={'value'}
            elif op in UNARY:fields|={'source'}
            elif op in BINARY:fields|={'left','right'}
            elif op=='concat':fields|={'sources'}
            if op in ('affine','conv1d'):fields|={'weight','bias'}
            if op=='take':fields|={'indices'};checked_ints(n['indices'])
            if op=='reshape':fields|={'shape'};checked_ints(n['shape'],True)
            if set(n)-fields:raise ValueError('Unknown operator fields')
            for field in ('value','weight','bias'):
                if field in n:
                    n[field]=numeric_array(n[field]);self.coefficients+=n[field].size
            if op=='affine' and n['weight'].ndim!=2:raise ValueError('Affine weight must be matrix')
            if op=='conv1d' and n['weight'].ndim!=3:raise ValueError('Convolution weight must have rank three')
            self.nodes.append(n);known.add(name)
        if output not in known:raise ValueError('Output reference missing')
        self.uses=Counter(r for n in self.nodes for r in sources(n));self.uses[output]+=1

    def run(self,external):
        values={k:np.asarray(v,dtype=np.float64) for k,v in external.items()}
        uses=self.uses.copy();cost=Cost(peak_scalars=sum(v.size for v in values.values()))
        for n in self.nodes:
            op=n['op'];refs=sources(n);args=[values[r] for r in refs]
            if op=='constant':result=n['value']
            elif op=='affine':
                w=n['weight'];x=args[0].reshape(-1)
                if w.shape[1]!=len(x):raise ValueError('Affine shape mismatch')
                result=w@x;cost.arithmetic+=w.size+w.shape[0]*max(0,w.shape[1]-1)
                if 'bias' in n:
                    if n['bias'].shape!=(w.shape[0],):raise ValueError('Bias shape mismatch')
                    result=result+n['bias'];cost.arithmetic+=result.size
            elif op=='relu':result=np.maximum(args[0],0);cost.comparisons+=result.size
            elif op=='tanh':result=np.tanh(args[0]);cost.transcendental+=result.size
            elif op=='sigmoid':result=1/(1+np.exp(-args[0]));cost.transcendental+=result.size;cost.arithmetic+=3*result.size
            elif op=='add':result=args[0]+args[1];cost.arithmetic+=result.size
            elif op=='multiply':result=args[0]*args[1];cost.arithmetic+=result.size
            elif op=='matmul':
                a,b=args
                if a.ndim not in (1,2) or b.ndim not in (1,2):raise ValueError('Matmul accepts vectors or matrices')
                result=a@b;inner=a.shape[-1]
                cost.arithmetic+=np.asarray(result).size*(inner+max(0,inner-1))
            elif op=='concat':result=np.concatenate([a.reshape(-1) for a in args])
            elif op=='take':result=args[0].reshape(-1)[n['indices']]
            elif op=='reshape':result=args[0].reshape(n['shape'])
            elif op=='transpose':
                if args[0].ndim!=2:raise ValueError('Transpose needs matrix')
                result=args[0].T
            elif op=='softmax':
                x=args[0];width=x.shape[-1];rows=x.size//width
                p=np.exp(x-np.max(x,axis=-1,keepdims=True));result=p/np.sum(p,axis=-1,keepdims=True)
                cost.comparisons+=rows*max(0,width-1);cost.transcendental+=x.size
                cost.arithmetic+=2*x.size+rows*max(0,width-1)
            elif op=='conv1d':
                x=args[0];w=n['weight'];out_channels,in_channels,kernel=w.shape
                if x.ndim!=2 or x.shape[1]!=in_channels or len(x)<kernel:raise ValueError('Convolution shape mismatch')
                length=len(x)-kernel+1;result=np.empty((length,out_channels))
                for i in range(length):result[i]=np.sum(w*x[i:i+kernel].T[None,:,:],axis=(1,2))
                terms=in_channels*kernel
                cost.arithmetic+=result.size*(terms+max(0,terms-1))
                if 'bias' in n:
                    if n['bias'].shape!=(out_channels,):raise ValueError('Convolution bias shape mismatch')
                    result=result+n['bias'];cost.arithmetic+=result.size
            else:raise ValueError('Unknown node')
            result=np.array(result,dtype=np.float64,copy=True)
            if result.ndim==0:result=result.reshape(1)
            if result.size==0 or result.size>MAX_LIVE:raise ValueError('Activation size exceeds limits')
            values[n['name']]=result
            cost.peak_scalars=max(cost.peak_scalars,sum(v.size for v in values.values()))
            if cost.peak_scalars>MAX_LIVE:raise ValueError('Live activation limit exceeded')
            for ref in refs:
                uses[ref]-=1
                if uses[ref]==0:del values[ref]
            if uses[n['name']]==0:del values[n['name']]
        return values[self.output],cost


class Network:
    def __init__(self,reply,mode):
        if not isinstance(reply,dict) or not isinstance(reply.get('network'),dict):raise ValueError('Expected response with network')
        self.reply=reply;self.spec=reply['network'];self.mode=mode;self.kind=self.spec.get('kind')
        self.nodes=0
        if self.kind=='graph':
            if set(self.spec)!={'kind','nodes','output'}:raise ValueError('Unknown network fields')
            self.graph=Graph(self.spec['nodes'],self.spec['output'],['input'])
            self.coefficients=self.graph.coefficients;self.nodes=len(self.graph.nodes)
        elif self.kind=='recurrent':
            if set(self.spec)!={'kind','initial_state','cell','state_output','readout','output'}:raise ValueError('Unknown recurrent fields')
            if mode not in ('index','sequence'):raise ValueError('Recurrence needs index or sequence input')
            self.initial=numeric_array(self.spec['initial_state'])
            if self.initial.ndim!=1:raise ValueError('State must be a vector')
            self.cell=Graph(self.spec['cell'],self.spec['state_output'],['state','input'])
            self.readout=Graph(self.spec['readout'],self.spec['output'],['state','input'])
            self.coefficients=self.cell.coefficients+self.readout.coefficients+self.initial.size
            self.nodes=len(self.cell.nodes)+len(self.readout.nodes)
        else:raise ValueError('Unsupported network kind')
        if self.nodes>MAX_NODES or self.coefficients>MAX_COEFFICIENTS:raise ValueError('Model budget exceeded')
        self.serialized_bytes=len(json.dumps(self.spec,separators=(',',':'),ensure_ascii=True).encode())

    def one(self,value):
        x=np.atleast_1d(np.asarray(value,dtype=float))
        with np.errstate(all='ignore'):
            if self.kind=='graph':return self.graph.run({'input':x})
            h=self.initial.copy();cost=Cost(peak_scalars=h.size+x.size)
            if self.mode=='index':
                n=int(value)
                if n!=value or not 0<=n<=64:raise ValueError('Unsupported index')
                inputs=[np.array([0.0])]*n
                for tick in inputs:
                    h,c=self.cell.run({'state':h,'input':tick})
                    if h.shape!=self.initial.shape:raise ValueError('Cell changed state shape')
                    cost.absorb(c,x.size)
                out,c=self.readout.run({'state':h,'input':np.array([0.0])})
                cost.absorb(c,x.size)
                return out,cost
            outputs=[]
            # The complete input and emitted output buffers count as live inference storage.
            for tick in x:
                u=np.atleast_1d(tick)
                h,c=self.cell.run({'state':h,'input':u})
                if h.shape!=self.initial.shape:raise ValueError('Cell changed state shape')
                cost.absorb(c,x.size+len(outputs))
                out,c=self.readout.run({'state':h,'input':u})
                if out.size!=1:raise ValueError('Stream readout must be scalar')
                cost.absorb(c,x.size+len(outputs));outputs.append(float(out[0]))
            cost.peak_scalars=max(cost.peak_scalars,x.size+len(outputs)+h.size)
            return np.asarray(outputs),cost

    def predict(self,x,y_shape):
        outputs=[];costs=[]
        for value in x:
            result,cost=self.one(value)
            required = int(np.prod(y_shape, dtype=int)) if y_shape else 1
            if result.size != required:
                raise ValueError('Wrong output shape')
            outputs.append(result.reshape(y_shape));costs.append(cost)
        return np.asarray(outputs),costs
