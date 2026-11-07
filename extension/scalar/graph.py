"""Sparse scalar arithmetic DAG: parameter leaves, literals, +, -, *, and ReLU."""
import numpy as np

class Wire:
    def __init__(self, graph, i):
        self.graph, self.i = graph, i

    def op(self, name, other):
        g = self.graph
        b = other if isinstance(other, Wire) else g.literal(other)
        return g.node(name, self.i, b.i)

    def __add__(self,b): return self.op("add",b)
    def __sub__(self,b): return self.op("sub",b)
    def __mul__(self,b): return self.op("mul",b)
    def relu(self): return self.graph.node("relu",self.i)

class Graph:
    def __init__(self):
        self.nodes = [("x",)]
        self.params = []
        self.written = []
        self.literals = {}
        self.x = Wire(self,0)

    def node(self,*op):
        self.nodes.append(op)
        return Wire(self,len(self.nodes)-1)

    def literal(self,v):
        v = float(v)
        if v not in self.literals:
            self.literals[v] = self.node("literal",v)
        return self.literals[v]

    def param(self,v):
        j = len(self.params)
        self.params.append(j)
        self.written.append(float(v))
        return self.node("weight",j)

    def weights(self,seed=11):
        # Each coefficient edge has scalar fan-in and fan-out. Full writer overwrites all slots.
        p = np.random.default_rng(seed).uniform(-np.sqrt(3),np.sqrt(3),len(self.params))
        p[:] = self.written
        return p

    def summary(self):
        counts = {op:sum(n[0]==op for n in self.nodes) for op in ("add","sub","mul","relu")}
        return dict(weights=len(self.params),literals=len(self.literals),gates=sum(counts.values()),**counts)

    def run(self,x,p,batch=1024):
        # Release intermediates after their last use; the largest DAG contains thousands.
        use = [0]*len(self.nodes)
        for node in self.nodes:
            if node[0] in ("add","sub","mul","relu"):
                for idx in node[1:]:
                    use[idx]+=1
        pieces = []
        for start in range(0,len(x),batch):
            values = [None]*len(use)
            active = use.copy()
            values[0] = x[start:start+batch]
            for i,(op,*args) in enumerate(self.nodes[1:],1):
                if op=="literal": values[i]=args[0]
                elif op=="weight": values[i]=p[args[0]]
                else:
                    a=values[args[0]]
                    if op=="relu": values[i]=np.maximum(0,a)
                    elif op=="add": values[i]=a+values[args[1]]
                    elif op=="sub": values[i]=a-values[args[1]]
                    elif op=="mul": values[i]=a*values[args[1]]
                    for idx in args:
                        active[idx]-=1
                        if active[idx]==0: values[idx]=None
            pieces.append(values[-1])
        return np.concatenate(pieces)
