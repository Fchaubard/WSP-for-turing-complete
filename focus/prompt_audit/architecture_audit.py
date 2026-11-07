"""Independent graph replay and operation/liveness checks for the frozen architecture replies."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCH = ROOT/"focus/architecture_search"
sys.path.insert(0,str(ARCH))
from network import Network
sys.path.insert(0,str(ROOT/"extension/writer"))
from design import observed, inputs

def references(node):
    if node["op"]=="constant":
        return []
    if "source" in node:
        return [node["source"]]
    if "left" in node:
        return [node["left"],node["right"]]
    return node["sources"]

def graph(nodes,output,external):
    # Keep values for evaluation separately from the abstract live-buffer sizes.
    values={name:np.asarray(value,dtype=float) for name,value in external.items()}
    live={name:value.size for name,value in values.items()}
    last={}
    for i,node in enumerate(nodes):
        for ref in references(node):
            last[ref]=i
    last[output]=len(nodes)
    arithmetic, comparisons, transcendental, peak = 0,0,0,sum(live.values())
    for i,node in enumerate(nodes):
        op=node["op"];args=[values[r] for r in references(node)]
        if op=="constant":
            value=np.array(node["value"],dtype=float)
        elif op=="affine":
            w=np.array(node["weight"],dtype=float);x=args[0].flatten()
            value=w@x
            arithmetic+=w.shape[0]*(2*w.shape[1]-1)
            if "bias" in node:
                value=value+np.asarray(node["bias"]);arithmetic+=w.shape[0]
        elif op=="conv1d":
            w=np.asarray(node["weight"]);x=args[0]
            channels,inputs,kernel=w.shape
            value=np.array([[np.sum(w[c]*x[j:j+kernel].T) for c in range(channels)]
                            for j in range(len(x)-kernel+1)])
            arithmetic+=value.size*(2*inputs*kernel-1)
            if "bias" in node:
                value=value+np.asarray(node["bias"]);arithmetic+=value.size
        elif op=="relu":
            value=np.maximum(args[0],0);comparisons+=value.size
        elif op=="tanh":
            value=np.tanh(args[0]);transcendental+=value.size
        elif op=="sigmoid":
            value=1/(1+np.exp(-args[0]));transcendental+=value.size;arithmetic+=value.size*3
        elif op=="add":
            value=args[0]+args[1];arithmetic+=value.size
        elif op=="multiply":
            value=args[0]*args[1];arithmetic+=value.size
        elif op=="matmul":
            value=np.asarray(args[0]@args[1]);arithmetic+=value.size*(2*args[0].shape[-1]-1)
        elif op=="reshape":
            value=args[0].reshape(node["shape"])
        elif op=="take":
            value=args[0].flatten()[node["indices"]]
        elif op=="transpose":
            value=args[0].T
        elif op=="concat":
            value=np.concatenate([a.flatten() for a in args])
        elif op=="softmax":
            x=args[0];n=x.shape[-1];b=x.size//n
            e=np.exp(x-x.max(axis=-1,keepdims=True));value=e/e.sum(axis=-1,keepdims=True)
            arithmetic+=b*(3*n-1);comparisons+=b*(n-1);transcendental+=x.size
        else:
            raise ValueError(op)
        value=np.asarray(value,dtype=float).copy()
        if value.ndim==0:value=value.reshape(1)
        values[node["name"]]=value
        live[node["name"]]=value.size
        peak=max(peak,sum(live.values()))
        for ref in set(references(node)+[node["name"]]):
            if last.get(ref,-1)<=i:
                live.pop(ref,None)
    return values[output],(arithmetic,comparisons,transcendental,peak)

def coefficient_count(spec):
    nodes=spec.get("nodes",[])+spec.get("cell",[])+spec.get("readout",[])
    total=sum(np.asarray(node[field]).size for node in nodes for field in ["value","weight","bias"] if field in node)
    return total+len(spec.get("initial_state",[]))

def independent(spec,mode,value):
    if spec["kind"]=="graph":
        return graph(spec["nodes"],spec["output"],dict(input=np.atleast_1d(value)))
    state=np.array(spec["initial_state"],dtype=float)
    results=[];counts=np.zeros(3,dtype=int)
    memory=state.size+np.atleast_1d(value).size
    if mode=="index":
        n=int(value)
        for _ in range(n):
            state,c=graph(spec["cell"],spec["state_output"],dict(state=state,input=np.array([0.])))
            counts+=c[:3];memory=max(memory,c[3]+1)
        out,c=graph(spec["readout"],spec["output"],dict(state=state,input=np.array([0.])))
        counts+=c[:3];memory=max(memory,c[3]+1)
        return out,(*counts,memory)
    length=len(value)
    for tick,x in enumerate(value):
        context=dict(state=state,input=np.array([x]))
        state,c=graph(spec["cell"],spec["state_output"],context)
        counts+=c[:3];memory=max(memory,c[3]+length+tick)
        out,c=graph(spec["readout"],spec["output"],dict(state=state,input=np.array([x])))
        counts+=c[:3];memory=max(memory,c[3]+length+tick)
        results.append(float(out[0]))
    memory=max(memory,length+len(results)+state.size)
    return np.array(results),(*counts,memory)

def main():
    assert (ARCH/"FROZEN.json").is_file()
    manifest=json.loads((ARCH/"manifest.json").read_text())
    published={r["id"]:r for r in csv.DictReader((ARCH/"models.csv").open())}
    records=[]
    for case in manifest:
        raw=(ARCH/"trials"/(case["id"]+".reply.txt")).read_text()
        reply=json.loads(raw);spec=reply["network"]
        assert set(reply)=={"family","inferred_rule","construction_summary","certificate","network"}
        assert all(isinstance(reply[k],str) for k in ["family","inferred_rule","construction_summary","certificate"])
        if published[case["id"]]["status"]=="invalid":
            nodes=spec["nodes"]
            if len(nodes)>64:
                reason=f"{len(nodes)} nodes exceed the 64-node predeclared bound"
            else:
                defined={"input"};missing=[]
                for node in nodes:
                    missing.extend(ref for ref in references(node) if ref not in defined)
                    defined.add(node["name"])
                assert missing
                reason="undefined input reference "+",".join(missing)
            try:Network(reply,case["mode"])
            except (ValueError,KeyError):pass
            else:raise AssertionError("Invalid model accepted")
            records.append(dict(task=case["task"],seed=case["seed"],status="invalid_confirmed",examples_checked=0,reason=reason))
            continue
        model=Network(reply,case["mode"])
        assert coefficient_count(spec)==model.coefficients==int(published[case["id"]]["coefficients"])
        train,_=observed(case["task"],case["seed"])
        outer=inputs(case["task"])["outer"]
        selected=list(train)+list(outer[:17])+list(outer[-17:])
        if case["task"]=="sum":
            selected.extend([np.array([1.,2.,3.]),np.arange(32,dtype=float)])
        for x in selected:
            expected,cost=independent(spec,case["mode"],x)
            actual,measured=model.one(x)
            assert np.array_equal(expected,actual),(case["task"],case["seed"],x)
            assert tuple(map(int,cost))==(measured.arithmetic,measured.comparisons,measured.transcendental,measured.peak_scalars),(case,cost,measured)
        records.append(dict(task=case["task"],seed=case["seed"],status="valid_checked",examples_checked=len(selected),reason="coefficients, arithmetic, comparisons, transcendental calls and peak live tensors agree"))
    # Exercise supported convolution and attention operations absent from the actual choices.
    synthetic=[
        ("convolution",{"kind":"graph","nodes":[{"name":"x","op":"reshape","source":"input","shape":[3,1]},
         {"name":"y","op":"conv1d","source":"x","weight":[[[2,3]]],"bias":[1]}],"output":"y"},[1,2,3],np.array([[9],[14]]),8),
        ("attention",{"kind":"graph","nodes":[{"name":"x","op":"reshape","source":"input","shape":[2,1]},
         {"name":"xt","op":"transpose","source":"x"},{"name":"qk","op":"matmul","left":"x","right":"xt"},
         {"name":"a","op":"softmax","source":"qk"},{"name":"y","op":"matmul","left":"a","right":"x"}],"output":"y"},
         [0,1],np.array([[.5],[math.e/(1+math.e)]]),20)]
    for label,spec,x,y,flops in synthetic:
        reply={"network":spec}
        actual,c=Network(reply,"vector").one(x)
        independent_value,independent_cost=independent(spec,"vector",x)
        assert np.allclose(actual,y,rtol=1e-15)
        assert np.array_equal(actual,independent_value)
        assert c.arithmetic==flops==independent_cost[0]
        assert c.peak_scalars==independent_cost[3]
        records.append(dict(task=label,seed="",status="synthetic_operator_check",examples_checked=1,reason=f"{flops} FLOPs; output and liveness verified"))
    with (HERE/"architecture_review.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0])
        writer.writeheader();writer.writerows(records)
    summary=dict(valid_models_checked=28,invalid_models_confirmed=2,
                 examples_checked=sum(int(r["examples_checked"]) for r in records),
                 extra_operators_checked=["convolution","attention with softmax"],
                 source_sha256=hashlib.sha256((ARCH/"network.py").read_bytes()).hexdigest(),
                 no_material_accounting_discrepancy=True)
    (HERE/"architecture_review.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))

if __name__=="__main__":
    main()
