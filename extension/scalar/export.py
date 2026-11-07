"""Export the two compensated, entirely affine/ReLU programs for standalone inference."""
import json
import numpy as np
from pathlib import Path
from compile import relu_sin_graph,static_input_check
from relu_power import power_graph
from infer import OPCODES

def export():
    models={"sin":relu_sin_graph(-78),"selfpower":power_graph(84,True)}
    data={}
    info={}
    for name,g in models.items():
        table=np.zeros((len(g.nodes),3),dtype=np.int32)
        fixed=[]
        for i,(op,*args) in enumerate(g.nodes):
            table[i,0]=OPCODES.index(op)
            if op=="literal":
                table[i,1]=len(fixed)
                fixed.append(args[0])
            else:
                table[i,1:len(args)+1]=args
        data[name+"_ops"]=table
        data[name+"_weights"]=g.weights()
        data[name+"_fixed"]=np.asarray(fixed)
        # Counts of the expanded affine graph, whether or not scalar edge
        # coefficients are tied / repeated / structurally +/-1. Every binary
        # dynamic addition has two coefficients (or input gain and offset).
        static=[]
        slots=0
        for op,*a in g.nodes:
            if op=="x":s=False
            elif op in ("weight","literal"):s=True
            else:
                s=all(static[i] for i in a)
                if not s:
                    slots+=2 if op in ("add","sub") else 1 if op=="mul" else 0
            static.append(s)
        info[name]={**g.summary(),**static_input_check(g),
                    "unshared_affine_occurrences":slots,"static_literals":fixed,
                    "declared_domain":[-16,16] if name=="sin" else [0,128],
                    "writer":"binary angle rotations" if name=="sin" else "greedy log and binary product/exponential"}
    root=Path(__file__).parent
    np.savez_compressed(root/"programs.npz",**data)
    (root/"programs.json").write_text(json.dumps(info,indent=2)+"\n")
if __name__=="__main__":
    export()
