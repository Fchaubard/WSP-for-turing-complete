"""Pure affine/ReLU binary-angle sine sweep and precision audits."""
from compile import relu_sin_graph,static_input_check
from run import labeled,samples,stress_samples,rows_for,save
import json
from pathlib import Path

def run():
    base=labeled(samples("sin"),"sin")
    stress=labeled(stress_samples("sin"),"sin")
    rows=[]
    counts={}
    for last in (-12,-24,-32,-44,-52,-64,-78):
        for double in (False,True):
            g=relu_sin_graph(last,double)
            m="relu_rotation_two_word" if double else "relu_rotation_single_word"
            counts[f"{m}_{last}"]=static_input_check(g)
            rows_for(rows,"sin",m,g,4-last+1,last,base)
            if last in (-52,-78):
                rows_for(rows,"sin",m,g,4-last+1,last,stress)
        save(rows,"relu.csv")
        print(last,flush=True)
    (Path(__file__).parent/"relu_counts.json").write_text(json.dumps(counts,indent=2)+"\n")
if __name__=="__main__":
    run()
