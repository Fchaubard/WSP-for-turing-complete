"""Replay frozen inference against the stored strongest-program counts and failure witnesses."""
import csv
import json
from pathlib import Path
import numpy as np
from infer import predict
from run import samples,stress_samples,reference,ulps
from stress import special

root=Path(__file__).parent
for task,file,method,order in [("sin","relu.csv","relu_rotation_two_word",83),
                               ("selfpower","relu_power.csv","relu_selfpower_two_word",84)]:
    inputs={**samples(task),**stress_samples(task),**special(task)}
    records=[]
    for source in (file,"stress.csv"):
        with (root/source).open() as f:
            records.extend(r for r in csv.DictReader(f) if r["task"]==task and r["method"]==method and int(r["order"])==order)
    for r in records:
        x=inputs[r["split"]]
        y=reference(x,task)
        delta=ulps(predict(task,x),y)
        assert len(x)==int(r["n"])
        assert np.sum(delta==0)==int(r["correct"])
        assert np.sum(delta<=1)==int(r["within1"])
        assert delta.max()==int(r["worst_ulp"])
    failures=json.loads((root/("hard_sin_failures.json" if task=="sin" else "counterexamples.json")).read_text())
    for e in failures:
        x=float.fromhex(e["x"])
        assert predict(task,x).hex()==e["output"]
        assert reference(np.array([x]),task)[0].hex()==e["mpfr"]
    print(f"{task}: {len(records)} split results and {len(failures)} counterexamples reproduced")
