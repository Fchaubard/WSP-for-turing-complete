"""Pooled raw errors for held-out counter transitions, autonomous traces and outputs."""

import json
import sys
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "counters"))
import run
import train


def main():
    rows = []
    saved = np.load(EXT / "counters/bp_states.npz", allow_pickle=False)
    for task, (nreg, code, readout) in run.PROGRAMS.items():
        data = [("WSP_compiler", run.compile_program(nreg, code, 215))]
        if task in train.TASKS:
            for name in ["structured", *[f"bp_{s}" for s in train.SEEDS]]:
                data.append((name, [saved[f"{task}_{name}_{k}"] for k in "DWE"]))
        for method, w in data:
            splits = {
                "all_central": run.one_step_pairs(nreg, code, range(4)),
                "outer": run.one_step_pairs(nreg, code, range(4, 11)),
            }
            if task in train.TASKS:
                splits["train"], splits["dev"] = train.split_central(nreg, code)
                splits["mixed"] = run.one_step_pairs(nreg, code, [0, 1, 4, 10, 1024])
            for split, (x, y) in splits.items():
                score = metric(run.forward(x, w), y)
                rows.append(
                    dict(
                        task=task,
                        method=method,
                        split=split,
                        prediction="one_step_state",
                        **score,
                    )
                )
            for split, values in [("central", range(4)), ("outer", range(4, 11))]:
                pred = []
                target = []
                outputs = []
                answers = []
                for counters, goal in run.trials(task, values):
                    pc = 0
                    z = run.pack_state(counters, pc, len(code))
                    while code[pc][0] != "halt":
                        z = run.forward(z, w)
                        counters, pc = run.interpret(counters, pc, code)
                        pred.append(z)
                        target.append(run.pack_state(counters, pc, len(code)))
                    outputs.append(z[readout])
                    answers.append(goal)
                for kind, a, b in [
                    ("autonomous_state", pred, target),
                    ("final_scalar", outputs, answers),
                ]:
                    rows.append(
                        dict(
                            task=task,
                            method=method,
                            split=split,
                            prediction=kind,
                            **metric(a, b),
                        )
                    )
    for r in json.loads((EXT / "counters/literal/scores.json").read_text()):
        assert r["algebraic_certificate"] == "output_relu_clips_decrement"
        for split, d in r.items():
            if isinstance(d, dict) and "n" in d:
                assert d["correct"] == d["n"]
                n = d["n"] * 6 * (128 if split == "trace128" else 1)
                rows.append(
                    dict(
                        task="relabeled",
                        method="WSP_Astra_" + str(r["seed"]),
                        split=split,
                        prediction="state",
                        n=n,
                        exact=n,
                        raw_rmse="0",
                    )
                )
    save("machine", rows)


if __name__ == "__main__":
    main()
