"""Evaluate all fixed free-topology WSP replies on their original and challenge panels."""

import argparse
import csv
from pathlib import Path
import sys
import json
from urllib.request import urlopen
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "synthesis"))
sys.path.insert(0, str(EXT / "synthesis/repair"))
from feedback import compile_reply, reference, selected
from experiment import variants, samples
from score import oracle
from challenge import panels, URL


def main(core):
    stress = panels(core.read_bytes() if core else urlopen(URL, timeout=90).read())
    records = {
        r["id"]: r for r in json.loads((EXT / "synthesis/results.json").read_text())
    }
    with (EXT / "synthesis/challenge.csv").open() as f:
        old = {(r["id"], r["split"]): int(r["exact"]) for r in csv.DictReader(f)}
    rows = []
    data = {}
    for task in stress:
        arrays = samples(task)
        arms = {"original_" + k: v for k, v in arrays.items()} | stress[task]
        data[task] = {
            k: (v, v * v if task == "square" else reference(v, task))
            for k, v in arms.items()
        }
    for case in variants():
        task, uid = case["task"], case["id"]
        record = records[uid]
        graph = (
            compile_reply(
                (EXT / "synthesis/trials" / (uid + ".reply.txt")).read_text(),
                case["arm"],
            )
            if record["status"] == "scored"
            else None
        )
        for split, (x, y) in data[task].items():
            if graph is None:
                m = {"n": len(y), "exact": "", "raw_rmse": "unavailable_invalid"}
            else:
                pred = np.concatenate(
                    [graph.run(x[i : i + 384]) for i in range(0, len(x), 384)]
                )
                m = metric(pred, y)
                expected = (
                    record[split.removeprefix("original_")]["exact"]
                    if split.startswith("original_")
                    else old[(uid, split)]
                )
                assert m["exact"] == expected, (uid, split, m)
            rows.append(dict(**case, split=split, **m))
    save("synthesis", rows)
    output = []
    with (EXT / "synthesis/repair/results.csv").open() as f:
        old = {
            (r["id"], r["stage"], r["split"]): int(r["correct"])
            for r in csv.DictReader(f)
        }
    for case in selected():
        task, uid = case["task"], case["id"]
        for stage, path in [
            ("original", EXT / "synthesis/trials"),
            ("repair", EXT / "synthesis/repair/trials"),
        ]:
            graph = compile_reply(
                (path / (uid + ".reply.txt")).read_text(), case["arm"]
            )
            for split, (x, y) in oracle(task).items():
                pred = np.concatenate(
                    [graph.run(x[i : i + 384]) for i in range(0, len(x), 384)]
                )
                m = metric(pred, y)
                assert m["exact"] == old[uid, stage, split]
                output.append(dict(**case, stage=stage, split=split, **m))
    save("repair", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--core", type=Path)
    main(p.parse_args().core)
