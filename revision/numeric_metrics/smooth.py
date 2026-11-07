"""Replay scalar size sweeps, product controls and training against MPFR."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
from urllib.request import urlopen
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "scalar"))
import run
import train
from compile import sin_graph, selfpower_graph, relu_sin_graph
from relu_power import power_graph
from stress import special
from hard_sin import URL, DIGEST
import hashlib


def main(core):
    raw = core.read_bytes() if core else urlopen(URL, timeout=90).read()
    assert hashlib.sha256(raw).hexdigest() == DIGEST
    vals = set()
    for line in raw.splitlines():
        try:
            x = float.fromhex(line.decode().strip())
        except ValueError:
            continue
        if 0 < abs(x) <= 16:
            vals.add(x)
    hard = np.array(sorted(vals))
    arrays = {}
    for task in ("sin", "selfpower"):
        data = {**run.samples(task), **run.stress_samples(task), **special(task)}
        data["outer"] = (
            np.r_[data["left"], data["right"]] if task == "sin" else data["right"]
        )
        if task == "sin":
            data.update(
                core_domain=hard, core_shell=hard[(abs(hard) > 3) & (abs(hard) <= 10)]
            )
        arrays[task] = run.labeled(data, task)
    scores = []
    grouped = defaultdict(dict)
    for fn in ("sweep", "relu", "relu_power", "stress", "hard_sin", "train"):
        with (EXT / "scalar" / (fn + ".csv")).open() as f:
            for r in csv.DictReader(f):
                grouped[r["task"], r["method"], int(r["order"]), int(r["stage"])][
                    r["split"]
                ] = int(r["correct"])

    def record(task, method, g, order, stage, p=None):
        group = grouped.pop((task, method, order, stage))
        if "left" in group or "right" in group:
            group["outer"] = None
        for split, previous in group.items():
            x, y, _ = arrays[task][split]
            values = g.run(x, g.weights() if p is None else p)
            result = metric(values, y)
            assert previous is None or result["exact"] == previous, (
                task,
                method,
                split,
                previous,
                result,
            )
            scores.append(
                dict(
                    task=task,
                    method=method,
                    order=order,
                    stage=stage,
                    split=split,
                    **result,
                )
            )

    for task, method, order, stage in list(grouped):
        if method.startswith("all_slot_") or method == "central_gauss_newton":
            continue
        if method.startswith("relu_rotation"):
            g = relu_sin_graph(stage, method.endswith("two_word"))
        elif method.startswith("relu_selfpower"):
            g = power_graph(order, method.endswith("two_word"))
        elif task == "sin":
            g = sin_graph(order, method == "two_word")
        else:
            g = selfpower_graph(order, stage, method == "two_word")
        record(task, method, g, order, stage)

    def capture(rows, task, method, g, order, stage, dataset, params=None):
        record(task, method, g, order, stage, params)

    run.rows_for = capture
    run.central_fits([], arrays["sin"])
    train.rows_for = capture
    train.save = lambda *args: None
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        train.__file__ = str(Path(d) / "train.py")
        train.run()
    assert not grouped
    save("scalar", scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--core", type=Path)
    main(p.parse_args().core)
