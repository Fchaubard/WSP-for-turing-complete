"""Replay each published tanh size, arithmetic mode, and test panel."""

import argparse
from pathlib import Path
import sys
import csv
from collections import defaultdict
from urllib.request import urlopen
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "tanh"))
import run
from circuit import tanh_graph


def main(core):
    payload = core.read_bytes() if core else urlopen(run.URL, timeout=90).read()
    data = run.data(payload)
    data["outer"] = np.r_[data["left"], data["right"]]
    refs = {k: run.reference(x, 53) for k, x in data.items()}
    groups = defaultdict(dict)
    with (EXT / "tanh/results.csv").open() as f:
        for r in csv.DictReader(f):
            groups[r["method"], int(r["bits"])][r["split"]] = int(r["exact"])
    rows = []
    for (method, bits), splits in groups.items():
        g = (
            None
            if method == "numpy_tanh"
            else tanh_graph(
                bits,
                method != "one_word",
                guard_round=method.startswith("guard"),
                triple=method == "guard_three_word",
            )
        )
        if "left" in splits:
            splits["outer"] = None
        for key, exact in splits.items():
            x = data[key]
            yhat = np.tanh(x) if g is None else g.run(x, g.weights(), batch=2048)
            score = metric(yhat, refs[key])
            assert exact is None or score["exact"] == exact
            rows.append(dict(task="tanh", method=method, bits=bits, split=key, **score))
    save("tanh", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--core", type=Path)
    main(p.parse_args().core)
