"""Compute paired held-out RMSEs from saved deployed graphs and Adam checkpoints."""

import json, sys
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "smooth_optimizer"))
from bench import samples, reference, load_graph


def main():
    rows = []
    for task in ("selfpower", "tanh"):
        g, w = load_graph(task)
        data = samples(task)
        data["outer"] = (
            np.r_[data["left"], data["right"]] if task == "tanh" else data["right"]
        )
        refs = {k: reference(x, task, 53) for k, x in data.items()}
        old = json.loads((EXT / "smooth_optimizer" / (task + ".json")).read_text())
        with np.load(EXT / "smooth_optimizer" / (task + "_weights.npz")) as ck:
            cases = [("WSP", 0, w, old[0]["scores"])]
            for r in old[1:]:
                for stage, label in [("initial", "Xavier"), ("terminal", "BP_Adam")]:
                    cases.append(
                        (
                            label,
                            r["seed"],
                            ck[stage + "_" + str(r["seed"])],
                            r["initial" if stage == "initial" else "final"],
                        )
                    )
            for method, seed, p, rec in cases:
                for k, x in data.items():
                    scores = metric(g.run(x, p), refs[k])
                    if k in rec:
                        assert scores["exact"] == rec[k]["exact"]
                    rows.append(
                        dict(task=task, method=method, seed=seed, split=k, **scores)
                    )
    save("smooth_adam", rows)


if __name__ == "__main__":
    main()
