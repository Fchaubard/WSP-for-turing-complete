"""Replay every seed and split, including raw model replies and gradient coefficients."""

import csv
import hashlib
import json
import sys

import numpy as np

from fixtures import (
    HERE,
    ROOT,
    SEEDS,
    TASKS,
    WIDTHS,
    forward,
    initial,
    inputs,
    observed,
    prompt,
    reference,
)
from fit import reverse_loss

sys.path.insert(0, str(ROOT))
from error_metrics import rmse


def read_replies():
    states = {}
    metadata = []
    for case in json.loads((HERE / "manifest.json").read_text()):
        task, seed, arm = case["task"], case["seed"], case["arm"]
        base = HERE / "trials" / case["id"]
        text = base.with_suffix(".prompt.txt").read_text()
        assert text == prompt(task, seed, arm)
        assert hashlib.sha256(text.encode()).hexdigest() == case["prompt_sha256"]
        reply = base.with_suffix(".reply.txt").read_bytes()
        record = json.loads(base.with_suffix(".meta.json").read_text())
        assert record["requested_model"] == "gpt-6-astra"
        assert record["completed_items"] == {"agent_message": 1}
        try:
            response = json.loads(reply)
            if not isinstance(response, list) or len(response) != len(WIDTHS):
                response = []
        except ValueError:
            response = []
        for i, width in enumerate(WIDTHS):
            try:
                theta = np.array(response[i], dtype=float)
            except (ValueError, TypeError, IndexError):
                theta = np.array([])
            valid = theta.shape == (4 * width + 1,) and np.isfinite(theta).all()
            metadata.append(
                dict(
                    id=case["id"],
                    task=task,
                    seed=seed,
                    arm=arm,
                    width=width,
                    valid=bool(valid),
                    reply_sha256=hashlib.sha256(reply).hexdigest(),
                )
            )
            states[task, seed, width, arm] = theta if valid else None
    return states, metadata


def main():
    writer, metadata = read_replies()
    rows = []
    with np.load(HERE / "fits.npz", allow_pickle=False) as fits:
        for task in TASKS:
            data = inputs(task)
            for seed in SEEDS:
                x, y = observed(task, seed)
                sets = {
                    "train": (x, y),
                    "dev": (data["inner"], reference(task, data["inner"])),
                    "outer": (data["outer"], reference(task, data["outer"])),
                }
                for width in WIDTHS:
                    for method in (
                        "formula",
                        "examples",
                        "adam_bfgs",
                        "trf_xavier",
                        "xavier",
                        "engineered",
                    ):
                        theta = (
                            writer[task, seed, width, method]
                            if method in ("formula", "examples")
                            else fits[f"{task}_{seed}_{width}_{method}"]
                        )
                        if method == "xavier":
                            assert np.array_equal(theta, initial(width, seed))
                        train_loss = (
                            float(reverse_loss(width, theta, x, y)[0])
                            if theta is not None
                            else None
                        )
                        for split, (xx, yy) in sets.items():
                            exact = rmse_val = None
                            if theta is not None:
                                pred, _, _ = forward(width, theta, xx)
                                assert np.isfinite(pred).all()
                                exact = int(np.count_nonzero(pred == yy))
                                rmse_val = rmse(pred, yy)
                                if exact == len(yy):
                                    assert rmse_val == 0
                            rows.append(
                                dict(
                                    task=task,
                                    size=width,
                                    seed=seed,
                                    method=method,
                                    split=split,
                                    exact=exact,
                                    rmse=rmse_val,
                                    train_loss=train_loss,
                                    params=4 * width + 1,
                                    n_examples=len(xx),
                                    n_scalars=yy.size,
                                )
                            )
    with (HERE / "scores.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    (HERE / "response_status.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        len(rows),
        "metric rows,",
        sum(r["valid"] for r in metadata),
        "of",
        len(metadata),
        "valid arrays",
    )


if __name__ == "__main__":
    main()
