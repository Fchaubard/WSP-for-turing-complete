"""Generate the prespecified examples, dense network parameters and isolated prompts."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extension/writer"))
from design import FORMULAS, SEEDS, inputs, observed, reference

WIDTHS = (1, 2, 4)
TASKS = ("add", "subtract")


def initial(width, seed):
    rng = np.random.default_rng(seed + 224 + width)
    first = rng.uniform(-np.sqrt(6 / (2 + width)), np.sqrt(6 / (2 + width)), (width, 2))
    second = rng.uniform(-np.sqrt(6 / (width + 1)), np.sqrt(6 / (width + 1)), width)
    return np.r_[first.ravel(), np.zeros(width), second, 0.0]


def extract(width, theta):
    return (
        theta[: 2 * width].reshape(width, 2),
        theta[2 * width : 3 * width],
        theta[3 * width : 4 * width],
        theta[-1],
    )


def forward(width, theta, x):
    w, c, v, b = extract(width, theta)
    z = x[:, :1] * w[:, 0] + x[:, 1:] * w[:, 1] + c
    h = np.maximum(z, 0.0)
    return np.sum(h * v, axis=1) + b, z, h


def prompt(task, seed, arm):
    x, y = observed(task, seed)
    text = (
        "Return weights for three independent two-input, one-output dense neural networks. "
        "Their common target is defined for coordinate values in [-10,10]. "
    )
    if arm == "formula":
        text += "The target specification is " + FORMULAS[task] + ". "
    text += (
        "The twelve labeled observations, in the format [[x1,x2],y], are "
        + json.dumps(list(zip(x.tolist(), y.tolist())), separators=(",", ":"))
        + ". The primary objective for each variant is to minimize TRAINING LOSS: one half "
        "of the mean over exactly these twelve examples of ((prediction-y)/max(1,abs(y))) squared. "
        "Among equally good training fits, seek exact generalization to new inputs throughout "
        "the stated domain. You may infer the rule from observations. Each network is a standard "
        "one-hidden-layer rectified linear network, with no input-to-output skip. Unit j has "
        "h[j]=max(0,(w[j,0]*x1+w[j,1]*x2)+c[j]); prediction=sum_j h[j]*v[j] + b. "
        "Operations use binary64. Flatten all rows of w (two entries per unit), then all c, "
        "then all v, then the single output bias b. Every listed scalar is free to change. "
        "Use no tools or external files. Return only a JSON array of three complete numeric "
        "arrays, one per width in order. Do not use expressions or explanations.\n"
    )
    for width in WIDTHS:
        theta = initial(width, seed)
        text += (
            f"Width {width}, {len(theta)} scalars, initial array: "
            + json.dumps(theta.tolist(), separators=(",", ":"))
            + "\n"
        )
    return text


def main():
    cases = []
    frozen = {}
    for task in TASKS:
        for name, xx in inputs(task).items():
            frozen[f"{task}_{name}_inputs"] = xx
            frozen[f"{task}_{name}_labels"] = reference(task, xx)
        for seed in SEEDS:
            xx, yy = observed(task, seed)
            frozen[f"{task}_{seed}_train_inputs"] = xx
            frozen[f"{task}_{seed}_train_labels"] = yy
            for width in WIDTHS:
                frozen[f"start_{seed}_{width}"] = initial(width, seed)
    np.savez_compressed(HERE / "fixtures.npz", **frozen)
    (HERE / "trials").mkdir(exist_ok=True)
    for task in TASKS:
        for seed in SEEDS:
            for arm in ("formula", "examples"):
                ident = hashlib.sha256(
                    f"dense-relu-arithmetic:{task}:{seed}:{arm}".encode()
                ).hexdigest()[:12]
                text = prompt(task, seed, arm)
                (HERE / "trials" / (ident + ".prompt.txt")).write_text(text)
                cases.append(
                    dict(
                        id=ident,
                        task=task,
                        seed=seed,
                        arm=arm,
                        sizes=WIDTHS,
                        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                )
    (HERE / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
