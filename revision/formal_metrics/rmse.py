"""Look up original observations, replay saved weights, and report unscaled RMSEs."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extension" / "writer"))
from design import inputs, observed, predict, reference


def read_rows(relative_path):
    with (ROOT / relative_path).open() as stream:
        return list(csv.DictReader(stream))


def save_rows(filename, rows):
    with (HERE / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(predictions, targets):
    """Scaling before squaring avoids intermediate overflow. Report in target units."""
    delta = np.abs(predictions - targets)
    largest = float(delta.max())
    if largest == 0 or not np.isfinite(largest):
        return largest
    return largest * float(np.sqrt(np.mean((delta / largest) ** 2)))


def fmt(value):
    if value == 0:
        return "0"
    if not np.isfinite(value):
        return r"\infty"
    if abs(value) < 0.001 or abs(value) >= 10000:
        mantissa, exponent = f"{value:.2e}".split("e")
        return mantissa + r"\times 10^{" + str(int(exponent)) + "}"
    return f"{value:.3g}"


def write_table_rows(name, rows):
    path = ROOT / "paper" / ("rmse_" + name + ".tex")
    with path.open("w") as stream:
        for row in rows:
            stream.write(" & ".join(row) + r"\\" + "\n")
        stream.write(r"\bottomrule" + "\n")


def optimizer():
    records = read_rows("extension/optimizer/results.csv")
    cache = {task: inputs(task) for task in {r["task"] for r in records}}
    scores = []
    with np.load(ROOT / "extension/optimizer/states.npz", allow_pickle=False) as states:
        for model in records:
            task, method = model["task"], model["method"]
            size, seed = int(model["size"]), int(model["seed"])
            weights = states[f"{task}_{size}_{seed}_{method}"]
            training_x, _ = observed(task, seed)
            sets = {"train": training_x, "dev": cache[task][model["dev_split"]],
                    "outer": cache[task]["outer"]}
            for split, x in sets.items():
                y = reference(task, x)
                yh = predict(task, size, weights, x)
                error = rmse(yh, y)
                equal = yh == y
                exact = float(np.mean(equal.all(axis=1) if equal.ndim == 2 else equal))
                assert exact == float(model[split + "_exact"])
                scores.append(dict(task=task, size=size, seed=seed, method=method,
                                   split=split, labels=int(y.size), exact=exact, rmse=error))
    save_rows("optimizer_rmse.csv", scores)
    write_optimizer_table(scores)
    # Cardinalities are fixed by the study design, independent of successes.
    assert len(scores) == 945 * 3


def write_optimizer_table(scores):
    target_panels = (
        (("square", "Square"), ("sin", "Sine"), ("selfpower", "Self-power"),
         ("tanh", "Tanh"), ("add", "Addition")),
        (("subtract", "Subtraction"), ("sum", "Running sum"), ("sort", "Sort"),
         ("fib", "Fibonacci"), ("tri", "Triangular")),
    )
    methods = (
        ("formula", "WSP, Formula and examples"),
        ("examples", "WSP, Examples only"),
        ("ls_formula", r"WSP, Formula and examples $\to$ LS"),
        ("ls_examples", r"WSP, Examples only $\to$ LS"),
        ("adam800", r"Xavier $\to$ Adam"),
        ("adam_bfgs", r"Xavier $\to$ Adam+BFGS"),
        ("ls_xavier", r"Xavier $\to$ LS"),
    )
    outer = defaultdict(list)
    for record in scores:
        if record["split"] == "outer":
            outer[record["task"], record["method"]].append(float(record["rmse"]))
    lines = []
    for panel_index, panel in enumerate(target_panels):
        if panel_index:
            lines.append(r"\midrule")
        lines.append("Method & " + " & ".join(name for _, name in panel) + r"\\\midrule")
        for key, label in methods:
            cells = [label]
            for task, _ in panel:
                values = outer[task, key]
                assert len(values) == 9
                cells.append("$" + fmt(median(values)) + "$")
            lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    (ROOT / "paper/rmse_optimizer.tex").write_text("\n".join(lines) + "\n")


def recurrent():
    scores = []
    for name in ("results.json", "longer/results.json"):
        archive = json.loads((ROOT / "extension/recurrent" / name).read_text())
        for model in archive["rows"]:
            for split, metric in model["metrics"].items():
                scores.append(dict(task=model["task"], size=model["width"], cell=model["cell"],
                    seed=model["seed"], method="WSP" if model["method"] == "program" else "BP",
                    budget=archive["settings"]["steps"], split=split, exact=metric["exact_all"],
                    rmse=metric["rmse"], nrmse=metric["nrmse"]))
    save_rows("recurrent_rmse.csv", scores)
    rows = []
    for task, label in (("sum","Sum"),("fib","Fibonacci"),("tri","Triangular"),("sort","Sort")):
        widths = [1,2,4,8] + ([16] if task == "sort" else [])
        for i, width in enumerate(widths):
            row = [label if i == 0 else "", str(width)]
            for cell in ("rnn","gru"):
                for method in ("WSP","BP"):
                    for split in ("ID","joint"):
                        vals = [r["rmse"] for r in scores if r["task"] == task and r["size"] == width and r["cell"] == cell and r["method"] == method and r["split"] == split and r["budget"] == 5000]
                        row.append("$"+fmt(median(vals))+"$" if vals else "---")
            rows.append(row)
    write_table_rows("recurrent", rows)


def shallow():
    records = read_rows("scalar/shallow.csv")
    group = defaultdict(list)
    for r in records:
        if r["split"] in ("left","right") and r["method"] in ("write_global","adam_lbfgs"):
            group[r["task"],int(r["size"]), r["method"],r["seed"]].append((int(r["n"]),float(r["rmse"])))
    pooled = defaultdict(list)
    for (task, size, method, seed), splits in group.items():
        pooled[task,size,method].append(math.sqrt(sum(n*x*x for n,x in splits)/sum(n for n,x in splits)))
    rows = []
    for width in (0,1,3,7,15,31,63):
        row = [str(width),str(3*width+2)]
        for task in ("square","sin","selfpower","tanh"):
            for method in ("write_global","adam_lbfgs"):
                row.append("$"+fmt(median(pooled[task,width,method]))+"$")
        rows.append(row)
    write_table_rows("shallow",rows)


if __name__ == "__main__":
    optimizer()
    recurrent()
    shallow()
