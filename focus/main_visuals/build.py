"""Draw fixed-case predictions from archived weights, without fitting or model calls."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "focus/iterative_eval"))
from measure import datasets, measures, observed, raw_prediction, reference

TASKS = ("square", "sin", "tanh")
LABELS = {"square": "Square", "sin": "Sine", "tanh": "Hyperbolic tangent"}
SIZE, SEED, ARM = 16, 164, "examples"
METHODS = ("examples", "iter_examples", "gd_lbfgs")
NAMES = {
    "examples": "One-shot WSP",
    "iter_examples": "Iterative WSP",
    "gd_lbfgs": "Gradient descent + L-BFGS",
}
COLORS = {"examples": "#176E78", "iter_examples": "#D46A44", "gd_lbfgs": "#7464A1"}
STYLES = {"examples": "-", "iter_examples": "--", "gd_lbfgs": "-."}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, records):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    source_paths = [ROOT / "focus/lbfgs/weights.npz",
                    ROOT / "focus/iterative_eval/comparators.csv",
                    ROOT / "focus/iterative_eval/endpoints.csv",
                    ROOT / "focus/iterative_eval/extended.csv"]
    trajectories = {}
    for path in (ROOT / "focus/iterative/trajectories").glob("*.json"):
        record = json.loads(path.read_text())
        if record["task"] in TASKS and record["seed"] == SEED and record["arm"] == ARM:
            trajectories[record["task"]] = record["states"][str(SIZE)]
            source_paths.append(path)
    assert len(trajectories) == len(TASKS)
    canonical = {}
    for path in source_paths[1:4]:
        for row in read_csv(path):
            if row["task"] in TASKS and int(row["size"]) == SIZE and int(row["seed"]) == SEED:
                canonical[row["task"], row["method"], row["split"]] = row
    weights = {}
    with np.load(source_paths[0]) as gradients:
        for task, trajectory in trajectories.items():
            weights[task] = {
                "examples": np.asarray(trajectory["initial"], dtype=float),
                "iter_examples": np.asarray(trajectory["weights"], dtype=float),
                "gd_lbfgs": gradients[f"{task}_{SIZE}_{SEED}_gd_lbfgs"].copy(),
            }
    checks, metric_rows, curve_rows, observed_rows = [], [], [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelcolor": "#283C47", "text.color": "#283C47",
                         "xtick.color": "#455B66", "ytick.color": "#455B66",
                         "axes.edgecolor": "#A0ABB1", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0))
    x = np.linspace(-10.0, 10.0, 1001)
    for ax, task in zip(axes, TASKS):
        truth = reference(task, x)
        pred = {method: raw_prediction(task, SIZE, weights[task][method], x)
                for method in METHODS}
        for i, value in enumerate(x):
            curve_rows.append(dict(task=task, size=SIZE, seed=SEED, x=float(value),
                                   target=float(truth[i]),
                                   **{method: float(pred[method][i]) for method in METHODS}))
        ox, oy = observed(task, SEED)
        observed_rows.extend(dict(task=task, seed=SEED, x=float(a), target=float(b))
                             for a, b in zip(ox, oy))
        # Uniform central shading identifies the supplied range, not a confidence band.
        ax.axvspan(-3, 3, color="#E8EDF0", zorder=-5)
        for boundary in (-3, 3):
            ax.axvline(boundary, color="#C1CCD1", lw=0.6, zorder=-4)
        for method in METHODS:
            ax.plot(x, pred[method], color=COLORS[method], linestyle=STYLES[method],
                    lw=1.55, zorder=2, label=NAMES[method])
        ax.plot(x, truth, color="#263842", lw=1.15, linestyle=(0, (1.7, 2.1)),
                zorder=3, label="Target")
        ax.scatter(ox, oy, s=7, color="#263842", edgecolor="white", linewidth=0.25,
                   zorder=4, label="Training examples")
        ax.set(title=LABELS[task], xlim=(-10, 10), xlabel="Input",
               xticks=[-10, -3, 0, 3, 10])
        ax.tick_params(length=2.5, width=0.6, pad=3)
        ax.grid(axis="y", color="#DEE5E9", linewidth=0.5, zorder=-6)
        ax.set_axisbelow(True)
        ax.margins(y=0.1)
        for method in METHODS:
            for split in ("train", "feedback_dev", "dev", "outer"):
                actual = measures(task, SIZE, weights[task][method], *datasets(task, SEED, None)[split])
                expected = canonical[task, method, split]
                assert np.isclose(actual["raw_rmse"], float(expected["raw_rmse"]), rtol=4e-13, atol=1e-30)
                assert actual["exact_count"] == int(expected["exact_count"])
                checks.append([task, method, split])
                metric_rows.append(dict(task=task, size=SIZE, seed=SEED, method=method,
                                        split=split, raw_rmse=actual["raw_rmse"],
                                        split_loss=actual["split_loss"],
                                        exact_count=actual["exact_count"], n_examples=actual["n_examples"]))
        assert float(canonical[task, "iter_examples", "feedback_dev"]["split_loss"]) < float(canonical[task, "examples", "feedback_dev"]["split_loss"])
        assert float(canonical[task, "iter_examples", "outer"]["raw_rmse"]) > float(canonical[task, "examples", "outer"]["raw_rmse"])
    axes[0].set_ylabel("Prediction")
    handles = [Line2D([0], [0], color="#263842", lw=1.2, linestyle=(0, (1.7, 2.1)), label="Target")]
    handles.extend(Line2D([0], [0], color=COLORS[m], linestyle=STYLES[m], lw=1.7, label=NAMES[m]) for m in METHODS)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=4,
               frameon=False, fontsize=8.1, handlelength=2.2, columnspacing=1.4)
    fig.legend(handles=[Patch(facecolor="#E8EDF0", edgecolor="#C1CCD1", linewidth=.6,
                              label="Training/development inputs"),
                        Line2D([0], [0], color="#263842", marker="o", linestyle="", markersize=3,
                               label="12 training examples")],
               loc="lower center", bbox_to_anchor=(.5, -.012), ncol=2, frameon=False,
               fontsize=7.8, handlelength=1.6, columnspacing=2.0)
    fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.89, wspace=.31)
    fig.savefig(HERE / "prediction_curves.pdf", metadata={"Title": "Frozen network predictions inside and outside the observed interval"})
    fig.savefig(HERE / "prediction_curves.png", dpi=250)
    plt.close(fig)
    write_csv("prediction_curves.csv", curve_rows)
    write_csv("observed_points.csv", observed_rows)
    write_csv("verified_metrics.csv", metric_rows)
    audit = dict(selection=dict(tasks=TASKS, size=SIZE, seed=SEED, arm=ARM,
                                rule="First recorded random start and largest tested scalar width; illustrative fixed cases, not median predictions"),
                 verified_saved_metric_rows=len(checks),
                 checked_fields=["raw_rmse", "exact_count"],
                 source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                 evaluation="Offline replay of frozen weights; no fitting or model calls")
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    caption = r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{prediction_curves.pdf}
\caption{\textbf{Improved central fit can coexist with a worse extrapolation rule.} Each panel uses a 16-unit scalar network with 50 coefficients, the first recorded random start (164), and examples-only WSP. Shading marks the training and development interval; dots are the twelve training pairs. Curves are evaluations of the frozen networks, not averages over runs. These illustrative regressions show how similar central predictions conceal different outer continuations; they do not estimate how often this happens. Tables~\ref{tab:focusouter} and~\ref{tab:sizes_outer} summarize errors across all three starts.}\label{fig:prediction-curves}
\end{figure}
"""
    (HERE / "prediction_curves.tex").write_text(caption)
    print(f"Verified {len(checks)} archived metric rows and exported three fixed-case panels.")


if __name__ == "__main__":
    main()
