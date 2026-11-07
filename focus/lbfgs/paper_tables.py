"""Generate the manuscript appendix from measured matched solver checkpoints.

Run with Python from any working directory. The script only writes
paper/focus_lbfgs_tables.tex and has no dependencies outside the standard library.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = {
    "square": "Square",
    "sin": "Sine",
    "selfpower": "Positive self-power",
    "tanh": "Hyperbolic tangent",
    "add": "Addition",
    "subtract": "Subtraction",
    "sum": "Running sum",
    "sort": "Sort four",
    "fib": "Fibonacci",
    "tri": "Triangular",
}
METHODS = ("gd", "gd_lbfgs", "adam_lbfgs", "lbfgs_xavier")
METHOD_NAMES = {
    "gd": "Gradient descent",
    "gd_lbfgs": r"Gradient descent $\to$ L-BFGS",
    "adam800": "Adam (800 updates)",
    "adam_lbfgs": r"Adam $\to$ L-BFGS",
    "lbfgs_xavier": "L-BFGS from Xavier",
}
SPLIT_NAMES = {
    "outer": "the outer test",
    "dev": "the central or early-index test",
    "train": "the observed training set",
}
SPLIT_DETAILS = {
    "outer": r"Continuous inputs have magnitudes from 3 to 10. Integer sequences use indices 33--64.",
    "dev": r"Continuous inputs have central magnitudes up to 3. Integer sequences use unseen indices 9--32.",
    "train": r"Each setting has 12 labeled examples, or nine integer indices 0--8 for a number sequence.",
}


def measured_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def setting(row):
    return row["task"], int(row["size"]), int(row["seed"])


def number(value):
    if value == 0:
        return "0"
    if 0.001 <= abs(value) < 1000:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def tabular_row(values):
    return " & ".join(map(str, values)) + r" \\"


def overview(rows, endpoint_rows, certificates):
    wsp = {
        setting(row): float(row["raw_rmse"])
        for row in endpoint_rows
        if row["method"] == "examples" and row["split"] == "outer"
    }
    # The one-hinge reference is independent of all numerical optimizers.
    one_hinge = {
        setting(row): float(row["real_optimum"]) for row in certificates
    }
    assert len(wsp) == 87 and len(one_hinge) == 12
    lines = [r"""\begin{table}[htbp]\centering\small
\caption{Optimizer outcomes relative to all 87 matched settings. Near the optimum means an absolute loss gap of at most $10^{-10}$ to the independently certified minimum in the 12 one-hinge settings. The last column counts settings with strictly smaller outer raw RMSE for the first examples-only WSP response than for the named fit. Each setting contributes one comparison.}
\label{tab:limitedmemory_overview}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Observed training set} & Near the & Entire outer & WSP lower\\
\cmidrule(lr){2-3}
Method & Median loss & Loss $\leq 10^{-10}$ & optimum /12 & set exact /87 & RMSE /87\\
\midrule"""]
    for method in ("gd", "gd_lbfgs", "adam800", "adam_lbfgs", "lbfgs_xavier"):
        outer = [r for r in rows if r["split"] == "outer" and r["method"] == method]
        assert len(outer) == 87
        losses = [float(r["train_loss"]) for r in outer]
        near_optimum = sum(
            abs(float(r["train_loss"]) - one_hinge[setting(r)]) <= 1e-10
            for r in outer if setting(r) in one_hinge
        )
        exact_sets = sum(int(r["exact_count"]) == int(r["n_examples"]) for r in outer)
        wsp_wins = sum(wsp[setting(r)] < float(r["raw_rmse"]) for r in outer)
        lines.append(tabular_row((
            METHOD_NAMES[method], number(median(losses)), sum(x <= 1e-10 for x in losses),
            near_optimum, exact_sets, wsp_wins,
        )))
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines)


def sizes_table(groups, split):
    lines = [
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Exact counts and raw RMSE for gradient optimizers on " + SPLIT_NAMES[split] +
        ". " + SPLIT_DETAILS[split] +
        r" Each entry is a median over the same three starts. Exact counts require every output of a stream"
        r" or list, or one number at a given integer index. RMSE includes every scalar output in the target's units.}",
        rf"\label{{tab:limitedmemory_{split}}}",
        r"""\setlength{\tabcolsep}{2pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrr *{8}{r}@{}}
\toprule
 & & & \multicolumn{2}{c}{Gradient descent} &
 \multicolumn{2}{c}{\shortstack{Gradient descent\\then L-BFGS}} &
 \multicolumn{2}{c}{\shortstack{Adam\\then L-BFGS}} &
 \multicolumn{2}{c}{\shortstack{L-BFGS\\from Xavier}}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Target & Size & Slots & Exact & RMSE & Exact & RMSE & Exact & RMSE & Exact & RMSE\\
\midrule""",
    ]
    row_count = 0
    for task, label in TASKS.items():
        sizes = sorted({size for t, size, _, _ in groups if t == task})
        for position, size in enumerate(sizes):
            slots = int(groups[task, size, split, METHODS[0]][0]["params"])
            cells = [label if position == 0 else "", size, slots]
            for method in METHODS:
                group = groups[task, size, split, method]
                assert {int(row["seed"]) for row in group} == {164, 317, 583} and len(group) == 3
                totals = {int(row["n_examples"]) for row in group}
                assert len(totals) == 1
                hits = int(median(int(row["exact_count"]) for row in group))
                rmse = median(float(row["raw_rmse"]) for row in group)
                cells.extend((f"{hits}/{totals.pop()}", number(rmse)))
            lines.append(tabular_row(cells))
            row_count += 1
    assert row_count == 29
    lines.extend((r"\bottomrule", r"\end{tabular}}", r"\end{table}"))
    return "\n".join(lines)


def main():
    rows = measured_rows(HERE / "scores.csv")
    primary = measured_rows(ROOT / "focus/endpoints/endpoints.csv")
    certificates = measured_rows(ROOT / "focus/refinement/certified_optima.csv")
    groups = defaultdict(list)
    for row in rows:
        if row["method"] in METHODS:
            groups[row["task"], int(row["size"]), row["split"], row["method"]].append(row)
    assert len(rows) == 87 * 6 * 3
    parts = [r"""\subsection{Gradient descent and limited-memory optimization}\label{app:limitedmemory}
This subsection retains the original training-only gradient comparisons. Its early integer diagnostic uses the original indices 9--32. For a comparison with iterative WSP on untouched indices 17--32, see Table~\ref{tab:sizes_dev}. These fits use the same nonconvex networks, initial weights, training observations and loss as the main comparison. Plain full-batch gradient descent starts from Xavier and takes at most 800 updates. At each update it tries step size one and halves it until the training loss decreases sufficiently. Limited-memory Broyden--Fletcher--Goldfarb--Shanno (L-BFGS) uses the most recent ten pairs of changes in weights and gradients to approximate curvature. We run it with no parameter bounds for up to 1,000 iterations. Its reported endpoint has the lowest training loss among the supplied starting point and its accepted iterates. Selection uses training loss throughout.

We start L-BFGS from three separate points: the original Xavier weights, the gradient-descent endpoint, or the endpoint after 800 adaptive moment estimation (Adam) updates. The Adam checkpoint in Table~\ref{tab:limitedmemory_overview} is scored before refinement. The separate Adam--BFGS baseline applies full-memory BFGS after Adam. Each additional fit uses derivatives of the empirical loss obtained by backpropagation through the network. The complete records include optimizer termination, per-setting scores and fitted weights.

Tables~\ref{tab:limitedmemory_outer}, \ref{tab:limitedmemory_dev} and~\ref{tab:limitedmemory_train} give raw root mean squared error (RMSE) and exact matches at each tested size. Tables~\ref{tab:sizes_outer}--\ref{tab:sizes_train} give WSP and gradient results on the same outer and training inputs and on the current held-out central or early split, which uses indices 17--32 for the two integer sequences. Size denotes hidden width or the number of residual blocks for sorting. Slots count trainable scalar coefficients.""",
             overview(rows, primary, certificates)]
    parts.extend(sizes_table(groups, split) for split in SPLIT_NAMES)
    path = ROOT / "paper/focus_lbfgs_tables.tex"
    path.write_text("\n\n".join(parts) + "\n")
    print(f"{path.name}: 29 task–architecture rows on each of three splits and five optimizer summaries")


if __name__ == "__main__":
    main()
