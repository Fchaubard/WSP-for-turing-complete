"""Self-contained LaTeX tables and a matched-comparison figure."""
METHODS = {
    "examples": "One-shot, examples only",
    "iter_examples": "Iterative, examples only",
    "formula": "One-shot, formula and examples",
    "iter_formula": "Iterative, formula and examples",
    "gd": "Gradient descent",
    "gd_lbfgs": "Gradient descent then L-BFGS",
    "adam_bfgs": "Adam then full BFGS",
    "ls_xavier": "Trust-region least squares",
}
NAMES = {"outer": "extrapolation", "dev": "central and early-index generalization",
         "train": "observed training examples"}
DOMAINS = {
    "outer": (r"Continuous inputs have training magnitudes up to 3 and outer values in $[-10,-3]$ or $[3,10]$, "
              r"with positive self-power restricted to $[3,10]$. Sequences test indices 33--64."),
    "dev": (r"Fresh continuous inputs have magnitudes up to 3. Number sequences use held-out indices "
            r"17--32. The central test excludes all revision feedback inputs."),
    "train": r"Observed data contain 12 examples per sampled task or the nine sequence indices 0--8.",
}


def cell_row(cells):
    return " & ".join(map(str, cells)) + r" \\"


def number(x, digits=3):
    if x == 0:
        return "0"
    if 0.001 <= abs(x) < 1000:
        return f"{x:.{digits}g}"
    mantissa, exp = f"{x:.{digits - 1}e}".split("e")
    return rf"${mantissa.rstrip('0').rstrip('.')}\!\times\!10^{{{int(exp)}}}$"


def table_value(text, best):
    # Bold math mode covers the multiplication sign and exponent in scientific notation.
    return r"{\bfseries\boldmath " + str(text) + "}" if best else str(text)


def proof_label(layout):
    return r"\textbf{LAYOUT PROOF --- UNMEASURED.} " if layout else ""


def sizes_table(medians, tasks, methods, split, selected, layout):
    if selected:
        start = "Outer-test results at one prespecified size per task. "
        label = "tab:focusouter"
    else:
        start = ("Outer extrapolation for weight-space programming (WSP) and gradient methods. " if split == "outer" else
                 "Weight-space programming (WSP) and gradient methods on " + NAMES[split] + ". ")
        label = "tab:sizes_" + split
    caption = (proof_label(layout) + start + DOMAINS[split] +
               r" Entries are three-start medians. Exact counts whole examples out of $N$. Root mean squared "
               r"error (RMSE) is in output units. Bold marks the highest positive exact count and lowest RMSE "
               r"across both panels, including all ties before rounding. Size counts hidden units, state "
               r"coordinates or residual blocks. Slots count trainable coefficients. Adam is adaptive moment "
               r"estimation. L-BFGS is limited-memory Broyden--Fletcher--Goldfarb--Shanno (BFGS).")
    lines = [r"\begin{table}[htbp]\centering", r"\caption{" + caption + "}",
             rf"\label{{{label}}}", r"\begingroup\fontsize{7.3}{7.9}\selectfont",
             r"\setlength{\tabcolsep}{1.3pt}\renewcommand{\arraystretch}{0.93}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr *{4}{rr}@{}}", r"\toprule"]
    sections = (
        ("A. Weight-space programming", methods[:4], ["One-shot", "Iterative", "One-shot", "Iterative"],
         ["Examples only", "Examples only", "Formula and examples", "Formula and examples"]),
        ("B. Gradient methods", methods[4:], ["Gradient descent", "Gradient descent", "Adam", "Trust-region"],
         ["Full-batch", "then L-BFGS", "then full BFGS", "least squares"]),
    )
    for panel, panel_methods, upper, lower in sections:
        if panel.startswith("B"):
            lines.extend((r"\addlinespace[3pt]", r"\midrule"))
        lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + panel + "}} & " +
                     " & ".join(r"\multicolumn{2}{c}{\shortstack{" + t + r"\\" + b + "}}" for t,b in zip(upper,lower)) +
                     r"\\")
        lines.append(r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}")
        lines.append(cell_row(("Target", "Size", "Slots", r"$N$", *(["Exact", "RMSE"]*4))))
        lines.append(r"\midrule")
        for ti, (task, (name, all_sizes, chosen)) in enumerate(tasks.items()):
            for si, size in enumerate([chosen] if selected else all_sizes):
                all_results = {m: medians[task, size, split, m] for m in methods}
                highest_count = max(result["exact_count"] for result in all_results.values())
                lowest_error = min(result["raw_rmse"] for result in all_results.values())
                rounded_best = number(lowest_error)
                collisions = [result["raw_rmse"] for result in all_results.values()
                              if result["raw_rmse"] > lowest_error and number(result["raw_rmse"]) == rounded_best]
                # One extra digit distinguishes close results when it will resolve the printed tie.
                digits = 4 if collisions and all(number(x, 4) != number(lowest_error, 4) for x in collisions) else 3
                group = [all_results[m] for m in panel_methods]
                if si == 0 and ti:
                    lines.append(r"\addlinespace[0.7pt]")
                cells = [name if si == 0 else "", size, group[0]["params"], group[0]["n"]]
                for g in group:
                    exact, error = g["exact_count"], g["raw_rmse"]
                    cells.extend((table_value(exact, exact > 0 and exact == highest_count),
                                  table_value(number(error, digits if number(error) == rounded_best else 3),
                                              error == lowest_error)))
                lines.append(cell_row(cells))
    lines.extend((r"\bottomrule", r"\end{tabular*}\endgroup", r"\end{table}", ""))
    return "\n".join(lines)


def endpoint_table(summary, layout):
    caption = (proof_label(layout) +
        r"Final prediction errors on the same 87 settings. Each error column gives the median scaled "
        r"root mean squared error (RMSE) across settings. For each prediction, the error is divided by "
        r"the larger of one and the absolute target value. The final column counts settings in which "
        r"every extrapolation output was exactly correct. Bold marks the best value in each column, including ties.")
    lines = [r"\begin{table}[htbp]\centering",r"\caption{" + caption + "}",r"\label{tab:paired_methods}",
             r"\small\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule",
             r"&\multicolumn{3}{c}{Median scaled RMSE}&\shortstack{Fully exact}\\",
             r"\cmidrule(lr){2-4}",
             cell_row(("Method", "Training", "Central / early", "Extrapolation", "outer sets (of 87)")),
             r"\midrule"]
    splits = ("train", "dev", "outer")
    best_error = {} if layout else {s: min(row[s] for row in summary.values()) for s in splits}
    most_exact = max(row["exact_sets"] for row in summary.values())
    for label, group in [
        ("Weight-space programming (WSP)", ("examples", "iter_examples", "formula", "iter_formula")),
        ("Gradient methods", ("gd", "gd_lbfgs", "adam_bfgs", "ls_xavier")),
    ]:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{" + label + r"}}\\")
        for method in group:
            record = summary[method]
            errors = ["--" if layout else table_value(number(record[s]), record[s] == best_error[s])
                      for s in splits]
            exact = table_value(record["exact_sets"], record["exact_sets"] == most_exact)
            lines.append(cell_row((METHODS[method], *errors, exact)))
        lines.append(r"\addlinespace[2pt]")
    lines.extend((r"\bottomrule\end{tabular*}",r"\end{table}",""))
    return "\n".join(lines)


def figure_tex(layout):
    return (r"\begin{figure}[htbp]\centering" + "\n" +
            r"\includegraphics[width=\linewidth]{iterative_figure.pdf}" + "\n" +
            r"\caption{" + proof_label(layout) +
            r"Extrapolation error after optimization. Each bar is the median across all 87 settings (29 task and architecture "
            r"pairs with three random starts each). Both weight-space programming (WSP) methods received "
            r"examples alone. Iterative WSP also received training and development feedback. Each output "
            r"error is divided by the larger of one and the target magnitude before computing root mean "
            r"squared error for that setting. Table~\ref{tab:focusouter} gives errors in the original "
            r"output units for individual tasks.}\label{fig:iterative_endpoints}" + "\n" +
            r"\end{figure}" + "\n")


def draw(path, summary, layout):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    methods = ("examples", "iter_examples", "gd", "gd_lbfgs")
    labels = ("One-shot WSP", "Iterative WSP", "Gradient descent", "Gradient descent + L-BFGS")
    # Layout proofs show fixed placeholder shapes. Only the release path displays scores.
    values = [0.15, 0.45, 0.75, 0.60] if layout else [summary[m]["outer"] for m in methods]
    ink, neutral = "#1d3039", "#53656e"
    fig = plt.figure(figsize=(7.2, 2.92), facecolor="white")
    fig.text(0.02, 0.935, "Prediction error outside the training range",
             fontsize=14, fontweight="bold", color=ink, va="top")
    fig.text(0.02, 0.82, "Examples only  ·  Median across 87 combinations of task, network size and random start",
             fontsize=8.9, color=neutral, va="top")

    ax = fig.add_axes((0.31, 0.30, 0.66, 0.45))
    positions = (3.2, 2.2, 1, 0)
    limit = max(1, max(values))
    ax.barh(positions, values, height=.53, color="#246477", zorder=3)
    ax.set_xlim(0, limit * 1.12)
    ax.set_ylim(-.50, 3.70)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=10.3, colors=ink)
    ticks = [v for v in MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]).tick_values(0, limit) if 0 <= v <= limit]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.tick_params(axis="x", length=0, pad=5, labelsize=8.8, colors=neutral)
    ax.grid(axis="x", linewidth=.6, color="#dce3e5")
    ax.axvline(0, color="#aab9be", linewidth=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Scaled root mean squared error (lower is better)", fontsize=9.1,
                  labelpad=6, color=ink)
    if layout:
        fig.text(.54, .47, "LAYOUT PROOF — UNMEASURED", color="#922e2e", rotation=15,
                 fontsize=16, ha="center")
    else:
        for y, value in zip(positions, values):
            ax.text(value + .014 * limit, y, f"{value:.3f}", va="center", color=ink,
                    fontsize=10.2, fontweight="bold")
    fig.text(.02, .042, "Each error is divided by the larger of 1 and the absolute target value.",
             fontsize=8.4, color=neutral, va="bottom")
    fig.savefig(path, metadata={"Title": "Prediction error outside the training range"})
    plt.close(fig)
