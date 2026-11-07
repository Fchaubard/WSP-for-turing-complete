"""Rank complete 87-setting endpoints by the sum of ten task-specific mean raw RMSEs."""
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ("square", "sin", "selfpower", "tanh", "add", "subtract", "sum", "sort", "fib", "tri")
METHODS = {
    "examples": "One-shot WSP, examples only",
    "formula": "One-shot WSP, formula + examples",
    "iter_examples": "Iterative WSP, examples only",
    "iter_formula": "Iterative WSP, formula + examples",
    "gd": "Gradient descent",
    "gd_lbfgs": "Gradient descent + L-BFGS",
    "adam800": "Adam, 800 updates",
    "adam_lbfgs": "Adam + L-BFGS",
    "adam_bfgs": "Adam + full BFGS",
    "lbfgs_xavier": "L-BFGS from Xavier",
    "ls_xavier": "Trust-region least squares",
    "pool_gd": "Gradient descent,\nfull feedback pool",
    "pool_gd_lbfgs": "Gradient descent + L-BFGS,\nfull feedback pool",
    "wsp_ls": "One-shot WSP + least squares,\nexamples only",
}
SOURCES = (
    ROOT / "output/iterative-wsp-endpoints.csv",
    ROOT / "focus/lbfgs/scores.csv",
    ROOT / "focus/refinement/before_after.csv",
)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def collect():
    rows = read(SOURCES[0])
    rows.extend(r for r in read(SOURCES[1]) if r["method"] == "adam800")
    rows.extend(dict(task=r["task"], size=r["size"], seed=r["seed"], method="wsp_ls",
                     split=r["split"], raw_rmse=r["after_rmse"], n_examples=r["n_examples"],
                     exact_count=r["after_exact_count"]) for r in read(SOURCES[2]))
    cells = {}
    for row in rows:
        if row["split"] != "outer":
            continue
        key = row["method"], row["task"], int(row["size"]), int(row["seed"])
        if key in cells or row["method"] not in METHODS:
            raise ValueError(f"Unexpected or repeated endpoint {key}")
        error = float(row["raw_rmse"])
        if not math.isfinite(error) or error < 0:
            raise ValueError(f"A raw RMSE is not a finite nonnegative value at {key}")
        cells[key] = error
    expected = {key[1:] for key in cells if key[0] == "examples"}
    if len(expected) != 87 or len(cells) != 87 * len(METHODS):
        raise ValueError("The ranking requires all 87 settings for every method")
    for method in METHODS:
        if {key[1:] for key in cells if key[0] == method} != expected:
            raise ValueError(f"Mismatched settings for {method}")
    task_rows, total_rows = [], []
    for method in METHODS:
        contributions = []
        all_errors = []
        for task in TASKS:
            errors = [v for (m, t, _, _), v in cells.items() if (m, t) == (method, task)]
            assert len(errors) == (6 if task == "sum" else 9)
            mean = math.fsum(errors) / len(errors)
            contributions.append(mean)
            all_errors.extend(errors)
            task_rows.append(dict(method=method, task=task, settings=len(errors),
                                  mean_raw_rmse=mean, sum_raw_rmse=math.fsum(errors)))
        total = math.fsum(contributions)
        # Independent high-precision aggregation checks the ordering and the reported scalar.
        with localcontext() as context:
            context.prec = 270
            reference = sum((sum((Decimal.from_float(cells[m, t, w, seed])
                                  for m, t, w, seed in cells if m == method and t == task), Decimal(0))
                             / (6 if task == "sum" else 9)) for task in TASKS)
        if not math.isclose(total, float(reference), rel_tol=5e-16):
            raise ValueError(f"Independent sum disagrees for {method}")
        largest = max(range(10), key=lambda i: contributions[i])
        total_rows.append(dict(method=method, method_name=METHODS[method].replace("\n", " "),
                               tasks=10, settings=87, sum_task_mean_raw_rmse=total,
                               sum_87_raw_rmse=math.fsum(all_errors),
                               dominant_task=TASKS[largest], dominant_fraction=contributions[largest]/total))
    total_rows.sort(key=lambda r: r["sum_task_mean_raw_rmse"])
    total_rows = [dict(rank=i, **row) for i, row in enumerate(total_rows, 1)]
    write(HERE / "task_contributions.csv", task_rows)
    write(HERE / "ranking.csv", total_rows)
    write(HERE / "per_setting.csv", [dict(method=m, task=t, size=w, seed=s, raw_rmse=v)
                                     for (m, t, w, s), v in sorted(cells.items())])
    return total_rows


def score_text(value, precise=False):
    digits = 8 if precise else 3
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def draw(rows, full=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    ink, grey, color = "#17313C", "#51656E", "#2B6E80"
    fig = plt.figure(figsize=(7.1, 5.45), facecolor="white")
    fig.text(.012, .965, "Summed extrapolation error across all ten tasks", size=12.4,
             weight="bold", color=ink, va="top")
    fig.text(.012, .908, "Mean raw RMSE within each task, then sum across tasks. Lower is better.",
             size=8.0, color=grey, va="top")
    ybottom, yheight = .16, .68
    labels = fig.add_axes((.012, ybottom, .412, yheight))
    ax = fig.add_axes((.425, ybottom, .345, yheight))
    totals = fig.add_axes((.80, ybottom, .195, yheight))
    for axis in (labels, ax, totals):
        axis.set_ylim(-.7, len(rows)-.3)
        axis.invert_yaxis()
    labels.axis("off")
    totals.axis("off")
    xmin = 1e8
    xmax = 1e234 if full else 1e34
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", length=0, labelsize=8, pad=5, colors=grey)
    ax.set_xticks([1e10, 1e65, 1e120, 1e175, 1e230] if full else [1e9,1e15,1e21,1e27,1e33])
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(axis="x", color="#DFE7EA", linewidth=.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Sum of task RMSEs (logarithmic scale)", size=8.3, color=ink, labelpad=8)
    labels.text(0, -.86, "Method", size=9.2, weight="bold", color=ink)
    totals.text(.0, -.86, "Total error", size=9.2, weight="bold", color=ink)
    for i, record in enumerate(rows):
        value = record["sum_task_mean_raw_rmse"]
        labels.text(.0, i, str(record["rank"]), size=8.4, color=grey, va="center")
        labels.text(.085, i, METHODS[record["method"]], size=8.9, color=ink, va="center", linespacing=1.05)
        shown = min(value, xmax)
        ax.barh(i, shown-xmin, left=xmin, height=.48, color=color, zorder=3)
        if value > xmax:
            ax.plot(xmax, i, marker=">", markersize=8, color=color, clip_on=False, zorder=4)
        totals.text(.0, i, score_text(value, record["rank"] <= 2), size=8.8, color=ink, va="center")
    footer = ("All 14 complete-grid methods · 87 shared task, size and start combinations" if full else
              "Arrows exceed the displayed range. The full-range chart is in the appendix.")
    fig.text(.012, .035, footer, fontsize=7.8, color=grey, va="bottom")
    name = "holistic_full_range" if full else "holistic_main"
    fig.savefig(HERE / (name+".pdf"), metadata={"Title": "Sum of raw extrapolation RMSE across ten tasks"})
    fig.savefig(HERE / (name+".png"), dpi=220)
    plt.close(fig)


def tex():
    common = (r"Each method is evaluated on the same 87 task, architecture and start combinations. "
              r"For each task, we average raw root mean squared error (RMSE) across its tested sizes and "
              r"three starts, then sum those ten task values. Each task contributes one mean; output scales are not normalized. "
              r"Ranks use unrounded totals. The two iterative WSP totals agree to six significant figures. "
              r"The full feedback-pool controls use additional labels, and WSP followed by least squares "
              r"receives an additional numerical stage. Adam means adaptive moment estimation; BFGS means Broyden--Fletcher--Goldfarb--Shanno. ")
    (HERE / "holistic_main.tex").write_text(
        r"\begin{figure}[htbp]\centering"+"\n"+r"\includegraphics[width=\linewidth]{holistic_main.pdf}"+"\n"+
        r"\caption{\textbf{Iterative WSP has the lowest summed raw extrapolation error in this benchmark.} "+common+
        r"Fibonacci supplies 99.8\% of the examples-only one-shot total; positive self-power supplies 78.1\% of its iterative counterpart. "
        r"The horizontal axis is logarithmic. Arrows mark the two totals beyond $10^{34}$, with their actual "
        r"values printed at the right. Figure~\ref{fig:holisticfull} shows the entire range.}"
        r"\label{fig:holisticraw}"+"\n"+r"\end{figure}"+"\n")
    (HERE / "holistic_full_range.tex").write_text(
        r"\begin{figure}[p]\centering"+"\n"+r"\includegraphics[width=\linewidth]{holistic_full_range.pdf}"+"\n"+
        r"\caption{The full range of summed raw extrapolation errors. "+common+
        r"Large-magnitude targets and individual divergent runs strongly affect this raw sum. "
        r"Fibonacci contributes more than 99.8\% of examples-only one-shot WSP's total. "
        r"Positive self-power contributes about 78.1\% of its iterative counterpart's total.}"
        r"\label{fig:holisticfull}"+"\n"+r"\end{figure}"+"\n")


def restore():
    # Preserve the previous four-bar chart and its exact caption and statistics.
    shutil.copyfile(ROOT / "focus/iterative_tables/iterative_figure.pdf", HERE / "prior_scaled_median.pdf")
    caption=(ROOT / "focus/iterative_tables/figure1.tex").read_text()
    caption=caption.replace("iterative_figure.pdf", "prior_scaled_median.pdf").replace("fig:iterative_endpoints", "fig:priorscaledmedian")
    (HERE / "prior_scaled_median.tex").write_text(caption)
    shutil.copyfile(ROOT / "focus/endpoints/figure1.pdf", HERE / "prior_task_comparisons.pdf")
    caption=(ROOT / "focus/endpoints/figure1.tex").read_text().replace("figure1.pdf", "prior_task_comparisons.pdf").replace("[t]", "[htbp]")
    (HERE / "prior_task_comparisons.tex").write_text(caption)
    # The earlier iterative PDF is immutable. Preserve its vector chart with a literal PDF crop.
    source = ROOT / "output/pdf/weight-space-programming-2026-11-06-iterative.pdf"
    # Letter page coordinates: x 108--504, y 302--524 from the top (pdftotext -bbox-layout).
    code = r"""\documentclass{article}
\usepackage[paperwidth=396bp,paperheight=222bp,margin=0pt]{geometry}
\usepackage{graphicx}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\hoffset=-1in\voffset=-1in
\setbox0=\hbox{\includegraphics[page=5,trim=108bp 268bp 108bp 302bp,clip,width=396bp,height=222bp]{source.pdf}}
\shipout\box0
\end{document}
"""
    with tempfile.TemporaryDirectory(prefix="wsp-old-plot-") as temp:
        folder=Path(temp)
        shutil.copyfile(source,folder/"source.pdf")
        (folder/"crop.tex").write_text(code)
        result=subprocess.run(["pdflatex","-interaction=batchmode","-halt-on-error","crop.tex"],cwd=folder,capture_output=True)
        if result.returncode:
            raise RuntimeError((folder/"crop.log").read_text()[-1500:])
        shutil.copyfile(folder/"crop.pdf",HERE/"prior_iterative_comparisons.pdf")
    (HERE / "prior_iterative_comparisons.tex").write_text(
        r"\begin{figure}[htbp]\centering"+"\n"+r"\includegraphics[width=\linewidth]{prior_iterative_comparisons.pdf}"+"\n"+
        r"\caption{The earlier paired comparison in its original format. Each bar divides the same 87 "
        r"settings according to whether the named WSP method has lower error, equal error, or higher error "
        r"than the stated comparison. The left panel uses the scaled training loss and the right uses raw "
        r"extrapolation RMSE. These counts show the direction of paired changes. Figures~\ref{fig:holisticraw} "
        r"and~\ref{fig:holisticfull} show their magnitudes through the sum.}\label{fig:prioriterativepaired}"+"\n"+
        r"\end{figure}"+"\n")


def main():
    rows=collect()
    draw(rows)
    draw(rows, full=True)
    tex()
    restore()
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    (HERE/"audit.json").write_text(json.dumps(dict(methods=14,settings_per_method=87,task_mean_sum_checked_with_decimal=True,sources=sources),indent=2)+"\n")
    print("Ranked 14 complete-grid methods; 1,218 endpoint values verified.")


if __name__ == "__main__":
    main()
