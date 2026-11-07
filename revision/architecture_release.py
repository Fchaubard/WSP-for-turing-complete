"""Build publication summary and downloadable records from the measured studies."""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
HOLISTIC = ROOT / "focus/holistic_plots"


def scientific(value):
    mantissa, exponent = f"{float(value):.5e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def main():
    with (HOLISTIC / "ranking.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    lines = [
        r"\begin{table}[htbp]\centering\footnotesize",
        r"\caption{Raw-error totals and their dominant task. The total sums each task's mean RMSE across sizes and starts. The final two columns identify the largest task contribution and its share of that method's total. The graph-choosing proposals are evaluated separately in Appendix~\ref{app:architecture_search}.}\label{tab:holisticrank}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rlrlr@{}}\toprule",
        r"Rank & Method & Raw-error sum & Largest contribution & Share \\\midrule",
    ]
    task_names = {"selfpower": "Positive self-power", "fib": "Fibonacci"}
    for record in records:
        fields = [record["rank"], record["method_name"], scientific(record["sum_task_mean_raw_rmse"]),
                  task_names[record["dominant_task"]], f"{100 * float(record['dominant_fraction']):.2f}\\%"]
        lines.append(" & ".join(fields) + r"\\")
    lines.append(r"\bottomrule\end{tabular*}\end{table}")
    (ROOT / "paper/holistic_rank_table.tex").write_text("\n".join(lines) + "\n")
    for source, name in [
        (HOLISTIC / "ranking.csv", "all-optimizer-raw-rmse-ranking.csv"),
        (HOLISTIC / "task_contributions.csv", "optimizer-rmse-task-contributions.csv"),
        (HOLISTIC / "holistic_main.png", "figure-1-all-optimizers.png"),
        (ROOT / "focus/architecture_search/models.csv", "architecture-choice-models.csv"),
        (ROOT / "focus/architecture_search/scores.csv", "architecture-choice-errors.csv"),
        (ROOT / "focus/architecture_search/costs.csv", "architecture-choice-costs.csv"),
    ]:
        shutil.copyfile(source, OUTPUT / name)


if __name__ == "__main__":
    main()
