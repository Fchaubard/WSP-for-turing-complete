"""Publish the paired experiment and a portable archive of per-study error measurements."""
import csv
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "output"
output.mkdir(exist_ok=True)
condition = {
    "formula": ("Weight-space programming (WSP)", "Formula and examples"),
    "examples": ("Weight-space programming (WSP)", "Examples only"),
    "bp": ("Backpropagation with Adam and BFGS", "Examples only"),
    "xavier": ("Untrained Xavier initialization", "No fitting"),
}
paired = output / "ten-task-exact-and-rmse.csv"
with (root / "extension/writer/scores.csv").open() as src, paired.open("w", newline="") as dest:
    reader = csv.DictReader(src)
    fields = ["task", "seed", "method", "information", "architecture_size", "stored_weights",
              "split", "example_count", "exact_fraction", "raw_rmse", "mean_scaled_absolute_error"]
    writer = csv.writer(dest)
    writer.writerow(fields)
    for row in reader:
        writer.writerow([row["task"], row["seed"], *condition[row["arm"]], row["size"], row["slots"],
                         row["split"], row["n"], row["exact"], row["rmse"], row["mean_scaled"]])

primary = output / "nonconvex-endpoints.csv"
architectures = {
    "square": "Scalar ReLU MLP with linear skip", "sin": "Scalar ReLU MLP with linear skip",
    "selfpower": "Scalar ReLU MLP with linear skip", "tanh": "Scalar ReLU MLP with linear skip",
    "add": "Two-input ReLU MLP", "subtract": "Two-input ReLU MLP",
    "sum": "Driven linear recurrent network", "fib": "Autonomous linear recurrent network",
    "tri": "Autonomous linear recurrent network", "sort": "Four-wire residual ReLU network",
}
primary_names = {"examples": "Weight-space programming (WSP), Examples only",
                 "formula": "Weight-space programming (WSP), Formula and examples",
                 "adam_bfgs": "Adam followed by BFGS", "ls_xavier": "Trust-region least squares",
                 "xavier": "Initial Xavier weights"}
with (root / "focus/endpoints/endpoints.csv").open() as source, primary.open("w", newline="") as dest:
    reader = csv.DictReader(source)
    fields = ["task", "architecture", "size", "params", "seed", "method", "split", "split_description",
              "n_examples", "n_scalars", "exact_count", "exact_fraction", "raw_rmse", "train_loss"]
    writer = csv.DictWriter(dest, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in reader:
        if row["method"] in primary_names:
            row.update(architecture=architectures[row["task"]], method=primary_names[row["method"]])
            writer.writerow(row)

gradient_comparison = output / "gradient-descent-and-lbfgs.csv"
gradient_names = {
    "gd": "Full-batch gradient descent",
    "gd_lbfgs": "Gradient descent followed by L-BFGS",
    "adam800": "Adam (800 updates)",
    "adam_lbfgs": "Adam followed by L-BFGS",
    "lbfgs_xavier": "L-BFGS from Xavier",
}
with gradient_comparison.open("w", newline="") as dest:
    writer = csv.DictWriter(dest, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    with primary.open() as source:
        writer.writerows(csv.DictReader(source))
    with (root / "focus/lbfgs/scores.csv").open() as source:
        for row in csv.DictReader(source):
            if row["method"] in gradient_names:
                row.update(architecture=architectures[row["task"]], method=gradient_names[row["method"]])
                writer.writerow(row)

files = [gradient_comparison, primary, paired, root / "output/optimizer-runs.csv", root / "extension/convex/scores.csv",
         root / "extension/audit/lattice.json", root / "writer/task_scores.json",
         root / "revision/metrics-guide.md"]
files.extend(sorted((root / "scalar").glob("*.csv")))
for name in ("formal_metrics", "numeric_metrics", "other_metrics"):
    directory = root / "revision" / name
    files.extend(sorted(directory.glob("*.csv")))
    files.append(directory / "README.md")
for study in ("objective", "arithmetic", "nonconvex", "endpoints", "refinement", "lbfgs", "coefficients"):
    directory = root / "focus" / study
    files.extend(sorted(directory.glob("*.csv")))
    if (directory / "README.md").exists():
        files.append(directory / "README.md")
for group in ("iterative_eval","iterative_tables","feedback_gradient","architecture_search","holistic_plots","prompt_audit","learning_plots","main_visuals","learning_review"):
    files.extend(sorted((root/"focus"/group).glob("*.csv")))
    for name in ("README.md","summary.json","audit.json","confirmation_seed.json"):
        candidate=root/"focus"/group/name
        if candidate.exists(): files.append(candidate)
files += [output/"iterative-wsp-endpoints.csv", output/"iterative-wsp-by-size.csv", output/"iterative-wsp-rounds.csv", output/"iterative-wsp-symmetry.csv", root/"focus/iterative/PROTOCOL.md"]
files.extend(output / name for name in (
    "all-optimizer-raw-rmse-ranking.csv", "optimizer-rmse-task-contributions.csv",
    "architecture-choice-models.csv", "architecture-choice-errors.csv", "architecture-choice-costs.csv",
    "iterative-wsp-learning-curves.csv", "iterative-wsp-learning-summary.csv",
    "iterative-wsp-proposal-curves.csv", "function-prediction-curves.csv"))
with zipfile.ZipFile(output / "all-exact-and-rmse.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root))
