"""Write paper inputs for matched WSP and gradient endpoints after independent scoring.

For a visibly watermarked layout proof only, use --layout-proof /tmp/<directory>.
The release path requires the collector freeze, the scorer audit and all 2,088 rows.
"""
import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median

from display import draw, figure_tex, endpoint_table, sizes_table

HERE = Path(__file__).resolve().parent
FOCUS = HERE.parent
TASKS = {
    "square": ("Square", (1, 4, 16), 16),
    "sin": ("Sine", (1, 4, 16), 16),
    "selfpower": ("Positive self-power", (1, 4, 16), 16),
    "tanh": ("Hyperbolic tangent", (1, 4, 16), 16),
    "add": ("Addition", (1, 2, 4), 2),
    "subtract": ("Subtraction", (1, 2, 4), 2),
    "sum": ("Running sum", (1, 2), 1),
    "sort": ("Sort four", (4, 5, 6), 5),
    "fib": ("Fibonacci", (1, 2, 3), 2),
    "tri": ("Triangular", (2, 3, 4), 3),
}
SPLITS = ("outer", "dev", "train")
SEEDS = (164, 317, 583)
WSP = ("examples", "iter_examples", "formula", "iter_formula")
GRADIENT = ("gd", "gd_lbfgs", "adam_bfgs", "ls_xavier")
METHODS = WSP + GRADIENT


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def sources(layout):
    if layout:
        primary = read(FOCUS / "endpoints/endpoints.csv")
        gradients = read(FOCUS / "lbfgs/scores.csv")
        fake = [dict(r, method="iter_" + r["method"]) for r in primary
                if r["method"] in ("examples", "formula")]
        return ([r for r in primary if r["method"] in ("examples", "formula", "adam_bfgs", "ls_xavier")]
                + [r for r in gradients if r["method"] in ("gd", "gd_lbfgs")] + fake)
    # Opening the scorer's outputs is allowed only after the common selection is frozen.
    freeze = FOCUS / "iterative/FROZEN.json"
    scored = FOCUS / "iterative_eval"
    audit = json.loads((scored / "audit.json").read_text())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if audit.get("global_freeze_sha256") != digest(freeze) or (
            audit.get("complete_trajectories"), audit.get("audited_final_arrays"),
            audit.get("original_wsp_rows_verified")) != (60, 174, 522):
        raise ValueError("The independent auditor has not attested the complete frozen experiment")
    for name in ("comparators.csv", "endpoints.csv"):
        if audit["result_sha256"].get(name) != digest(scored / name):
            raise ValueError(f"{name} differs from the independent scorer's signed-off results")
    originals = [r for r in read(scored / "comparators.csv") if r["method"] in METHODS
                 and r["method"] not in ("iter_examples", "iter_formula")]
    revised = [r for r in read(scored / "endpoints.csv") if r["method"] in ("iter_examples", "iter_formula")]
    return originals + revised


def verified_rows(raw, layout):
    rows = {}
    for r in raw:
        task, method, split = r["task"], r["method"], r["split"]
        if split not in SPLITS:
            continue
        size, seed, slots, n, scalars, exact = (
            int(r[k]) for k in ("size", "seed", "params", "n_examples", "n_scalars", "exact_count")
        )
        rmse, loss = float(r["raw_rmse"]), float(r["train_loss"])
        scaled_rmse = None if layout else r["scaled_rmse"]
        key = task, size, seed, method, split
        if key in rows or not (0 <= exact <= n <= scalars and rmse >= 0 and loss >= 0
                               and math.isfinite(rmse) and math.isfinite(loss)):
            raise ValueError(f"Duplicate or invalid measured data at {key}")
        if not layout and split == "dev" and task in ("fib", "tri") and n != 16:
            raise ValueError("Held-out dev for integer sequences must use the 16 indices 17–32")
        rows[key] = dict(task=task, size=size, seed=seed, params=slots, method=method, split=split,
                         n_examples=n, n_scalars=scalars, exact_count=exact, raw_rmse=rmse, train_loss=loss, scaled_rmse=scaled_rmse)
    required = {(task, size, seed, method, split)
                for task, (_, sizes, _) in TASKS.items() for size in sizes
                for seed in SEEDS for method in METHODS for split in SPLITS}
    if rows.keys() != required:
        raise ValueError(f"Missing {len(required - rows.keys())} or unexpected {len(rows.keys() - required)} rows")
    for task, size, seed, _, split in required:
        designs = {(rows[task, size, seed, m, split]["params"], rows[task, size, seed, m, split]["n_examples"],
                    rows[task, size, seed, m, split]["n_scalars"]) for m in METHODS}
        if len(designs) != 1:
            raise ValueError(f"Unmatched design at {task, size, seed, split}")
    return rows


def size_medians(rows):
    medians = {}
    for task, (_, sizes, _) in TASKS.items():
        for size in sizes:
            for split in SPLITS:
                for method in METHODS:
                    group = [rows[task, size, s, method, split] for s in SEEDS]
                    if len({(g["n_examples"], g["params"]) for g in group}) != 1:
                        raise ValueError("All three starts must share denominators and graph")
                    medians[task, size, split, method] = dict(
                        task=task, size=size, params=group[0]["params"], split=split, method=method,
                        n=group[0]["n_examples"], exact_count=median(g["exact_count"] for g in group),
                        raw_rmse=median(g["raw_rmse"] for g in group),
                    )
    return medians


def method_summary(rows, layout):
    summary = {}
    for method in METHODS:
        record = {}
        for split in SPLITS:
            group = [row for row in rows.values() if row["method"] == method and row["split"] == split]
            if len(group) != 87:
                raise ValueError(f"Expected 87 settings for {method} on {split}")
            record[split] = None if layout else median(row["scaled_rmse"] for row in group)
            if split == "outer":
                record["exact_sets"] = sum(row["exact_count"] == row["n_examples"] for row in group)
        summary[method] = record
    return summary


def matched_comparisons(rows):
    pairs = [("iter_examples", "examples"), ("iter_formula", "formula")]
    pairs += [(w, g) for g in GRADIENT for w in WSP]
    deltas, summary = [], []
    for left, right in pairs:
        for split in SPLITS:
            metrics = ["exact_count", "raw_rmse"]
            if split == "train":
                metrics.append("train_loss")
            for metric in metrics:
                outcomes = Counter()
                for task, (_, sizes, _) in TASKS.items():
                    for size in sizes:
                        for seed in SEEDS:
                            original = rows[task, size, seed, right, split][metric]
                            trial = rows[task, size, seed, left, split][metric]
                            d = trial - original
                            # For counts, more exact predictions is better. For errors, lower is better.
                            ordered = -d if metric == "exact_count" else d
                            outcome = "row_better" if ordered < 0 else "equal" if ordered == 0 else "comparator_better"
                            outcomes[outcome] += 1
                            if metric == "train_loss":
                                outcomes["tie_1e_10" if abs(d) <= 1e-10 else
                                         "row_better_1e_10" if ordered < 0 else "comparator_better_1e_10"] += 1
                            deltas.append(dict(task=task, size=size, seed=seed, split=split, metric=metric,
                                               row_method=left, comparator=right, row_value=trial,
                                               comparator_value=original, row_minus_comparator=d, outcome=outcome))
                summary.append(dict(row_method=left, comparator=right, split=split, metric=metric, n=87,
                                    **{k: outcomes[k] for k in ("row_better", "equal", "comparator_better")},
                                    **{k: outcomes[k] if metric == "train_loss" else "" for k in
                                       ("row_better_1e_10", "tie_1e_10", "comparator_better_1e_10")}))
    return deltas, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-proof", type=Path)
    args = parser.parse_args()
    out = HERE if args.layout_proof is None else args.layout_proof.resolve()
    layout = args.layout_proof is not None
    if layout and Path(tempfile.gettempdir()).resolve() not in out.parents and Path("/tmp").resolve() not in out.parents:
        raise ValueError("Layout proofs may only be written beneath a temporary directory")
    source_rows = sources(layout)
    if not layout:
        from scaled_endpoint_errors import add_scaled_errors
        add_scaled_errors(source_rows)
    rows = verified_rows(source_rows, layout)
    medians = size_medians(rows)
    deltas, summary = matched_comparisons(rows)
    out.mkdir(parents=True, exist_ok=True)
    if not layout:
        write_csv(out / "size_medians.csv", list(medians.values()))
        write_csv(out / "paired_settings.csv", deltas)
        write_csv(out / "paired_summary.csv", summary)
    endpoint_summary = method_summary(rows, layout)
    if not layout:
        write_csv(out / "method_endpoints.csv", [dict(method=m, **endpoint_summary[m]) for m in METHODS])
    outer = sizes_table(medians, TASKS, METHODS, "outer", True, layout)
    tables = [sizes_table(medians, TASKS, METHODS, split, False, layout) for split in SPLITS]
    (out / "representative.tex").write_text(outer)
    (out / "endpoint_tables.tex").write_text("\n\n".join(tables) + "\n")
    (out / "paired_table.tex").write_text(endpoint_table(endpoint_summary, layout))
    (out / "figure1.tex").write_text(figure_tex(layout))
    draw(out / "iterative_figure.pdf", endpoint_summary, layout)
    print(f"{'WATERMARKED LAYOUT' if layout else 'MEASURED RELEASE'}: {len(rows)} endpoint/split rows; "
          f"{len(medians)} method/size/split medians; {len(deltas)} paired metrics; {out}")


if __name__ == "__main__":
    main()
