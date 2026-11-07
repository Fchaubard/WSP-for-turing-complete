"""Render the primary nonconvex comparison and label the earlier 90-setting archive."""

import csv
from iteration_dashboard import section as iteration_section, lead as iterative_lead
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "output"
PRIMARY = ROOT / "focus/endpoints"
TASKS = {
    "square": (
        "Square",
        "Affine input-to-output connection and rectified linear hidden units",
        "hidden units",
    ),
    "sin": (
        "Sine",
        "Affine input-to-output connection and rectified linear hidden units",
        "hidden units",
    ),
    "selfpower": (
        "Positive self-power",
        "Affine input-to-output connection and rectified linear hidden units",
        "hidden units",
    ),
    "tanh": (
        "Hyperbolic tangent",
        "Affine input-to-output connection and rectified linear hidden units",
        "hidden units",
    ),
    "add": (
        "Addition",
        "Dense rectified linear hidden layer with two inputs and one output",
        "hidden units",
    ),
    "subtract": (
        "Subtraction",
        "Dense rectified linear hidden layer with two inputs and one output",
        "hidden units",
    ),
    "sum": (
        "Running sum",
        "Linear recurrent network with a learned input map and readout",
        "state coordinates",
    ),
    "sort": (
        "Sort four numbers",
        "Four input wires connected by rectified linear residual updates",
        "residual updates",
    ),
    "fib": (
        "Fibonacci numbers",
        "Autonomous linear recurrence with learned initial state and readout",
        "state coordinates",
    ),
    "tri": (
        "Triangular numbers",
        "Autonomous linear recurrence with learned initial state and readout",
        "state coordinates",
    ),
}
METHODS = {
    "examples": "WSP with examples only",
    "formula": "WSP with formula and examples",
    "adam_bfgs": "Adam followed by BFGS",
    "ls_xavier": "Trust-region least squares",
    "xavier": "Initial Xavier weights",
}
SPLITS = ("train", "dev", "outer")
TOLERANCE = 1e-10


def read(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def cell(value):
    return html.escape(str(value))


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join('<th scope="col">' + cell(h) + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + cell(x) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def fmt(x):
    rounded = f"{x:.3g}"
    if "e" not in rounded:
        return rounded
    value, power = rounded.split("e")
    superscript = str(int(power)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{value} × 10{superscript}"


def index(records):
    index = defaultdict(dict)
    for r in records:
        if r["method"] not in METHODS:
            continue
        key = (r["task"], int(r["size"]), int(r["seed"]))
        method_split = (r["method"], r["split"])
        assert method_split not in index[key]
        index[key][method_split] = r
    assert len(index) == 87
    for rows in index.values():
        assert all(
            (m, s) in rows for m in ("xavier", "adam_bfgs", "ls_xavier") for s in SPLITS
        )
    return index


def primary_tables(data):
    benchmarks = {
        (r["task"], 1, int(r["seed"])): float(r["real_optimum"])
        for r in read(HERE / "optimizer/results.csv")
        if r["method"] == "xavier"
        and int(r["size"]) == 1
        and r["task"] in ("square", "sin", "tanh", "selfpower")
    }
    assert len(benchmarks) == 12
    summary = []
    for method, name in METHODS.items():
        by_split = [
            [r[method, s] for r in data.values() if (method, s) in r] for s in SPLITS
        ]
        assert len({len(x) for x in by_split}) == 1
        if not by_split[0]:
            continue
        attained = [
            float(data[k][method, "train"]["train_loss"]) - optimum
            for k, optimum in benchmarks.items()
            if (method, "train") in data[k]
        ]
        assert all(gap >= -TOLERANCE for gap in attained)
        summary.append(
            [
                name,
                len(by_split[0]),
                fmt(median(float(r["train_loss"]) for r in by_split[0])),
                f"{sum(gap <= TOLERANCE for gap in attained)} / {len(attained)}",
                *(
                    sum(int(r["exact_count"]) == int(r["n_examples"]) for r in rows)
                    for rows in by_split
                ),
            ]
        )
    totals = table(
        [
            "Method",
            "Scored settings (of 87)",
            "Median training loss",
            "Near certified optimum",
            "Exact on all training inputs",
            "Exact on all held-out central / early inputs",
            "Exact on all outer inputs",
        ],
        summary,
    )
    compared = []
    for writer in ("examples", "formula"):
        for solver in ("adam_bfgs", "ls_xavier"):
            for split, metric, tol, label in (
                ("train", "train_loss", TOLERANCE, "Training loss"),
                ("outer", "raw_rmse", 0.0, "Outer RMSE"),
            ):
                diff = [
                    float(g[writer, split][metric]) - float(g[solver, split][metric])
                    for g in data.values()
                    if (writer, split) in g
                ]
                compared.append(
                    [
                        METHODS[writer],
                        METHODS[solver],
                        label,
                        len(diff),
                        sum(x < -tol for x in diff),
                        sum(abs(x) <= tol for x in diff),
                        sum(x > tol for x in diff),
                    ]
                )
    paired = table(
        [
            "Direct writer",
            "Gradient method",
            "Quantity",
            "Matched settings",
            "Lower for WSP",
            "Tie",
            "Lower for gradient method",
        ],
        compared,
    )
    return totals, paired


def task_sections(data):
    sections = []
    for task, (label, arch, size_unit) in TASKS.items():
        examples = [(key, r) for key, r in data.items() if key[0] == task]
        sizes = sorted({key[1] for key, _ in examples})
        rows = []
        for size in sizes:
            rr = [g for key, g in examples if key[1] == size]
            assert len(rr) == 3
            for method, name in METHODS.items():
                by_split = [
                    [g[method, s] for g in rr if (method, s) in g] for s in SPLITS
                ]
                if not by_split[0]:
                    params = rr[0]["xavier", "train"]["params"]
                    rows.append([size, params, name, 0, "—", *(["—"] * 6)])
                    continue
                assert all(len(q) == len(by_split[0]) for q in by_split)
                first = by_split[0][0]
                line = [
                    size,
                    first["params"],
                    name,
                    len(by_split[0]),
                    fmt(median(float(q["train_loss"]) for q in by_split[0])),
                ]
                for group in by_split:
                    n = {int(q["n_examples"]) for q in group}
                    assert len(n) == 1
                    hits = median(int(q["exact_count"]) for q in group)
                    line.extend(
                        [
                            f"{hits:,.0f} / {n.pop():,}"
                            if float(hits).is_integer()
                            else f"{hits:,.1f} / {n.pop():,}",
                            fmt(median(float(q["raw_rmse"]) for q in group)),
                        ]
                    )
                rows.append(line)
        titles = [
            size_unit.capitalize(),
            "Scalar slots",
            "Method",
            "Scored starts",
            "Training loss",
            "Train exact",
            "Train RMSE",
            "Central / early exact",
            "Central / early RMSE",
            "Outer exact",
            "Outer RMSE",
        ]
        sections.append(
            '<details class="task"><summary>'
            + cell(label)
            + "</summary><p>"
            + cell(arch)
            + ".</p>"
            + table(titles, rows)
            + "</details>"
        )
    return "".join(sections)


def limited_memory_section():
    folder = ROOT / "focus/lbfgs"
    methods = json.loads((folder / "summary.json").read_text())["methods"]
    labels = {
        "gd": "Gradient descent alone",
        "gd_lbfgs": "Gradient descent followed by L-BFGS",
        "adam800": "Adam alone",
        "adam_lbfgs": "Adam followed by L-BFGS",
        "lbfgs_xavier": "L-BFGS directly from Xavier",
    }
    rows = []
    for method, name in labels.items():
        r = methods[method]
        assert r["n"] == 87 and r["n_certified"] == 12
        rows.append(
            (
                name,
                fmt(r["median_train_loss"]),
                r["train_near_zero"],
                r["near_certified"],
                r["perfect"]["train"],
                r["perfect"]["outer"],
            )
        )
    comparisons = table(
        [
            "Training method",
            "Median training loss",
            "Training loss at most 10⁻¹⁰ (87)",
            "Near certified optimum (12)",
            "All training outputs exact (87)",
            "All outer outputs exact (87)",
        ],
        rows,
    )
    sine = {
        r["method"]: r
        for r in read(folder / "by_task.csv")
        if r["task"] == "sin" and r["split"] == "outer"
    }
    gd, polished = sine["gd"], sine["gd_lbfgs"]
    return f"""<h2 id="limited-memory">Does limited-memory curvature refinement help?</h2>
<p>Starting from the same initial weights, full-batch gradient descent takes a step against the derivative of the observed training loss. It may halve a step until the loss drops sufficiently, and makes at most 800 updates. Limited-memory BFGS (L-BFGS) uses up to ten recent parameter and gradient differences to estimate curvature. We ran it alone, after gradient descent, or after 800 Adam updates, with up to 1,000 refinement iterations per setting. Each refinement retains its lowest observed training loss among the starting point and accepted iterates, including the returned point if better.</p>
{comparisons}
<p>On sine, the median training loss across three sizes and three seeds fell from {fmt(float(gd["median_train_loss"]))} for gradient descent to {fmt(float(polished["median_train_loss"]))} after refinement. On those same settings the median outer RMSE increased from {fmt(float(gd["median_rmse"]))} to {fmt(float(polished["median_rmse"]))}. <a href="gradient-descent-and-lbfgs.csv">Download the per-setting gradient descent and L-BFGS measurements</a>.</p>"""


def historical_table():
    historical = json.loads((HERE / "optimizer/summary.json").read_text())
    by_name = {r["method"]: r for r in historical["methods"]}
    names = dict(
        METHODS,
        adam800="Adam after 800 updates",
        ls_examples="Least squares initialized by examples-only WSP",
        ls_formula="Least squares initialized by formula-given WSP",
    )
    rows = []
    for code, label in names.items():
        r = by_name[code]
        rows.append(
            [
                label,
                r["hinge_certified_near"],
                r["train_near_best"],
                r["shell_all_exact"],
            ]
        )
    return table(
        [
            "Historical method",
            "Near certified global optimum (12 one-rectifier cases)",
            "Near lowest measured training loss (90)",
            "Whole outer sample exact (90)",
        ],
        rows,
    )


STYLE = """
:root{color-scheme:light;--ink:#173239;--muted:#455b5e;--line:#cee0da;--green:#20665b}
*{box-sizing:border-box}html{background:#f7f7f2;color:var(--ink)}
body{max-width:1200px;margin:auto;padding:48px 5vw 66px;font:16px/1.55 system-ui,sans-serif}
header{border-top:5px solid var(--green);padding-top:1.2rem}h1,h2{font-family:Georgia,serif;line-height:1.12}
h1{font-size:clamp(2rem,5.5vw,4rem);max-width:16em;margin:.3em 0}h2{font-size:1.7rem;margin:2em 0 .45em}
p{max-width:56rem}.lead{font-size:1.15rem}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:#fffefb;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}tbody td:first-child,th:first-child{text-align:left}
th{color:#305851;vertical-align:bottom;font-size:12px}summary{cursor:pointer;color:var(--green);font-weight:650}
details{margin:.4rem 0;padding:.6rem 0;border-bottom:1px solid var(--line)}details>p{margin:.6rem 0}.task table{min-width:850px}
.task td:nth-child(3),.task th:nth-child(3){text-align:left}a{color:#145e83}.note{color:var(--muted);font-size:.9rem}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line)}@media print{body{padding:0}h2{break-after:avoid}tr{break-inside:avoid}}
"""


def page(records):
    data = index(records)
    totals, paired = primary_tables(data)

    def perfect(method):
        scored = [g[method, "outer"] for g in data.values() if (method, "outer") in g]
        return sum(int(r["exact_count"]) == int(r["n_examples"]) for r in scored)

    examples_exact = perfect("examples")
    formulas_exact = perfect("formula")
    adam_exact = perfect("adam_bfgs")
    trust_exact = perfect("ls_xavier")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-shot and iterative WSP on 87 nonconvex settings</title><style>{STYLE}</style></head><body>
<header><h1>One-shot and iterative weight-space programming</h1>
<p class="lead">{iterative_lead()}</p></header>
{iteration_section()}
<h2>Original one-shot study</h2>
<p>Weight-space programming (WSP) directly assigns network coefficients. A pretrained language model, Astra, received the initial weights and labeled examples in each trial. One WSP condition also received the target formula. The gradient methods used the same examples and the same initial weights. Each setting fixes one task, architecture size and initialization seed. The 87 settings comprise 29 task–size combinations at three seeds. In 18 settings a dense hidden layer was trained to add or subtract.</p>
<p>Every method was asked to minimize one half of the mean squared training error after dividing each scalar error by the larger of one and the absolute target value. For each setting, two parameter arrays had an average loss strictly smaller than the loss at their midpoint. This witnesses nonconvexity in the network parameters. Training used twelve labeled examples for the continuous tasks and nine indices for Fibonacci and triangular numbers. WSP prompts gave accurate extrapolation as a secondary goal.</p>
<h2>What did each method achieve?</h2>
<p>Backpropagation computes loss derivatives. Adaptive moment estimation (Adam) took 800 gradient steps before Broyden–Fletcher–Goldfarb–Shanno (BFGS) refinement, which estimates local curvature. Trust-region least squares used derivatives of individual residuals from the same Xavier start, a random initialization scaled to layer width. The two numerical methods have different evaluation budgets. Exactness requires equality with the reference rounded to binary64, the 64-bit floating-point format, coordinate by coordinate for a vector or a sequence.</p>
<p>A rectifier returns the larger of zero and its input. An affine connection adds a weighted input to an offset. For the twelve one-input networks that combine a direct affine connection with a single hidden rectifier, we independently solved the global training optimum in real arithmetic using exact fractions. The near-optimum column counts valid fits within 10<sup>−10</sup> of that reference. It uses the same central examples as the primary trials.</p>
{totals}
<p class="note">A setting contributes one success only if every tested example is exactly correct on the indicated split. An unusable returned array has no numerical loss or error. The scored-setting denominator identifies evaluable outputs. The three-seed comparisons share held-out inputs and the proposals for several sizes can come from a single model response.</p>
<h2>Who had the smaller loss or outer error on the same setting?</h2>
<p>For training loss, differences within 10<sup>−10</sup> count as ties. For outer predictions we compare the raw root mean squared error (RMSE), the square root of the mean squared prediction error in the target’s own units. Each sequence or vector coordinate contributes one error. An outer tie requires equal RMSE. Each comparison uses the same task, size, seed, examples and split for both methods.</p>
{paired}
{limited_memory_section()}
<h2 id="returned-weights">What weights did the model actually return?</h2>
<p>For two-hidden-unit addition and subtraction, all twelve replies used the same algebraic rule. Let <i>t</i> be the sum or difference of the two inputs, and let <i>R</i> be a rectifier. One unit computes <i>R</i>(<i>t</i>), the other computes <i>R</i>(−<i>t</i>), and the output subtracts the second from the first. Every returned array matched all 2,048 outer pairs. The one-hidden-unit replies used an offset inside the rectifier and cancelled it at the output. None matched every outer pair. On sine, none of the eighteen returned networks satisfied the target’s exact odd symmetry across [−10,10], meaning that every proposed function differed somewhere from the negative of its value at the opposite input.</p>
<p class="note">The archive contains the subtraction prompt, literal answer and event metadata under <code>focus/arithmetic/trials/738159e5daa8</code> and the formula-given sine example under <code>focus/objective/trials/ba89053bf8ca</code>, with file suffixes <code>.prompt.txt</code>, <code>.reply.txt</code> and <code>.meta.json</code>. It also contains the reproducible coefficient checks in <code>focus/coefficients/</code>. The sixty records show one returned model message each and no recorded tool event. These observations establish the coefficients and their effect. <a href="research-bundle.zip">Download the original prompts, replies, metadata and coefficient audit</a>.</p>
<h2>How does each task change with network size?</h2>
<p>Expand a task to see all sizes. Each row reports medians over the scored starts shown. The exact comparison counts whole examples, and RMSE pools scalar errors within a start before taking the median across starts. “Central / early” means held-out central inputs or indices 9–32 for the two integer sequences. Their outer split is indices 33–64. The continuous outer split changes input magnitude from the central interval [−3,3] to magnitudes 3–10, restricted to positive inputs for self-power. A scalar slot is one free numerical parameter.</p>
{task_sections(data)}
<h2>Earlier 90-setting comparison</h2>
<details><summary>View the earlier protocol and its results</summary><p>The earlier ten-task benchmark included 21 settings where prediction was affine in the trainable parameters. Its writing prompt prioritized maximum scaled error, while its gradient fits minimized the mean of squared scaled errors. The 87-setting comparison above repeats the WSP requests with the same primary training objective as the solvers and uses dense hidden-layer networks for arithmetic. In the historical table, “near” allows an additive training loss difference of 10<sup>−10</sup>.</p>
{historical_table()}<p><a href="optimizer-runs.csv">Download the historical 90-setting optimizer table, including 945 method–setting rows</a>.</p></details>
<footer><a href="nonconvex-endpoints.csv">Download the primary 87-setting endpoint table</a> · <a href="pdf/weight-space-programming.pdf">Read the manuscript and proofs</a> · <a href="all-exact-and-rmse.zip">Download the study error tables</a> · <a href="research-bundle.zip">Download code, weights and original replies</a></footer>
</body></html>"""


def export_historical():
    original = read(HERE / "optimizer/results.csv")

    def key(row):
        return tuple(row[n] for n in ("task", "size", "seed", "method"))

    metrics = {
        (key(r), r["split"]): r
        for r in read(ROOT / "revision/formal_metrics/optimizer_rmse.csv")
    }
    assert len(metrics) == len(original) * 3
    with (OUT / "optimizer-runs.csv").open("w", newline="") as file:
        w = csv.DictWriter(
            file, fieldnames=[*original[0], *(s + "_rmse" for s in SPLITS)]
        )
        w.writeheader()
        for row in original:
            for s in SPLITS:
                m = metrics[key(row), s]
                assert float(m["exact"]) == float(row[s + "_exact"])
                row[s + "_rmse"] = m["rmse"]
            w.writerow(row)


if __name__ == "__main__":
    records = read(PRIMARY / "endpoints.csv")
    assert (ROOT / "focus/objective/status.csv").is_file(), (
        "Score the primary WSP trials first"
    )
    status = read(ROOT / "focus/objective/status.csv")
    assert len(status) == 138 and all(r["status"] != "missing" for r in status)
    counts = {
        m: sum(r["status"] == "valid" and r["method"] == m for r in status)
        for m in ("examples", "formula")
    }
    arithmetic = read(ROOT / "focus/arithmetic/scores.csv")
    for m in counts:
        counts[m] += sum(
            r["method"] == m and r["split"] == "train" and r["exact"] != ""
            for r in arithmetic
        )
        assert counts[m] == sum(
            r["method"] == m and r["split"] == "train" for r in records
        ), "Regenerate primary endpoints after scoring replies"
    export_historical()
    (OUT / "optimizer-comparison.html").write_text(page(records))
