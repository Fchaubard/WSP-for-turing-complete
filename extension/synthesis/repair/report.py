"""Regenerate the repair appendix's numerical rows from the two archived score tables."""
import csv
import json

from feedback import HERE, selected


def pair(before, after):
    return before if before == after else before + r" $\to$ " + after


def main():
    with (HERE / "results.csv").open() as f:
        scores = {(r["id"], r["stage"], r["split"]): r for r in csv.DictReader(f)}
    status = {(r["id"], r["stage"]): r for r in json.loads((HERE / "status.json").read_text())}
    if len(status) != 12:
        raise RuntimeError("All six paired trials require terminal records.")

    def score(case_id, stage, column):
        outcome = status[case_id, stage]
        if outcome["status"] != "valid":
            return outcome["status"]
        value = outcome["ops"] if column == "ops" else scores[case_id, stage, column]["correct"]
        return f"{int(value):,}"

    rows = []
    for case in selected():
        name = {"square": "Square", "sin": "Sine", "selfpower": "Self-power"}[case["task"]]
        operator = "Affine" if case["arm"] == "affine_relu" else "Product"
        cells = [name, operator]
        for col in ("ops", "inner", "outer", "fresh_outer"):
            cells.append(pair(score(case["id"], "original", col), score(case["id"], "repair", col)))
        rows.append(" & ".join(cells) + r" \\")
    path = HERE.parents[2] / "paper" / "repair_rows.tex"
    path.write_text("\n".join(rows) + "\n" + r"\bottomrule" + "\n")


if __name__ == "__main__":
    main()
