"""Score the closed-form accumulator and archived learned cells at the two prescribed scales."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECURRENT = HERE.parents[1] / "extension" / "recurrent"
sys.path.insert(0, str(RECURRENT))
from calibrated import ScaledSum, evaluate, setup, torch

setup()
rows = []
for scale in (1.0, 2.0**-40):
    wsp = ScaledSum(101, scale)
    with torch.no_grad():
        wsp.a.fill_(1.)
        wsp.b.fill_(1.)
    records = [("WSP", 101, evaluate(wsp))]
    records.extend(("BP", r["seed"], r["metrics"]) for r in json.loads((RECURRENT / "calibrated.json").read_text()) if r["scale"] == scale)
    for method, seed, metrics in records:
        for split, values in metrics.items():
            rows.append(dict(scale=scale, method=method, seed=seed, split=split, exact=values["exact_all"],
                             rmse=values["rmse"], nrmse=values["nrmse"]))
with (HERE/"scaled_cell.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print([r for r in rows if r["method"] == "WSP" and r["split"] == "joint"])
