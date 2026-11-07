"""Accumulate root mean square error in target units without squaring large errors."""

from decimal import Decimal, localcontext
from pathlib import Path
import csv
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXT = ROOT / "extension"


def metric(prediction, target):
    prediction, target = np.broadcast_arrays(prediction, target)
    with np.errstate(all="ignore"):
        error = np.abs(prediction - target)
    n = error.size
    exact = int((prediction == target).sum())
    if not np.isfinite(error).all():
        return {"n": n, "exact": exact, "raw_rmse": "nonfinite"}
    scale = float(error.max())
    if scale == 0:
        rmse = "0"
    else:
        normalized = float(np.sqrt(np.mean((error / scale) ** 2)))
        with localcontext() as ctx:
            ctx.prec = 19
            rmse = str(Decimal.from_float(scale) * Decimal.from_float(normalized))
    return {"n": n, "exact": exact, "raw_rmse": rmse}


def save(name, rows):
    with (HERE / (name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(name, len(rows), flush=True)
