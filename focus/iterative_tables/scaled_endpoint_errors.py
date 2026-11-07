"""Recompute scaled RMSE from saved weights without squaring very large residuals."""
import json
import math
import sys
from pathlib import Path

import numpy as np

FOCUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOCUS / "iterative_eval"))
from measure import control_weights, datasets, original_weights, raw_prediction
from error_metrics import rmse


def add_scaled_errors(records):
    weights = control_weights(original_weights())
    for path in (FOCUS / "iterative/trajectories").glob("*.json"):
        case = json.loads(path.read_text())
        for size, state in case["states"].items():
            key = case["task"], int(size), case["seed"]
            weights[key]["iter_" + case["arm"]] = np.asarray(state["weights"])
    for record in records:
        task, size, seed = record["task"], int(record["size"]), int(record["seed"])
        x, y = datasets(task, seed, None)[record["split"]]
        prediction = raw_prediction(task, size, weights[task, size, seed][record["method"]], x)
        if not math.isclose(rmse(prediction, y), float(record["raw_rmse"]), rel_tol=5e-13):
            raise ValueError(f"Saved weights and published raw RMSE disagree for {task}, {size}, {seed}")
        residual = (prediction - y) / np.maximum(1.0, np.abs(y))
        error = rmse(residual, np.zeros_like(residual))
        recorded_loss = float(record["split_loss"])
        if not math.isfinite(error):
            raise ValueError("Expected finite scaled RMSE")
        if math.isfinite(recorded_loss) and not math.isclose(error, math.sqrt(recorded_loss * 2), rel_tol=5e-13):
            raise ValueError("Recomputed scaled RMSE disagrees with the recorded squared loss")
        record["scaled_rmse"] = error
