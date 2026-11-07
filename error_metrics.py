"""Raw root mean squared error in target units, pooled over scalar coordinates."""
import numpy as np

def rmse(prediction, target):
    # Extended precision protects a finite difference from overflow on supported CPUs.
    error = np.abs(np.asarray(prediction, dtype=np.longdouble) - np.asarray(target, dtype=np.longdouble))
    if error.size == 0:
        raise ValueError("RMSE requires at least one output.")
    if not np.all(np.isfinite(error)):
        return float("inf")
    largest = np.max(error)
    if largest == 0:
        return 0.0
    return float(largest * np.sqrt(np.mean((error / largest) ** 2)))
