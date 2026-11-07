"""Certify global convexity status on the 90 matched training fixtures."""
import csv
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extension/writer"))
from design import TASKS, SIZES, SEEDS, SMOOTH, SERIES, initial, observed, predict
from compare import objective


def rational_array(array):
    """Freeze the observed binary64 values as exact rational constants."""
    return np.vectorize(lambda z: Fraction(float(z)), otypes=[object])(array)


def witness_vectors(task, size, slots, sign):
    """Reciprocal 4:1/1:4 rescalings leave endpoint predictions identical."""
    left = np.zeros(slots)
    right = np.zeros(slots)
    if task in SMOOTH:
        upstream, downstream = 2 + size, 2 + 2 * size  # first hinge bias, output coefficient
    elif task == 'sort':
        upstream, downstream = 0, 4  # first coordinate of each vector in the first block
    else:
        upstream, downstream = size * size, size * size + size  # first entries of b and v
        if task in SERIES:
            left[0] = right[0] = 1  # constant output through the nine observed indices
    left[upstream], right[upstream] = 1, 4
    left[downstream], right[downstream] = 4 * sign, sign
    return left, (left + right) / 2, right


def output_path(task, x, y):
    """Exact path f = baseline + gain * feature, with gain in {±4,±25/4}."""
    xr = rational_array(x)
    baseline = np.full(y.shape, Fraction(0), dtype=object)
    feature = np.full(y.shape, Fraction(1), dtype=object)
    if task == 'sum':
        feature = xr.copy()
    elif task == 'sort':
        baseline = xr.copy()
        feature.fill(Fraction(0))
        feature[:, 0] = np.maximum(xr[:, 0], Fraction(0))
    assert any(value != 0 for value in feature.flat)
    return baseline, feature


def exact_forward(task, size, weights, x):
    """Evaluate the saved graph with Fraction operands including its zero state."""
    p = rational_array(weights)
    if task in SERIES:
        return predict(task, size, p, x)  # x supplies integer indices, not features
    xr = rational_array(x)
    if task != 'sum':
        return predict(task, size, p, xr)
    # The archived floating implementation initializes with np.zeros(dtype=float).
    # An object-valued zero is essential to keep the first multiplication exact.
    transition = p[:size * size].reshape(size, size)
    drive, readout = p[size * size:size * size + size], p[-size:]
    h = np.full((len(x), size), Fraction(0), dtype=object)
    output = []
    for u in xr.T:
        h = h @ transition.T + u[:, None] * drive
        output.append(h @ readout)
    return np.asarray(output).T


def exact_loss(prediction, label):
    residual = (prediction - label) / np.maximum(Fraction(1), np.abs(label))
    return sum(value ** 2 for value in residual.flat) / (2 * label.size)


def assess(task, size, seed):
    start = initial(task, size, seed)
    x, y = observed(task, seed)
    info = dict(task=task, size=size, seed=seed, params=start.size, n_examples=len(x),
                n_scalars=y.size)
    if task in ('add', 'subtract') or (task == 'sum' and size == 0):
        info.update(status='convex', reason='fixed-design weighted linear least squares')
        return info, None
    reason = ('free hinge bias and readout' if task in SMOOTH else
              'free input and output vectors of one active residual rectifier' if task == 'sort'
              else 'free input (or initial-state) and readout vectors')
    baseline, feature = output_path(task, x, y)
    yr = rational_array(y)
    denominator = np.maximum(Fraction(1), np.abs(yr)) ** 2
    correlation = sum(((baseline - yr) * feature / denominator).flat)
    sign = 1 if correlation >= 0 else -1
    left, middle, right = witness_vectors(task, size, start.size, sign)
    signed_feature = sign * feature
    endpoint = baseline + 4 * signed_feature
    midpoint = baseline + Fraction(25, 4) * signed_feature
    endpoint_loss = exact_loss(endpoint, yr)
    middle_loss = exact_loss(midpoint, yr)
    gap = middle_loss - endpoint_loss
    assert gap > 0
    checked_loss = objective(task, size, start, x, y)
    values = []
    for weights, rational_pred, rational_loss in (
        (left, endpoint, endpoint_loss), (middle, midpoint, middle_loss),
        (right, endpoint, endpoint_loss)
    ):
        assert np.array_equal(exact_forward(task, size, weights, x), rational_pred)
        float_pred = predict(task, size, weights, x)
        assert np.allclose(float_pred, rational_pred.astype(float), rtol=2e-15, atol=1e-15)
        value = checked_loss(weights)[0]
        assert np.isclose(value, float(rational_loss), rtol=2e-15, atol=1e-15)
        values.append(value)
    gap_measured = values[1] - (values[0] + values[2]) / 2
    assert gap_measured > 0 and np.isclose(gap_measured, float(gap), rtol=5e-15)
    info.update(status='nonconvex', reason=reason, sign=sign,
                active_feature_outputs=sum(v != 0 for v in feature.flat),
                loss_left=values[0], loss_mid=values[1], loss_right=values[2],
                jensen_gap=gap_measured)
    witness = dict(task=task, size=size, seed=seed, theta_left=left.tolist(), theta_right=right.tolist(),
                   exact_gap_numerator=str(gap.numerator), exact_gap_denominator=str(gap.denominator))
    return info, witness


def main():
    rows, witnesses = [], []
    for task in TASKS:
        for size in SIZES[task]:
            for seed in SEEDS:
                row, witness = assess(task, size, seed)
                rows.append(row)
                if witness is not None:
                    witnesses.append(witness)
    with (HERE / 'eligibility.csv').open('w', newline='') as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / 'witnesses.json').write_text(json.dumps(witnesses, indent=2) + '\n')
    print('Status counts:', dict(Counter(r['status'] for r in rows)))
    for task in TASKS:
        gaps = [r['jensen_gap'] for r in rows if r['task'] == task and r['status'] == 'nonconvex']
        if gaps:
            print(f'{task}: {len(gaps)} nonconvex, gap range {min(gaps):.8g} to {max(gaps):.8g}')


if __name__ == '__main__':
    main()
