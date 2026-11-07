"""Exact rational witnesses for the d+r coordinate-edit construction (Python standard library)."""

import argparse
from fractions import Fraction as Q
from itertools import combinations
import json
import math
from pathlib import Path
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def solve(matrix, target):
    """Gaussian elimination over Q; return None when a square matrix is singular."""
    n = len(matrix)
    work = [list(row) + [value] for row, value in zip(matrix, target)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if work[i][col]), None)
        if pivot is None:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        divisor = work[col][col]
        work[col] = [x / divisor for x in work[col]]
        for i, row in enumerate(work):
            if i != col and row[col]:
                coefficient = row[col]
                work[i] = [x - coefficient * y for x, y in zip(row, work[col])]
    return [row[-1] for row in work]


def observability(matrix, n_rows):
    cols = transpose(matrix)
    row = [Q(i == 0) for i in range(len(matrix))]
    result = []
    for _ in range(n_rows):
        result.append(row)
        row = [dot(row, col) for col in cols]
    return result


def characteristic(matrix):
    """Get monic characteristic coefficients in ascending degree, using cyclic c."""
    d = len(matrix)
    rows = observability(matrix, d + 1)
    coeff = solve(transpose(rows[:-1]), [-v for v in rows[-1]])
    if coeff is None:
        raise ValueError("Initialization is unobservable in the fixed readout")
    return coeff + [Q(1)]


def remainder(poly, monic):
    work = list(poly)
    for index in range(len(work) - 1, len(monic) - 2, -1):
        scale = work[index]
        start = index - len(monic) + 1
        for j, value in enumerate(monic):
            work[start + j] -= scale * value
    return work[: len(monic) - 1]


def inject(matrix, sparse):
    result = [row.copy() for row in matrix]
    for index, value in sparse.items():
        result[index][0] += value
    return result


def xavier(seed, d):
    """Xavier-uniform binary64 draws embedded into Q without decimal conversion."""
    rng = random.Random(seed)
    rec_bound = math.sqrt(6 / (d + d))
    in_bound = math.sqrt(6 / (1 + d))
    matrix = [[Q.from_float(rng.uniform(-rec_bound, rec_bound))
               for _ in range(d)] for _ in range(d)]
    b = [Q.from_float(rng.uniform(-in_bound, in_bound)) for _ in range(d)]
    return matrix, b


def sparse_solution(rows, rhs):
    """Use one coordinate per independent equation (rows have full rank)."""
    for indices in combinations(range(len(rows[0])), len(rows)):
        submatrix = [[row[i] for i in indices] for row in rows]
        weights = solve(submatrix, rhs)
        if weights is not None:
            return dict(zip(indices, weights))
    raise ValueError("No invertible minor for the prescribed conditions")


def first_mismatch(matrix, b, goal):
    """Evaluate the dense state directly until the first discrepancy."""
    h = b
    for k, expected in enumerate(goal):
        if h[0] != expected:
            return k
        h = [dot(row, h) for row in matrix]
    return None


def float_diagnostic(matrix, b, goal):
    """Evaluate rounded weights with explicit binary64 multiply and separate sequential add."""
    mat = [list(map(float, row)) for row in matrix]
    h = list(map(float, b))
    residual = []
    for g in goal:
        residual.append(abs(h[0] - g) / max(1, abs(g)))
        updated = []
        for row in mat:
            total = 0.0
            for weight, activation in zip(row, h):
                total += weight * activation
            updated.append(total)
        h = updated
    return dict(binary64_equal=sum(e == 0 for e in residual), binary64_max_scaled=max(residual))


def witnesses(matrix, initial_input, targets, horizon):
    d = len(matrix)
    base = characteristic(matrix)
    deltas = []
    for j in range(d):
        changed = characteristic(inject(matrix, {j: Q(1)}))
        deltas.append([a - b for a, b in zip(changed, base)])
    original_rows = observability(matrix, d)

    for task, (annihilator, goal) in targets.items():
        r = len(annihilator) - 1
        if r > d:
            continue
        # Full all-horizon pole placement modulo the target annihilator.
        target_rem = [-value for value in remainder(base, annihilator)]
        columns = [remainder(delta, annihilator) for delta in deltas]
        sparse = sparse_solution(transpose(columns), target_rem)
        written = inject(matrix, sparse)
        assert not any(remainder(characteristic(written), annihilator))
        b = solve(observability(written, d), goal[:d])
        assert b is not None and first_mismatch(written, b, goal) is None
        n_a = sum(value != 0 for value in sparse.values())
        n_b = sum(x != y for x, y in zip(initial_input, b))
        assert n_a + n_b <= d + r
        yield "results", dict(task=task, width=d, order=r, recurrent_edits=n_a,
                              input_edits=n_b, predicted=d + r, verified_impulses=horizon,
                              **float_diagnostic(written, b, goal))

        for length in range(1, d + r + 1):
            if length <= d:
                written = matrix
                rows = original_rows[:length]
                rhs = [g - dot(row, initial_input) for g, row in zip(goal, rows)]
                input_delta = sparse_solution(rows, rhs)
                b = [x + input_delta.get(j, 0) for j, x in enumerate(initial_input)]
                sparse = {}
            else:
                # Enforce the target recurrence only at the H-d observed offsets.
                rows = []
                rhs = []
                for k in range(length - d):
                    window = goal[k:k + d + 1]
                    rows.append([dot(delta, window) for delta in deltas])
                    rhs.append(-dot(base, window))
                sparse = sparse_solution(rows, rhs)
                written = inject(matrix, sparse)
                b = solve(observability(written, d), goal[:d])
                assert b is not None
            n_a = sum(value != 0 for value in sparse.values())
            n_b = sum(x != y for x, y in zip(initial_input, b))
            mismatch = first_mismatch(written, b, goal)
            assert n_a + n_b <= length and (mismatch is None or mismatch >= length)
            if length == d + r:
                assert mismatch is None and not any(remainder(characteristic(written), annihilator))
            yield "horizon_sweep", dict(task=task, width=d, order=r, imposed_horizon=length,
                                        recurrent_edits=n_a, input_edits=n_b, predicted=length,
                                        first_mismatch_index=mismatch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("general.json"))
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--horizon", type=int, default=32)
    args = ap.parse_args()
    if not args.widths or min(args.widths) < 1 or args.horizon < max(args.widths) + 4:
        ap.error("Positive widths and a verification horizon of at least max(widths)+4 are required")
    fib = [0, 1]
    while len(fib) < args.horizon:
        fib.append(fib[-2] + fib[-1])
    targets = dict(prefix_sum=([-1, 1], [1] * args.horizon),
                   fibonacci=([-1, -1, 1], fib),
                   triangular=([-1, 3, -3, 1],
                               [n * (n + 1) // 2 for n in range(args.horizon)]))
    tables = {"results": [], "horizon_sweep": []}
    for d in args.widths:
        for seed in args.seeds:
            matrix, b = xavier(seed, d)
            for table, row in witnesses(matrix, b, targets, args.horizon):
                tables[table].append({**row, "seed": seed})
    result = dict(initialization="Python Random.uniform binary64 Xavier draws embedded exactly in Q",
                  readout="first coordinate fixed; homogeneous dynamics with scalar impulse at time 0",
                  polynomial_certificate="target monic divides edited characteristic polynomial; first d impulses agree",
                  finite_check="full recurrent state propagated with fractions.Fraction",
                  float_diagnostic="Round optimal rational pole-placement weights once to binary64. Evaluate sequential binary64 multiply/add in ascending column order; scaled errors divide by max(1,abs(reference)).",
                  **tables)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, rows in tables.items():
        tight = sum(row["recurrent_edits"] + row["input_edits"] == row["predicted"] for row in rows)
        print(f"{name}: {len(rows)} settings verified, {tight} attain predicted cost")


if __name__ == "__main__":
    main()
