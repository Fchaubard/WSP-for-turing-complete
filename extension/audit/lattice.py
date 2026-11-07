"""Capacity and interpolation audit for x^2 on three prescribed binary64 grids.

Runs under Python >=3.9 using separate multiplications and additions with IEEE
binary64 round-to-nearest ties-to-even. No BLAS, vectorization, or FMA is used.
"""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import struct


def xavier_width(width, seed):
    rng = random.Random(seed)
    limit = math.sqrt(6 / (1 + width))
    slope = rng.uniform(-math.sqrt(3), math.sqrt(3))
    return [slope, 0.0] + [rng.uniform(-limit, limit) for _ in range(width)] + [0.0] * width + [
        rng.uniform(-limit, limit) for _ in range(width)
    ]


def digest(values):
    return sha256(b"".join(struct.pack(">d", v) for v in values)).hexdigest()


def paired_secants(points, labels):
    """Each affine piece interpolates its two assigned sites; share final site if odd."""
    samples = list(zip(points, labels))
    pairs = list(zip(samples[::2], samples[1::2]))
    if len(samples) % 2:
        pairs.append((samples[-2], samples[-1]))
    lines = []
    for (x0, y0), (x1, y1) in pairs:
        m = (y1 - y0) / (x1 - x0)
        lines.append((m, y0 - m*x0))
    knots, jumps = [], []
    for (m0, b0), (m1, b1) in zip(lines, lines[1:]):
        jump = m1 - m0
        knots.append((b0 - b1) / jump)
        jumps.append(jump)
    return lines[0], knots, jumps


def write(points, labels, width):
    sites = points[: 2 * (width + 1)]
    (a, b), knots, jumps = paired_secants(sites, labels[:len(sites)])
    active = len(knots)
    pad = width - active
    coefficients = ([a, b] + [Q(1)] * active + [Q(0)] * pad
                    + [-t for t in knots] + [Q(0)] * pad + jumps + [Q(0)] * pad)
    doubles = list(map(float, coefficients))
    assert all(Q(v) == q for v, q in zip(doubles, coefficients))
    assert all(abs(v) <= 128 for v in doubles)
    initial = xavier_width(width, 2147)
    final = initial.copy()
    final[:] = doubles
    return final, initial


def predict(weights, x):
    n = (len(weights) - 2) // 3
    s = weights[0] * x + weights[1]
    for k in range(n):
        pre = weights[2 + k] * x + weights[2 + n + k]
        term = weights[2 + 2*n + k] * max(0., pre)
        s = s + term
    return s


def score(weights, grid):
    errors = [abs(predict(weights, x) - x*x) for x in map(float, grid)]
    return {"n": len(errors), "exact": errors.count(0.), "max_abs": max(errors),
            "rmse": math.sqrt(math.fsum(e * e for e in errors) / len(errors))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("lattice.json"))
    args = ap.parse_args()
    all_points = [Q(n, 16) for n in range(-160, 161)]
    midpoint = [(a+b)/2 for a, b in zip(all_points, all_points[1:])]
    central = [x for x in all_points if abs(x) <= 3]
    outer = [x for x in all_points if abs(x) > 3]
    rng = random.Random(77391)
    diffuse_outer = [rng.choice([-1,1]) * rng.uniform(3.,10.) for _ in range(4096)]
    grids = {"full": all_points, "inner": central, "outer": outer}
    rows = []
    for name, points in grids.items():
        labels = [x*x for x in points]
        edge = (len(points) - 1) // 2
        for w in range(edge + 1):
            weights, initial = write(points, labels, w)
            scores = score(weights, points)
            bound = min(len(points), 2*(w+1))
            assert scores["exact"] == bound
            row = {"grid": name, "width": w, "slots": len(weights), "exact": bound,
                   "max_abs": scores["max_abs"], "rmse": scores["rmse"]}
            if w in (edge, edge-1):
                row["configuration"] = {
                    "initial_sha256": digest(initial), "final_sha256": digest(weights),
                    "edits": sum(x != y for x, y in zip(initial, weights)),
                    "max_abs_coefficient": max(map(abs, weights)),
                }
                row["transfer"] = {
                    **{other:score(weights, v) for other,v in grids.items() if other != name},
                    "full_midpoints":score(weights, midpoint),
                    "iid_outer":score(weights, diffuse_outer),
                }
                if name == "full" and w == edge:
                    assert row["transfer"]["full_midpoints"]["exact"] == 0
                    assert scores["max_abs"] == 0
            rows.append(row)
    # An explicit degree-at-most-two prior permits identification from only three labels.
    nodes = [-1,0,1]
    labels = [n*n for n in nodes]
    submitted = []
    for seed in range(5):
        rng = random.Random(seed)
        nodal = [rng.uniform(-math.sqrt(1.5), math.sqrt(1.5)) for _ in nodes]
        # Squared loss on the degree-two Lagrange basis at its three nodes has identity Hessian.
        for _ in range(2):
            nodal = [v - (v-label) for v, label in zip(nodal,labels)]
        assert nodal == labels
        ym, yz, yp = map(Q, nodal)
        inferred = [ym*x*(x-1)/2 + yz*(1-x*x) + yp*x*(x+1)/2 for x in all_points]
        trained_weights, _ = write(all_points, inferred, 160)
        submitted.append(digest(trained_weights))
    assert len(set(submitted)) == 1
    reference_weights, _ = write(all_points, [x*x for x in all_points], 160)
    assert submitted[0] == digest(reference_weights)
    poly_control = dict(seeds=5, observed_x=nodes, training_labels=labels, recovered_nodal_coefficients=nodal,
                        knowns="degree at most two, deployment grid, Lagrange basis, secant compiler",
                        complete_network_sha256=submitted[0], shell=score(trained_weights, outer))
    resolution_sweep = []
    for q in (16,32,64,128):
        points = [Q(i, q) for i in range(-10*q,10*q+1)]
        labels = [x*x for x in points]
        edge = 10*q
        # Bound roundoff to the corresponding real spline by summing magnitudes at each add.
        u = Q(1,2**53)
        bound = Q(1,2**39) + edge*Q(1,2**31) + u*(1411*edge + 180226*edge*(edge+1)//2)
        assert 2*bound < Q(1,q*q)
        w, _ = write(points, labels, edge)
        w_minus, _ = write(points, labels, edge-1)
        mids = [(a+b)/2 for a,b in zip(points,points[1:])]
        on, below, off = score(w,points), score(w_minus,points), score(w,mids)
        assert on["exact"] == len(points) and below["exact"] == len(points)-1 and off["exact"] == 0
        resolution_sweep.append(dict(denominator=q, minimum_width=edge, slots=len(w),
                                     certified_roundoff_upper=float(bound), forbidden_gap=float(Q(1,2*q*q)),
                                     on_grid=on, one_hinge_below=below, midpoints=off))
    output = {"grid":"x=n/16, n=-160,...,160; inner |x|<=3; outer 3<|x|<=10",
              "architecture":"a*x+b plus width many v*ReLU(u*x+c) residual terms",
              "coefficient_cap":128, "xavier_init_seed":2147, "iid_outer_seed":77391,
              "execution":"separate binary64 operations in increasing unit index, RN ties-to-even",
              "resolution_sweep":resolution_sweep, "quadratic_prior_label_control":poly_control, "rows":rows}
    args.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    for row in rows:
        if "transfer" in row:
            print(row["grid"],row["width"],row["exact"],row["transfer"])


if __name__ == "__main__":
    main()
