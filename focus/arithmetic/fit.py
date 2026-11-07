"""Fit the same displayed arrays with Adam/BFGS and trust-region least squares."""

from fractions import Fraction
import json
import platform

import numpy as np
import scipy
from scipy.optimize import least_squares, minimize

from fixtures import (
    HERE,
    SEEDS,
    TASKS,
    WIDTHS,
    extract,
    forward,
    initial,
    observed,
)


def reverse_loss(width, theta, x, y):
    prediction, z, h = forward(width, theta, x)
    scale = np.maximum(1.0, np.abs(y))
    residual = (prediction - y) / scale
    adjoint = residual / (len(y) * scale)
    _, _, v, _ = extract(width, theta)
    dz = (adjoint[:, None] * v) * (z > 0.0)
    gradient = np.r_[(dz.T @ x).ravel(), dz.sum(axis=0), h.T @ adjoint, adjoint.sum()]
    return 0.5 * np.mean(residual**2), gradient


def residual_and_jacobian(width, theta, x, y):
    prediction, z, h = forward(width, theta, x)
    _, _, v, _ = extract(width, theta)
    active_weight = (z > 0.0) * v
    left = (active_weight[:, :, None] * x[:, None, :]).reshape(len(x), 2 * width)
    jac = np.column_stack((left, active_weight, h, np.ones(len(x))))
    scale = np.maximum(1.0, np.abs(y))
    return (prediction - y) / scale, jac / scale[:, None]


def engineered(task, width):
    sign = 1.0 if task == "add" else -1.0
    p = np.zeros(4 * width + 1)
    w, c, v, _ = extract(width, p)
    w[0] = (1.0, sign)
    v[0] = 1.0
    if width == 1:
        # On the stated bounded real domain an offset keeps this ReLU active.
        c[0] = 20.0
        p[-1] = -20.0
    else:
        w[1] = (-1.0, -sign)
        v[1] = -1.0
    return p


def check_gradient(width, p, x, y):
    r, jac = residual_and_jacobian(width, p, x, y)
    analytic = reverse_loss(width, p, x, y)[1]
    np.testing.assert_allclose(jac.T @ r / len(y), analytic, atol=2e-14, rtol=2e-13)
    step = 1e-6
    centered = np.array(
        [
            (
                reverse_loss(width, p + step * u, x, y)[0]
                - reverse_loss(width, p - step * u, x, y)[0]
            )
            / (2 * step)
            for u in np.eye(len(p))
        ]
    )
    error = max(abs(centered - analytic) / np.maximum(1.0, abs(analytic)))
    assert error < 1e-7, error
    return error


def fit(width, p0, x, y):
    def fun(p):
        return reverse_loss(width, p, x, y)

    theta = p0.copy()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    for step in range(1, 801):
        _, grad = fun(theta)
        grad *= min(1.0, 5.0 / max(np.linalg.norm(grad), 1e-250))
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        theta -= (
            0.02
            * (first / (1 - 0.9**step))
            / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        )
    result = minimize(
        fun, theta, method="BFGS", jac=True, options=dict(maxiter=1000, gtol=1e-14)
    )
    best = result.x if result.fun <= fun(theta)[0] else theta
    diagnostics = dict(
        adam_bfgs=dict(
            adam_steps=800,
            bfgs_iterations=int(result.nit),
            nfev=800 + int(result.nfev),
            njev=800 + int(result.njev),
            status=int(result.status),
            termination=str(result.message),
        )
    )
    nfun = njac = 0

    def residual(p):
        nonlocal nfun
        nfun += 1
        return residual_and_jacobian(width, p, x, y)[0]

    def jacobian(p):
        nonlocal njac
        njac += 1
        return residual_and_jacobian(width, p, x, y)[1]

    result = least_squares(
        residual,
        p0.copy(),
        jac=jacobian,
        method="trf",
        max_nfev=250,
        xtol=4e-15,
        ftol=4e-15,
        gtol=4e-15,
    )
    p, r = result.x.copy(), result.fun.copy()
    start_residual = residual(p0)
    assert r @ r <= start_residual @ start_residual + 1e-14
    accepted = 0
    # Same post-TRF least-squares refinement as the 90-setting comparison.
    for _ in range(10):
        cost = r @ r
        if cost == 0.0:
            break
        delta = np.linalg.lstsq(jacobian(p), -r, rcond=1e-12)[0]
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            cand = p + alpha * delta
            if np.array_equal(cand, p):
                continue
            other = residual(cand)
            if other @ other < cost:
                p, r = cand, other
                accepted += 1
                break
        else:
            break
    diagnostics["trf_xavier"] = dict(
        nfev=nfun,
        njev=njac,
        trf_nfev=int(result.nfev),
        trf_njev=int(result.njev),
        polish_steps=accepted,
        status=int(result.status),
        termination=str(result.message),
    )
    return dict(adam_bfgs=best, trf_xavier=p), diagnostics


def verify_nonconvex(task, width, x, y):
    sign = 1 if task == "add" else -1

    def endpoint(a):
        p = np.zeros(4 * width + 1)
        w, c, v, _ = extract(width, p)
        w[0] = [a, sign * a]
        c[0] = 8 * a
        v[0] = 1 / a
        p[-1] = -8
        return p

    a, b = endpoint(1), endpoint(2)
    mid = (a + b) / 2

    def exact_real_loss(theta):
        p = list(map(Fraction.from_float, theta))
        result = Fraction(0)
        for (u, v), target in zip(x, y):
            u, v, target = map(Fraction.from_float, (u, v, target))
            pred = p[-1] + sum(
                p[3 * width + j]
                * max(0, p[2 * j] * u + p[2 * j + 1] * v + p[2 * width + j])
                for j in range(width)
            )
            result += ((pred - target) / max(1, abs(target))) ** 2
        return result / (2 * len(y))

    exact = [exact_real_loss(p) for p in [a, b, mid]]
    numeric = [float(reverse_loss(width, p, x, y)[0]) for p in [a, b, mid]]
    gap = exact[2] - (exact[0] + exact[1]) / 2
    assert gap > 0 and numeric[2] > (numeric[0] + numeric[1]) / 2
    return dict(
        endpoint_a=a.tolist(),
        endpoint_b=b.tolist(),
        midpoint=mid.tolist(),
        exact_real_losses=[str(q) for q in exact],
        real_midpoint_gap=float(gap),
        binary64_losses=numeric,
        binary64_midpoint_gap=float(numeric[2] - (numeric[0] + numeric[1]) / 2),
    )


def main():
    states, diagnostics, witnesses = {}, [], []
    for task in TASKS:
        for seed in SEEDS:
            x, y = observed(task, seed)
            for width in WIDTHS:
                p = initial(width, seed)
                check = check_gradient(width, p, x, y)
                trained, details = fit(width, p, x, y)
                key = f"{task}_{seed}_{width}"
                for method, theta in (
                    trained | dict(xavier=p, engineered=engineered(task, width))
                ).items():
                    states[key + "_" + method] = theta
                diagnostics.append(
                    dict(
                        task=task,
                        seed=seed,
                        width=width,
                        gradient_max_error=check,
                        **details,
                    )
                )
                witnesses.append(
                    dict(
                        task=task,
                        seed=seed,
                        width=width,
                        **verify_nonconvex(task, width, x, y),
                    )
                )
    np.savez_compressed(HERE / "fits.npz", **states)
    (HERE / "optimization.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (HERE / "midpoint_witnesses.json").write_text(
        json.dumps(witnesses, indent=2) + "\n"
    )
    (HERE / "environment.json").write_text(
        json.dumps(
            dict(
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                platform=platform.platform(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
