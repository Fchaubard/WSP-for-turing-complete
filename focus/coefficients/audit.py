"""Audit fixed public replies, algebraic symmetry, and measured endpoints without a model call."""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import csv
import json
import itertools
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extension/writer"))
from design import initial as old_initial

sys.path.insert(0, str(ROOT / "focus/arithmetic"))
from fixtures import extract, initial as arithmetic_initial


def load_json(path):
    return json.loads(path.read_text())


def parse_reply(path):
    text = path.read_text().strip()
    if text.startswith(chr(96) * 3):
        text = text.split("\n", 1)[1].rsplit(chr(96) * 3, 1)[0]
    return json.loads(text)


def symmetry(theta, width, parity):
    """Exact supremum of |f(-x)-parity*f(x)| on [-10,10] for a real spline."""
    q = [Fraction.from_float(float(t)) for t in theta]
    slope, intercept = q[:2]
    terms = list(
        zip(q[2 : 2 + width], q[2 + width : 2 + 2 * width], q[2 + 2 * width :])
    )

    def evaluate(x):
        return slope * x + intercept + sum(v * max(0, w * x + c) for w, c, v in terms)

    points = {Fraction(0), Fraction(10)}
    for w, c, v in terms:
        if w and v and abs(c / w) < 10:
            points.add(abs(c / w))
    max_error, where = max(
        (abs(evaluate(-x) - parity * evaluate(x)), x) for x in points
    )
    return max_error, where


def arithmetic_structure(task, width, p):
    sign = 1 if task == "add" else -1
    w, c, v, b = extract(width, p)
    if width == 1:
        orientation = int(w[0, 0])
        assert orientation in (-1, 1) and tuple(w[0]) == (
            orientation,
            orientation * sign,
        )
        assert v[0] == orientation and b == -orientation * c[0]
        return (
            "offset_full_domain" if c[0] >= 20 else "offset_partial_domain",
            float(c[0]),
        )
    assert b == 0 and np.all(c == 0)
    gains = Counter()
    active = 0
    for row, weight in zip(w, v):
        if weight != 0:
            direction = int(row[0])
            assert direction in (-1, 1) and tuple(row) == (direction, direction * sign)
            gains[direction] += weight
            active += 1
    assert gains == {1: 1.0, -1: -1.0}
    return ("opposite_pair" if active == 2 else "repeated_opposite_pair", active)


def recurrent_or_sort_structure(task, width, p):
    if task == "sum" and width == 1:
        assert p.tolist() == [1, 1, 1]
        return "scalar_accumulator"
    if task == "fib" and width == 2:
        forms = ([1, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 1, 1, 0])
        assert p.tolist() in forms
        return "fibonacci_state_" + str(forms.index(p.tolist()) + 1)
    if task == "tri" and width == 3:
        forms = (
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        )
        assert p.tolist() in forms
        return "triangular_state_" + str(forms.index(p.tolist()) + 1)
    if task == "sort" and width >= 5:
        gates = []
        for w, v in p.reshape(width, 2, 4):
            if not np.any(w) or not np.any(v):
                continue
            assert sorted(w) == [-1, 0, 0, 1] and np.array_equal(v, -w)
            gates.append((int(w.argmax()), int(w.argmin())))
        assert len(gates) == 5
        for permutation in itertools.permutations(range(4)):
            a = list(permutation)
            for i, j in gates:
                a[i], a[j] = min(a[i], a[j]), max(a[i], a[j])
            assert a == sorted(permutation)
        return "compare_exchange_" + "_".join(str(i + 1) + str(j + 1) for i, j in gates)
    return ""


def main():
    with (ROOT / "focus/endpoints/endpoints.csv").open() as f:
        observed = {
            (r["task"], int(r["size"]), int(r["seed"]), r["method"]): r
            for r in csv.DictReader(f)
            if r["method"] in ("formula", "examples") and r["split"] == "outer"
        }
    assert len(observed) == 174
    events = Counter()
    rows = []
    for folder in ("objective", "arithmetic"):
        root = ROOT / "focus" / folder
        for case in load_json(root / "manifest.json"):
            task, seed, arm, uid = (case[k] for k in ("task", "seed", "arm", "id"))
            base = root / "trials" / uid
            meta = load_json(base.with_suffix(".meta.json"))
            prompt = base.with_suffix(".prompt.txt").read_text()
            assert (
                meta["requested_model"] == "gpt-6-astra" and meta["effort"] == "medium"
            )
            assert meta["completed_items"] == {"agent_message": 1}
            events.update(meta["completed_items"])
            assert "primary objective" in prompt.lower() and "tools" in prompt.lower()
            values = parse_reply(base.with_suffix(".reply.txt"))
            assert len(values) == len(case["sizes"])
            for j, width in enumerate(case["sizes"]):
                p = np.asarray(values[j], dtype=float)
                p0 = (
                    arithmetic_initial(width, seed)
                    if folder == "arithmetic"
                    else old_initial(task, width, seed)
                )
                assert p.shape == p0.shape and np.isfinite(p).all()
                parity = 0
                defect = where = None
                if task in ("sin", "tanh", "square"):
                    parity = 1 if task == "square" else -1
                    defect, where = symmetry(p, width, parity)
                kind, attribute = (
                    arithmetic_structure(task, width, p)
                    if folder == "arithmetic"
                    else ("", "")
                )
                outcome = observed.pop((task, width, seed, arm))
                rows.append(
                    dict(
                        task=task,
                        size=width,
                        seed=seed,
                        arm=arm,
                        reply=str(base.with_suffix(".reply.txt").relative_to(ROOT)),
                        array_index=j,
                        slots=p.size,
                        changed_slots=int(np.count_nonzero(p != p0)),
                        outer_exact=int(outcome["exact_count"]),
                        outer_n=int(outcome["n_examples"]),
                        outer_rmse=outcome["raw_rmse"],
                        arithmetic_structure=kind,
                        offset_or_active_units=attribute,
                        recurrent_or_sort_structure=recurrent_or_sort_structure(
                            task, width, p
                        ),
                        target_parity={0: "", -1: "odd", 1: "even"}[parity],
                        exact_real_parity=(defect == 0) if parity else "",
                        max_real_parity_defect=float(defect) if parity else "",
                        max_at_positive_x=float(where) if parity else "",
                    )
                )
    assert not observed and len(rows) == 174 and sum(events.values()) == 60
    with (HERE / "arrays.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    print("60 replies,", len(rows), "arrays,", events)
    print(
        "Total scalar slots",
        sum(r["slots"] for r in rows),
        "changed",
        sum(r["changed_slots"] for r in rows),
    )
    for t in ("square", "sin", "tanh"):
        for d in (1, 4, 16):
            values = [r for r in rows if r["task"] == t and r["size"] == d]
            print(
                t,
                d,
                sum(r["exact_real_parity"] for r in values),
                "/",
                len(values),
                "exactly symmetric",
            )
    for t in ("add", "subtract"):
        print(t, Counter(r["arithmetic_structure"] for r in rows if r["task"] == t))


if __name__ == "__main__":
    main()
