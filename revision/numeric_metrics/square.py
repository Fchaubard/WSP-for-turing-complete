"""Check both WSP square circuits on exactly the same rational-reference panels."""

from fractions import Fraction
import struct
import sys
from common import EXT, metric, save
import numpy as np

sys.path.insert(0, str(EXT / "square_full"))
sys.path.insert(0, str(EXT / "audit"))
import verify
from build import construct, compile_graph
from literal_square_certificate import shell_stress
from square_certificate_review import midpoint_inputs, exponent_inputs


def main():
    old, _, new = construct()
    panels = shell_stress()
    patterns = verify.stress_suites()
    for k, vs in (patterns | {"union": set.union(*patterns.values())}).items():
        panels["full_" + k] = np.array(
            [struct.unpack(">d", struct.pack(">Q", v))[0] for v in sorted(vs)]
        )
    panels["independent_midpoints"] = midpoint_inputs()
    panels["independent_exponents"] = exponent_inputs()
    rows = []
    for method, raw in [("WSP_first_reply", old), ("WSP_engineered", new)]:
        g = compile_graph(raw)
        for name, subset in panels.items():
            x = np.asarray(sorted(subset))
            y = np.fromiter(
                (float(Fraction.from_float(float(v)) ** 2) for v in x), float
            )
            pred = np.concatenate(
                [g.run(x[i : i + 512]) for i in range(0, len(x), 512)]
            )
            score = metric(pred, y)
            bits = int((pred.view(np.uint64) == y.view(np.uint64)).sum())
            if method == "WSP_engineered":
                assert bits == len(y)
            rows.append(
                dict(task="square", method=method, split=name, exact_bits=bits, **score)
            )
    save("square", rows)


if __name__ == "__main__":
    main()
