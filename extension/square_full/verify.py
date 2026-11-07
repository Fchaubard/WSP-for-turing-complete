"""Stress exact rounding on signed binary64 inputs with an independent rational oracle."""
import collections
from fractions import Fraction
import hashlib
import json
import math
import platform
import random
import struct

import numpy as np
from build import HERE, ORIGINAL_SHA256, compile_graph, construct
from literal_square_certificate import shell_stress


def bits(x):
    return struct.unpack(">Q", struct.pack(">d", x))[0]


def signed(points):
    return {bits(sign * float(x)) for x in points for sign in (-1, 1) if 0 <= x <= 10}


def neighborhood(x, n):
    yield x
    for toward in [0.0, math.inf]:
        y = x
        for _ in range(n):
            y = math.nextafter(y, toward)
            yield y


def stress_suites():
    rng = random.Random(3354251)
    # Powers of two include the bottom subnormal, minimum normal, normalization
    # decisions, the full subnormal output region, and ordinary shell exponents.
    boundaries = {x for e in range(-1074, 4) for x in neighborhood(2.0**e, 3)}
    boundaries.update(neighborhood(10.0, 3))
    exponent_samples = set()
    for e in range(-1074, 4):
        for _ in range(14):
            exponent_samples.add(math.ldexp(rng.randrange(2**52, 2**53), e - 52))
    # Pair adjacent inputs on either side of the mathematical square-root of
    # subnormal output half-steps, including both extreme transitions.
    indices = {0, 1, 2, 2**52 - 2, 2**52 - 1}
    for e in range(52):
        for i in range(40):
            indices.add(rng.randrange(2**e, 2**(e+1)))
    half_steps = {x for n in indices
                  for x in neighborhood(math.ldexp(math.sqrt(n + 0.5), -537), 2)}
    # Normal RN-even ties and modular near-misses tested on the blind shell
    # also occur safely just above minimum normal and at ordinary exponents.
    shell = shell_stress()
    normal_ties = {math.ldexp(math.frexp(abs(x))[0], e)
                   for label in ("round_to_even_ties", "near_midpoints")
                   for x in shell[label] for e in (-510, -509, -508, -7, 0, 3)}
    return dict(
        signed_zeros_and_blind_failure=signed([0, float.fromhex("0x1.0000000000001p-508")]),
        powers_and_neighbors=signed(boundaries),
        stratified_binades=signed(exponent_samples),
        subnormal_half_step_neighbors=signed(half_steps),
        normal_ties_and_near_ties=signed(normal_ties),
        original_shell_stress=signed({abs(x) for points in shell.values() for x in points}),
    )


def graph_cost(c):
    kinds = dict(collections.Counter(entry[0] for entry in c.ops))
    constants = sum(t[0] == "const" for kind, a, b in c.ops
                    for t in ([a] if kind == "relu" else [a,b]))
    reachable = {c.env["y"][1]} if "y" in c.env else {len(c.ops)}
    for i in range(len(c.ops), 0, -1):
        if i in reachable:
            reachable.update(term[1] for term in c.ops[i-1][1:] if term[0] == "wire")
    return dict(operations=len(c.ops), operation_types=kinds,
                reachable_operations=len(reachable - {0}), literal_operand_occurrences=constants)


def main():
    blind, shared, engineered = construct()
    saved = json.loads((HERE / "program.json").read_bytes())
    assert saved == engineered
    circuit = compile_graph(saved)
    suites = stress_suites()
    inputs = sorted(set.union(*suites.values()))
    raw = np.array(inputs, dtype=np.uint64)
    x = raw.view(np.float64)
    y = np.array([float(Fraction.from_float(float(t))**2) for t in x], dtype=np.float64)
    pred = np.concatenate([circuit.run(x[i:i+512]) for i in range(0, len(x), 512)])
    bad = set(map(int, raw[pred.view(np.uint64) != y.view(np.uint64)]))
    results = {label: dict(inputs=len(points), exact_bits=len(points-bad))
               for label, points in suites.items()}
    for value in sorted(bad)[:12]:
        i = inputs.index(value)
        print("mismatch", x[i].hex(), pred[i].hex(), y[i].hex())
    q = float.fromhex("0x1.0000000000001p-508")
    witness = next(i for i, t in enumerate(x) if t == q)
    original = compile_graph(blind)
    out = dict(
        original_sha256=ORIGINAL_SHA256,
        program_sha256=hashlib.sha256((HERE / "program.json").read_bytes()).hexdigest(),
        original=graph_cost(original), shared_expanded_operations=len(compile_graph(shared).ops),
        engineered=graph_cost(circuit), exact_bits=len(inputs)-len(bad), distinct_inputs=len(inputs),
        suites=results, oracle="float(Fraction.from_float(x)**2); both signed zero inputs included",
        blind_failure=dict(x=q.hex(), blind=float(original.run(np.array([q]))[0]).hex(),
                           repaired=float(pred[witness]).hex(), correct=float(y[witness]).hex()),
        python=platform.python_version(), numpy=np.__version__)
    print(json.dumps(out, indent=2))
    assert not bad
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
