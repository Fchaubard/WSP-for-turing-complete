"""Evaluate the archived affine/ReLU circuit on decimal or hexadecimal binary64 inputs."""
import json
import math
import sys

import numpy as np
from build import HERE, compile_graph


def parse_value(text):
    val = float.fromhex(text) if text.lower().lstrip("+-").startswith("0x") else float(text)
    if not math.isfinite(val) or abs(val) > 10:
        raise ValueError("expected a finite binary64 input in [-10,10]")
    return val


def main(args):
    if not args:
        raise SystemExit("Usage: python extension/square_full/infer.py NUMBER [NUMBER ...]")
    try:
        x = np.array([parse_value(arg) for arg in args], dtype=np.float64)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model = compile_graph(json.loads((HERE / "program.json").read_text()))
    # Bound the peak storage needed by the evaluator, which retains every wire.
    for start in range(0, len(x), 512):
        batch = x[start:start+512]
        for val, square in zip(batch, model.run(batch)):
            print(f"{float(val).hex()} -> {float(square).hex()}  ({float(square)!r})")


if __name__ == "__main__":
    main(sys.argv[1:])
