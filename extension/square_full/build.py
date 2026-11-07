"""Build the explicitly engineered all-domain repair of blind graph 3354251e2."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthesis"
sys.path.insert(0, str(SYNTH))
from experiment import Compiler

ORIGINAL = SYNTH / "trials" / "3354251e2.reply.txt"
ORIGINAL_SHA256 = "aca0c3ddf093c8ed86cc50e136bc586f1a239e2547524d965694e41827f1ac55"
POWERS = [1 << i for i in range(10)]


def construct():
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA256
    original = json.loads(raw)
    end = next(i for i, ins in enumerate(original) if ins[0] == "v") + 1
    graph = list(original[:end])

    # Move the initial factor 256 ahead of every inverse rescaling, so a
    # representable normal result cannot lose bits in an earlier subnormal.
    graph.append(["v", "256*v"])
    for k in POWERS:
        graph.extend([["h", f"relu(v-256*f{k})"], ["v", f"h+(v-h)*2**-{2*k}"]])
    graph.extend([
        ["S", "+".join(f"{k}*f{k}" for k in POWERS)],
        ["normalflag", "relu(1-relu(S-514))"],
        ["normal", "relu(v-256*(1-normalflag))"],
        ["rem", "hi"], ["prefix", "0"], ["units", "0"],
    ])

    # In subnormal units, y=P/2**(54+j), P=hi*2**54+mid*2**27+lo.
    # For even j>0, prefix is its integer part and bit[j-1] its guard.
    # No exact square is an exact half of the minimum-subnormal spacing.
    def select(s, candidate):
        return [
            ["select", f"relu(1-relu(S-{s})-relu({s}-S))"],
            ["units", f"units+relu(({candidate})-2**52*(1-select))"],
        ]

    for i in reversed(range(52)):
        graph.extend([["bit", f"relu(1-relu(2**{i}-rem))"], ["rem", f"rem-bit*2**{i}"]])
        if i % 2:
            graph.extend(select(515 + (i + 1) // 2, "prefix+bit"))
        graph.append(["prefix", "2*prefix+bit"])
    graph.append(["guard", "relu(1-relu(2**26-mid))"])
    graph.extend(select(515, "hi+guard"))
    graph.append(["y", "normal+units*2**-1074"])
    return original, original[:end], graph


def compile_graph(graph):
    compiler = Compiler("affine_relu")
    compiler.block(graph)
    return compiler


if __name__ == "__main__":
    original, shared, graph = construct()
    (HERE / "program.json").write_text(json.dumps(graph, separators=(",", ":")) + "\n")
    for name, instructions in [("blind", original), ("shared", shared), ("engineered", graph)]:
        print(name, len(compile_graph(instructions).ops))
