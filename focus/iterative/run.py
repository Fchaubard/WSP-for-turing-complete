"""Generate central-only prompts, dispatch independent requests and freeze selected weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extension/writer"))
from design import TASKS, SMOOTH, SERIES, observed, reference, predict

sys.path.insert(0, str(ROOT / "focus/arithmetic"))
from fixtures import forward

sys.path.insert(0, str(ROOT))
from error_metrics import rmse

# Import the isolated runner by path to avoid its common filename.
spec = importlib.util.spec_from_file_location(
    "literal_runner", ROOT / "extension/writer/run.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PROTOCOL = "75c8a10bc407ddd2808e3a7101a82fcda21fe45b42c8d28ecd48cad31cc88638"


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    Path(f.name).replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(raw):
    if raw.strip().startswith(chr(96) * 3):
        raw = raw.strip().split("\n", 1)[1].rsplit(chr(96) * 3, 1)[0]
    return json.loads(raw)


def central(task, seed):
    x, y = observed(task, seed)
    if task in SERIES:
        dev = np.arange(9, 17)
    else:
        rng = np.random.default_rng(7810000 + 10000 * TASKS.index(task) + seed)
        shape = (
            (256,)
            if task in SMOOTH
            else (256, 2 if task in ("add", "subtract") else 4 if task == "sort" else 6)
        )
        dev = rng.uniform(0 if task == "selfpower" else -3, 3, shape)
    return {"train": (x, y), "dev": (dev, reference(task, dev))}


def predict_one(task, width, p, x):
    return (
        forward(width, p, x)[0]
        if task in ("add", "subtract")
        else predict(task, width, p, x)
    )


def measure(task, width, p, data, show_worst=False):
    out = {}
    with np.errstate(all="ignore"):
        for split, (x, y) in data.items():
            pred = predict_one(task, width, p, x)
            norm = (pred - y) / np.maximum(1.0, np.abs(y))
            loss = float(0.5 * np.mean(norm**2))
            if not np.isfinite(pred).all() or not np.isfinite(loss):
                return None
            size = len(x)
            matched = (pred == y).reshape(size, -1).all(axis=1)
            record = dict(
                loss=loss, exact=int(matched.sum()), n=size, rmse=rmse(pred, y)
            )
            if show_worst:
                scaled = np.max(np.abs(norm).reshape(size, -1), axis=1)
                rows = np.flatnonzero(~matched)
                chosen = rows[np.argsort(-scaled[rows], kind="stable")[:3]]
                record["examples"] = [
                    dict(
                        input=x[i].tolist(),
                        target=y[i].tolist(),
                        prediction=pred[i].tolist(),
                        maximum_scaled_error=float(scaled[i]),
                    )
                    for i in chosen
                ]
            out[split] = record
    return out


def exact(scores):
    return all(q["exact"] == q["n"] for q in scores.values())


def pair(scores):
    return scores["dev"]["loss"], scores["train"]["loss"]


def prompt_for(case, states, active, data, round_number):
    original = (
        ROOT / "focus" / case["source"] / "trials" / (case["id"] + ".prompt.txt")
    ).read_text()
    detail = []
    for width in active:
        p = states[str(width)]["weights"]
        detail.append(
            dict(
                size=width,
                incumbent_weights=p,
                diagnostic=measure(case["task"], width, np.array(p), data, True),
            )
        )
    data_text = json.dumps(detail, separators=(",", ":"), allow_nan=False)
    return (
        "You are revising previously returned numerical neural-network weights. Use no tools or external files. "
        "Here is the original task specification and the exact original initial arrays. Treat its output formatting as applying only to the original request.\n"
        + original
        + "\nEND ORIGINAL TASK.\n"
        + f"REVISION {round_number}. These are the current incumbent vectors for the active architectures, in the listed order, with training and development diagnostics. "
        "The development inputs are new central observations, or the early continuation of a sequence. Each loss is half the mean of squared coordinate errors divided by max(1,abs(target)) squared. "
        "Smaller development loss improves the candidate; among identical development losses smaller training loss improves it. Exact counts require every coordinate of an example to match. "
        "For each split we show up to three worst examples with nonzero errors. Seek a rule that also generalizes to examples not shown. "
        "Revise each current vector within its specified architecture. You may keep it as is. Return ONLY a JSON array of complete numeric arrays in the active order shown below, "
        "even if only one size is active. Do not return vectors for stopped sizes, explanations or expressions. "
        "The original task's input support remains the same.\n" + data_text + "\n"
    ), detail


def valid_vector(raw, n):
    if (
        not isinstance(raw, list)
        or len(raw) != n
        or any(type(v) not in (int, float) for v in raw)
    ):
        return None
    try:
        p = np.array(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return p if p.shape == (n,) and np.isfinite(p).all() else None


def decide(state, p, score, old, iteration, ident):
    accepted = score is not None and pair(score) < pair(old)
    if accepted:
        state["weights"] = p.tolist()
        state["accepted_round"] = iteration
        state["reject_streak"] = 0
    else:
        state["reject_streak"] += 1
    state["attempts"] += 1
    chosen = score if accepted else old
    state["stop"] = (
        "exact_train_dev"
        if exact(chosen)
        else "two_nonimprovements"
        if state["reject_streak"] >= 2
        else "round_limit"
        if iteration >= 3
        else ""
    )
    state["history"].append(
        dict(
            round=iteration,
            request_id=ident,
            proposal=p.tolist() if p is not None else None,
            valid=score is not None,
            accepted=accepted,
            before_train=old["train"],
            before_dev=old["dev"],
            proposal_train=score["train"] if score else None,
            proposal_dev=score["dev"] if score else None,
            incumbent_train=chosen["train"],
            incumbent_dev=chosen["dev"],
        )
    )


def process(case, replay):
    task, seed = case["task"], case["seed"]
    data = central(task, seed)
    source = ROOT / "focus" / case["source"] / "trials" / (case["id"] + ".reply.txt")
    initial = parse(source.read_text())
    states = {}
    for width, raw in zip(case["sizes"], initial, strict=True):
        p = np.asarray(raw, dtype=float)
        score = measure(task, width, p, data)
        assert score is not None
        states[str(width)] = dict(
            initial=p.tolist(),
            weights=p.tolist(),
            attempts=0,
            accepted_round=0,
            reject_streak=0,
            initial_train=score["train"],
            initial_dev=score["dev"],
            history=[],
            stop="exact_train_dev" if exact(score) else "",
        )
    trace = dict(case, protocol_sha256=PROTOCOL, states=states)
    path = HERE / "trajectories" / (case["id"] + ".json")
    for iteration in range(1, 4):
        active = [width for width in case["sizes"] if not states[str(width)]["stop"]]
        if not active:
            break
        ident = case["id"] + "-r" + str(iteration)
        base = HERE / "trials" / ident
        prompt, detail = prompt_for(case, states, active, data, iteration)
        prompt_path = base.with_suffix(".prompt.txt")
        if prompt_path.exists():
            assert prompt_path.read_text() == prompt, ("prompt drift", ident)
        elif replay:
            raise RuntimeError("Missing archived prompt " + ident)
        else:
            prompt_path.write_text(prompt)
        if not replay:
            write(
                base.with_suffix(".feedback.json"),
                dict(
                    case=case["id"],
                    round=iteration,
                    active=active,
                    feedback=detail,
                    prompt_sha256=sha(prompt_path),
                    protocol_sha256=PROTOCOL,
                ),
            )
        error = base.with_suffix(".error.json")
        if not error.exists() and not replay:
            start = time.monotonic()
            try:
                runner.run(dict(id=ident), HERE)
            except Exception as exc:
                write(
                    error,
                    dict(
                        error=type(exc).__name__,
                        message=str(exc)[-800:],
                        elapsed_seconds=time.monotonic() - start,
                    ),
                )
        proposal = []
        if not error.exists():
            meta = read(base.with_suffix(".meta.json"))
            if meta["completed_items"] != {"agent_message": 1}:
                raise RuntimeError("Unexpected tool event " + ident)
            try:
                proposal = parse(base.with_suffix(".reply.txt").read_text())
            except (ValueError, TypeError, IndexError):
                pass
        if not isinstance(proposal, list) or len(proposal) != len(active):
            proposal = [None] * len(active)
        for width, raw in zip(active, proposal, strict=True):
            st = states[str(width)]
            p = valid_vector(raw, len(st["weights"]))
            old = measure(task, width, np.array(st["weights"]), data)
            new = measure(task, width, p, data) if p is not None else None
            decide(st, p, new, old, iteration, ident)
        if not replay:
            write(path, trace)
        print(
            ident,
            task,
            case["arm"],
            "round",
            iteration,
            "active",
            len(active),
            flush=True,
        )
    assert all(st["stop"] for st in states.values())
    trace["complete"] = True
    if replay:
        assert read(path) == trace
    else:
        write(path, trace)
    return case["id"]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--replay", action="store_true")
    args.add_argument("--workers", type=int, default=12)
    options = args.parse_args()
    assert sha(HERE / "PROTOCOL.md") == PROTOCOL
    cases = [
        dict(c, source=folder)
        for folder in ("objective", "arithmetic")
        for c in read(ROOT / "focus" / folder / "manifest.json")
    ]
    assert len(cases) == 60 and sum(len(c["sizes"]) for c in cases) == 174
    random.Random(21871).shuffle(cases)
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(process, c, options.replay) for c in cases]
        for f in as_completed(futures):
            f.result()
    paths = sorted((HERE / "trajectories").glob("*.json"))
    assert len(paths) == 60 and all(read(p)["complete"] for p in paths)
    marker = dict(
        protocol_sha256=PROTOCOL, trajectories={p.name: sha(p) for p in paths}
    )
    if options.replay:
        assert marker == read(HERE / "FROZEN.json")
    else:
        write(HERE / "FROZEN.json", marker)
    print("FROZEN 60 complete histories", flush=True)


if __name__ == "__main__":
    main()
