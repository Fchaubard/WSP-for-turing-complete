"""Literal WSP repeats with a matched training objective on the original nonconvex graphs."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0, str(root / "extension/writer"))
from design import SEEDS, SIZES, TASKS, initial, make_prompt, observed, predict
from compare import parse, score


def make_cases():
    cases = []
    for task in TASKS:
        if task in ("add", "subtract"):
            continue
        sizes = [s for s in SIZES[task] if task != "sum" or s > 0]
        for seed in SEEDS:
            for arm in ("formula", "examples"):
                case_id = hashlib.sha256(f"objective-v1:{task}:{seed}:{arm}".encode()).hexdigest()[:12]
                prompt = make_prompt(task, seed, arm)
                start = prompt.index(". Aim for exact outputs")
                end = prompt.index("Do not use any tools", start)
                prompt = (prompt[:start] + ". Your primary objective is to minimize the training loss on the supplied observations: half the mean squared error after dividing each scalar prediction error by max(1, abs(target)). Include every output coordinate in that mean. Among similarly fitting solutions, aim for exact outputs on unseen inputs across the stated domain. You may infer a rule from observations. " + prompt[end:])
                if task == "sum":
                    prompt = prompt.replace("Three architectures", "Two architectures").replace("[vector0,vector1,vector2]", "[vector0,vector1]")
                    lines = [line for line in prompt.splitlines() if not line.startswith("Variant 0 (")]
                    prompt = "\n".join(line.replace("Variant 1 (", "Variant 0 (").replace("Variant 2 (", "Variant 1 (") for line in lines)+"\n"
                (here / "trials" / (case_id + ".prompt.txt")).write_text(prompt)
                cases.append(dict(id=case_id, task=task, seed=seed, arm=arm, sizes=sizes))
    (here / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")


def replay():
    rows = []
    status = []
    for case in json.loads((here / "manifest.json").read_text()):
        task, seed, arm = (case[k] for k in ("task", "seed", "arm"))
        path = here / "trials" / (case["id"] + ".reply.txt")
        values = None
        if path.exists():
            try:
                values = parse(path.read_text())
            except (ValueError, IndexError):
                pass
        for j, size in enumerate(case["sizes"]):
            try:
                w = np.asarray(values[j], dtype=float)
                valid = len(values) == len(case["sizes"]) and w.shape == initial(task, size, seed).shape and np.isfinite(w).all()
            except (TypeError, IndexError, ValueError):
                valid = False
            status.append(dict(id=case["id"], task=task, size=size, seed=seed, method=arm, status="valid" if valid else "invalid" if path.exists() else "missing"))
            if not valid:
                continue
            x,y = observed(task, seed)
            training_loss = .5 * np.mean(((predict(task,size,w,x) - y) / np.maximum(1, np.abs(y)))**2)
            for r in score(task,seed,size,w,arm):
                rows.append(dict(task=task,size=size,params=w.size,seed=seed,method=arm,split=r["split"],n=r["n"],exact=r["exact"],rmse=r["rmse"],train_loss=training_loss))
    for filename,data in [("status",status), ("scores",rows)]:
        if data:
            with (here / (filename + ".csv")).open("w", newline="") as f:
                out=csv.DictWriter(f, fieldnames=data[0]); out.writeheader(); out.writerows(data)
    print(sum(r["status"]=="valid" for r in status), "valid arrays out of", len(status))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=["design","score"])
    if parser.parse_args().action == "design":
        make_cases()
    else:
        replay()
