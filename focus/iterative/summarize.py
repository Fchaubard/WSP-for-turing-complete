"""Summarize only frozen central-feedback trajectories, without any test data."""

from collections import Counter
import csv
import json

from run import HERE, PROTOCOL, ROOT, read, sha


def main():
    frozen = read(HERE / "FROZEN.json")
    assert frozen["protocol_sha256"] == PROTOCOL
    rows = []
    calls = {}
    input_files = []
    for name, digest in frozen["trajectories"].items():
        file = HERE / "trajectories" / name
        assert sha(file) == digest
        tr = read(file)
        original = ROOT / "focus" / tr["source"] / "trials" / tr["id"]
        input_files.extend(
            original.with_suffix(ext)
            for ext in (".prompt.txt", ".reply.txt", ".meta.json")
        )
        for width, state in tr["states"].items():
            initial_dev = state["initial_dev"]
            final_dev = (
                state["history"][-1]["incumbent_dev"]
                if state["history"]
                else initial_dev
            )
            initial_train = state["initial_train"]
            final_train = (
                state["history"][-1]["incumbent_train"]
                if state["history"]
                else initial_train
            )
            assert (final_dev["loss"], final_train["loss"]) <= (
                initial_dev["loss"],
                initial_train["loss"],
            )
            rows.append(
                dict(
                    task=tr["task"],
                    size=int(width),
                    seed=tr["seed"],
                    arm=tr["arm"],
                    case=tr["id"],
                    attempts=state["attempts"],
                    accepted=sum(r["accepted"] for r in state["history"]),
                    invalid=sum(not r["valid"] for r in state["history"]),
                    accepted_round=state["accepted_round"],
                    stop=state["stop"],
                    initial_train_loss=initial_train["loss"],
                    final_train_loss=final_train["loss"],
                    initial_dev_loss=initial_dev["loss"],
                    final_dev_loss=final_dev["loss"],
                    initial_dev_exact=initial_dev["exact"],
                    final_dev_exact=final_dev["exact"],
                    dev_n=final_dev["n"],
                )
            )
            for h in state["history"]:
                calls[h["request_id"]] = tr["arm"]
    assert len(rows) == 174
    rows.sort(key=lambda r: (r["task"], r["size"], r["seed"], r["arm"]))
    with (HERE / "selection.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    result = {
        "protocol_sha256": PROTOCOL,
        "original_cases": len(frozen["trajectories"]),
        "new_requests": len(calls),
        "requests_by_arm": dict(Counter(calls.values())),
        "stops": dict(Counter(r["stop"] for r in rows)),
        "arrays_attempted": sum(r["attempts"] for r in rows),
        "arrays_accepted": sum(r["accepted"] for r in rows),
        "invalid_or_nonfinite_attempts": sum(r["invalid"] for r in rows),
        "initially_exact": sum(r["attempts"] == 0 for r in rows),
    }
    result["selection_by_arm"] = {}
    for arm in ("examples", "formula"):
        group = [r for r in rows if r["arm"] == arm]
        changes = Counter(
            "lower"
            if r["final_train_loss"] < r["initial_train_loss"]
            else "higher"
            if r["final_train_loss"] > r["initial_train_loss"]
            else "equal"
            for r in group
        )
        result["selection_by_arm"][arm] = {
            "settings": len(group),
            "dev_loss_strictly_lower": sum(
                r["final_dev_loss"] < r["initial_dev_loss"] for r in group
            ),
            "training_loss": dict(changes),
        }
    input_files.extend(sorted((HERE / "trials").glob("*")))
    hashes = {str(file.relative_to(ROOT)): sha(file) for file in input_files}
    (HERE / "archive_sha256.json").write_text(
        json.dumps(hashes, indent=1, sort_keys=True) + "\n"
    )
    (HERE / "selection_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
