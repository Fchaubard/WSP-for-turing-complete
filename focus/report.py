"""Generate the ten-task outer comparison and headline quantities from measured endpoints."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_dir = root / "focus/endpoints"
methods = ("examples", "formula", "adam_bfgs", "ls_xavier")
labels = dict(square="Square", sin="Sine", selfpower="Positive self-power", tanh="Hyperbolic tangent",
              add="Addition", subtract="Subtraction", sum="Running sum", sort="Sort four",
              fib="Fibonacci", tri="Triangular")
representative = dict(square=16,sin=16,selfpower=16,tanh=16,add=2,subtract=2,sum=1,sort=5,fib=2,tri=3)
def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def format_rmse(x):
    if x == 0: return "0"
    if 0.001 <= x < 1000: return f"{x:.3g}"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"\({mantissa}" + r"\!\times\!10^{" + str(int(exponent)) + r"}\)"

rows = read(data_dir / "endpoints.csv")
avail = {r["method"] for r in rows}
if "ls_xavier" not in avail:
    raise ValueError(f"Unknown method labels {avail}")
grp = defaultdict(list)
for r in rows:
    if r["method"] in methods:
        grp[r["task"],int(r["size"]),r["method"],r["split"]].append(r)
outer = []
for task, size in representative.items():
    parts = [labels[task],str(size)]
    for m in ("formula","examples","adam_bfgs","ls_xavier"):
        g = grp[task,size,m,"outer"]
        if len(g) != 3:
            parts.extend(["--","--"])
            continue
        parts.extend([f"{100*statistics.median(float(z['exact_fraction']) for z in g):.1f}",
                      format_rmse(statistics.median(float(z['raw_rmse']) for z in g))])
    outer.append(" & ".join(parts)+r"\\")
(root / "paper/focus_selected_rows.tex").write_text("\n".join(outer) + "\n" + r"\bottomrule" + "\n")
stats = {}
for m in methods:
    g = [r for r in rows if r["method"] == m and r["split"] == "train"]
    ct = {s:sum(float(r["exact_fraction"])==1 for r in rows if r["method"]==m and r["split"]==s)
          for s in ("train","dev","outer")}
    opt = { (r["task"],int(r["size"]),int(r["seed"])):float(r["real_optimum"])
            for r in read(root / "extension/optimizer/results.csv") if r["real_optimum"] and r["certificate"] in ("affine","observed_knot","interior_intersection") and r["task"] in ("square","sin","selfpower","tanh") and int(r["size"])==1}
    certified = [float(r["train_loss"]) - opt[r["task"],1,int(r["seed"])] for r in g if (r["task"],int(r["size"]),int(r["seed"])) in opt]
    stats[m] = dict(n=len(g), all_exact=ct, near_certified=sum(abs(v)<=1e-10 for v in certified),
                    n_certified=len(certified), train_near_zero=sum(float(r["train_loss"])<=1e-10 for r in g))
(root / "focus/summary.json").write_text(json.dumps(stats, indent=2)+"\n")
print(json.dumps(stats))

