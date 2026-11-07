"""Pair fixed first replies and gradient endpoints on verified nonconvex objectives."""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = dict(square="Square", sin="Sine", selfpower="Positive self-power", tanh="Hyperbolic tangent",
             add="Addition", subtract="Subtraction", sum="Running sum", sort="Sort four",
             fib="Fibonacci", tri="Triangular")
METHODS = dict(formula="Formula and examples", examples="Examples only",
               adam_bfgs="Adam + BFGS", ls_xavier="Least squares")
SOLVERS = ("adam_bfgs", "ls_xavier")
WRITERS = ("examples", "formula")
SPLITS = ("train", "dev", "outer")
SEEDS = (164, 317, 583)
OLD = "matched objective on eight existing graphs"
NEW = "matched objective on new arithmetic graphs"
FITS = ("xavier", *METHODS)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, records):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def key(row):
    return row["task"], int(row["size"]), int(row["seed"])


def attempts():
    rows=[]
    for folder in ("objective", "arithmetic"):
        directory=ROOT/"focus"/folder
        manifest=json.loads((directory/"manifest.json").read_text())
        results={}
        if folder=="objective" and (directory/"status.csv").exists():
            results={(r["id"],int(r["size"])):r["status"] for r in read(directory/"status.csv")}
        if folder=="arithmetic" and (directory/"response_status.json").exists():
            results={(r["id"],r["width"]):("valid" if r["valid"] else "invalid")
                     for r in json.loads((directory/"response_status.json").read_text())}
        for case in manifest:
            for size in case["sizes"]:
                rows.append(dict(protocol=OLD if folder=="objective" else NEW,request=case["id"],
                                 task=case["task"],size=size,seed=case["seed"],method=case["arm"],
                                 status=results.get((case["id"],size),"missing")))
    return rows


def record(row, source, method, split, n, scalars, exact, rmse):
    task, size, seed = key(row)
    return dict(protocol=source, task=task, size=size, params=int(row["params"]), seed=seed,
                method=method, split=split,
                split_description=("early indices 9–32" if split == "dev" and task in ("fib", "tri")
                                   else {"train": "observed", "dev": "central held-out", "outer": "outer held-out"}[split]),
                n_examples=n, n_scalars=scalars, exact_count=exact, exact_fraction=exact/n,
                raw_rmse=rmse, train_loss=float(row["train_loss"]))


def original(eligibility):
    import numpy as np
    sys.path.insert(0, str(ROOT / "extension/writer"))
    from design import observed, inputs, reference, predict
    sys.path.insert(0, str(ROOT))
    from error_metrics import rmse

    fit_rows = read(ROOT / "extension/optimizer/results.csv")
    eligible = {key(r) for r in eligibility if r["status"] == "nonconvex"}
    assert len(eligible) == 69
    records = []
    with np.load(ROOT / "extension/optimizer/states.npz", allow_pickle=False) as states:
        for r in fit_rows:
            task, size, seed = key(r)
            if (task,size,seed) not in eligible:
                continue
            method = r["method"]
            # These checkpoint names encode a former request or start with a former reply.
            label = ("archived_" + method if method in ("examples","formula","ls_formula","ls_examples") else method)
            theta = states[f"{task}_{size}_{seed}_{method}"]
            xx, yy = observed(task,seed)
            test = inputs(task)
            tr = predict(task,size,theta,xx)
            training_loss = 0.5*np.mean(((tr-yy)/np.maximum(1,np.abs(yy)))**2)
            assert math.isclose(training_loss,float(r["train_loss"]),rel_tol=1e-13,abs_tol=1e-32)
            source = dict(r, train_loss=training_loss)
            for split,x in (("train",xx),("dev",test[r["dev_split"]]),("outer",test["outer"])):
                y=reference(task,x)
                p=predict(task,size,theta,x)
                equal=p==y
                success=equal.all(axis=1) if equal.ndim==2 else equal
                exact=int(np.count_nonzero(success))
                assert exact / len(x) == float(r[split+"_exact"])
                records.append(record(source, OLD, label, split, len(x), y.size, exact, rmse(p,y)))
    return records


def common_objective(path):
    records=[]
    for r in read(path):
        split={"observed":"train", "inner":"dev", "early":"dev", "outer":"outer"}.get(r["split"])
        if split is None: continue
        n=int(r["n"])
        fraction=float(r["exact"])
        exact=round(n*fraction)
        assert exact/n==fraction and r["method"] in WRITERS
        width={"sum":6,"sort":4}.get(r["task"],1)
        records.append(record(r,OLD,r["method"],split,n,n*width,exact,float(r["rmse"])))
    return records


def arithmetic(path):
    records = []
    aliases = {"trf_xavier": "ls_xavier", "wsp_formula": "formula", "wsp_examples": "examples"}
    for r in read(path):
        method = aliases.get(r["method"], r["method"])
        n, scalars = int(r["n_examples"]), int(r["n_scalars"])
        if r["exact"]=="":
            continue
        exact = int(r["exact"])
        assert 0 <= exact <= n == scalars
        records.append(record(r, NEW, method, r["split"], n, scalars, exact, float(r["rmse"])))
    return records


def index(records):
    groups = defaultdict(dict)
    for r in records:
        ident = key(r)
        position = (r["method"], r["split"])
        if position in groups[ident]:
            raise ValueError(f"Duplicate record for {ident}, {position}")
        groups[ident][position] = r
    return groups


def compare(wsp, bp, tolerance=0):
    delta = wsp - bp
    return "wsp_smaller" if delta < -tolerance else "bp_smaller" if delta > tolerance else "tie"


def paired(records):
    output = []
    groups = index(records)
    for (task, size, seed), g in groups.items():
        for wsp, bp in itertools.product(WRITERS, SOLVERS):
            for split, metric in (("train", "train_loss"), *((s, "raw_rmse") for s in SPLITS)):
                if (wsp, split) not in g or (bp, split) not in g:
                    continue
                a, b = g[wsp, split], g[bp, split]
                assert (a["n_examples"], a["n_scalars"], a["params"]) == (b["n_examples"], b["n_scalars"], b["params"])
                x, y = a[metric], b[metric]
                output.append(dict(protocol=a["protocol"], task=task, size=size, seed=seed, params=a["params"],
                                   wsp=wsp, gradient=bp, split=split, metric=metric, wsp_score=x, bp_score=y,
                                   wsp_minus_bp=x-y, strict_outcome=compare(x, y),
                                   outcome_1e_10=compare(x, y, 1e-10) if metric == "train_loss" else "",
                                   n_examples=a["n_examples"], n_scalars=a["n_scalars"],
                                   wsp_exact_count=a["exact_count"], bp_exact_count=b["exact_count"]))
    return output


def aggregate_pairs(pairs):
    by = defaultdict(list)
    for r in pairs:
        for scope, label in (("all","all"), ("protocol",r["protocol"]), ("task",r["task"]),
                             ("architecture",f'{r["task"]}/{r["size"]}')):
            by[scope, label, r["wsp"], r["gradient"], r["split"], r["metric"]].append(r)
    out = []
    for (scope, label, wsp, bp, split, metric), rows in by.items():
        strict = Counter(r["strict_outcome"] for r in rows)
        tol = Counter(r["outcome_1e_10"] for r in rows)
        by_units = scope in ("task","architecture")
        out.append(dict(scope=scope, group=label, wsp=wsp, gradient=bp, split=split, metric=metric,
                        settings=len(rows), wsp_smaller=strict["wsp_smaller"], equal=strict["tie"], bp_smaller=strict["bp_smaller"],
                        wsp_smaller_1e_10=tol["wsp_smaller"] if metric=="train_loss" else "",
                        within_1e_10=tol["tie"] if metric=="train_loss" else "",
                        bp_smaller_1e_10=tol["bp_smaller"] if metric=="train_loss" else "",
                        median_paired_delta=median(r["wsp_minus_bp"] for r in rows) if by_units or metric=="train_loss" else "",
                        wsp_all_exact_settings=sum(r["wsp_exact_count"]==r["n_examples"] for r in rows),
                        bp_all_exact_settings=sum(r["bp_exact_count"]==r["n_examples"] for r in rows)))
    return out


def summarize(records):
    by = defaultdict(list)
    for r in records:
        by[r["task"],r["size"],r["method"],r["split"]].append(r)
    out = []
    for (task, size, method, split), rows in by.items():
        n_values = {r["n_examples"] for r in rows}
        assert len(n_values)==1
        out.append(dict(task=task,size=size,params=rows[0]["params"],method=method,split=split,
                        starts=len(rows), examples_per_start=rows[0]["n_examples"],
                        scalars_per_start=rows[0]["n_scalars"],median_exact_count=median(r["exact_count"] for r in rows),
                        median_exact_fraction=median(r["exact_fraction"] for r in rows),
                        median_raw_rmse=median(r["raw_rmse"] for r in rows),
                        median_train_loss=median(r["train_loss"] for r in rows),
                        all_exact_starts=sum(r["exact_count"]==r["n_examples"] for r in rows)))
    return out


def endpoint_summary(records):
    by = defaultdict(list)
    for r in records:
        if r["method"] not in FITS: continue
        for scope, label in (("all","all"),("protocol",r["protocol"]),("task",r["task"])):
            by[scope,label,r["method"],r["split"]].append(r)
    return [dict(scope=scope,group=label,method=m,split=s,settings=len(rr),
                 all_exact_settings=sum(r["exact_count"]==r["n_examples"] for r in rr),
                 median_raw_rmse=median(r["raw_rmse"] for r in rr) if scope=="task" else "",
                 median_train_loss=median(r["train_loss"] for r in rr))
            for (scope,label,m,s),rr in by.items()]


def compare_initial(records):
    out=[]
    for (task,size,seed),g in index(records).items():
        for method in METHODS:
            for split,metric in (("train","train_loss"), *((s,"raw_rmse") for s in SPLITS)):
                if (method,split) not in g: continue
                a,b=g["xavier",split],g[method,split]
                out.append(dict(protocol=b["protocol"],task=task,size=size,seed=seed,method=method,
                                split=split,metric=metric,start=a[metric],end=b[metric],
                                end_minus_start=b[metric]-a[metric],
                                exact_count_at_start=a["exact_count"],exact_count_at_end=b["exact_count"]))
    return out


def fmt(value):
    if value == 0: return "0"
    if not math.isfinite(value): return r"$\infty$"
    if 0.001<=abs(value)<1000: return f"{value:.3g}"
    a, b = f"{value:.2e}".split("e")
    return f"${a}"+r"\!\times\!10^{"+str(int(b))+"}$"


def write_sizes(summaries):
    # One pair of exact and RMSE columns per method. Task-specific units never mix.
    methods = ["examples","formula","adam_bfgs","ls_xavier"]
    by = {(r["task"], r["size"], r["method"], r["split"]): r for r in summaries}
    for split in SPLITS:
        lines = []
        for task,label in TASKS.items():
            sizes = sorted({r["size"] for r in summaries if r["task"]==task})
            for j,size in enumerate(sizes):
                r=by[task,size,"xavier",split]
                cells=[label if j==0 else "",str(size),str(r["params"])]
                for m in methods:
                    v=by.get((task,size,m,split))
                    cells.extend(["--","--"] if not v else
                                 [f'{v["median_exact_count"]:g}/{v["examples_per_start"]}',
                                  fmt(v["median_raw_rmse"])])
                lines.append(" & ".join(cells)+r" \\")
        (HERE/f"sizes_{split}.tex").write_text("\n".join(lines)+"\n")


def draws_figure(agg):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    panels = (("train","train_loss","A  On the training examples"),
              ("outer","raw_rmse","B  On the outer test set"))
    colors=("#266279","#dedbd2","#a44c2e")
    names = ("WSP smaller", "Equal", "Gradient smaller")
    tasks=[t for t in TASKS if any(r["scope"]=="task" and r["group"]==t and r["wsp"]=="examples" for r in agg)]
    fig,axes=plt.subplots(1,2,figsize=(7.35,2.98),sharey=True)
    with plt.rc_context({"font.family":"DejaVu Sans"}):
        for ax,(split,metric,title) in zip(axes,panels):
            relevant = {r["group"]:r for r in agg if r["scope"]=="task" and r["wsp"]=="examples"
                        and r["gradient"]=="adam_bfgs" and r["split"]==split and r["metric"]==metric}
            for y,task in enumerate(tasks):
                row = relevant[task]; start=0
                for k,color in zip(("wsp_smaller","equal","bp_smaller"),colors):
                    count=row[k]
                    if count:
                        ax.barh(y,count,left=start,height=.69,color=color)
                        ax.text(start+count/2,y,str(count),ha="center",va="center",fontsize=7.4,
                                fontweight="bold",color="#222222" if k=="equal" else "white")
                    start+=count
            ax.set(xlim=(0,9.04),xticks=[0,3,6,9],xlabel="Paired settings")
            ax.xaxis.label.set_size(8)
            ax.tick_params(axis="both",labelsize=8,length=0,pad=3)
            ax.set_title(title,loc="left",fontsize=9,fontweight="bold",pad=6)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.set_axisbelow(True);ax.grid(axis="x",color="#e0e4e6",lw=.55)
        axes[0].set_yticks(np.arange(len(tasks)),[TASKS[t] for t in tasks])
        axes[0].invert_yaxis()
        fig.legend([Patch(facecolor=c) for c in colors],names,loc="lower center",ncols=3,frameon=False,
                   handlelength=1.0,handletextpad=.45,fontsize=8,bbox_to_anchor=(.54,-.005))
        fig.subplots_adjust(left=.17,right=.99,bottom=.20,top=.91,wspace=.10)
        fig.savefig(HERE/"figure1.pdf",metadata={"Title":"Paired optimization and held-out prediction"})
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--provisional",action="store_true",help="Allow only completed experiment strata.")
    args=parser.parse_args()
    elig=ROOT/"focus/nonconvex/eligibility.csv"
    if elig.exists(): eligibility=read(elig)
    elif args.provisional:
        eligibility=[dict(r,status=("convex" if r["task"] in ("add","subtract") or (r["task"]=="sum" and r["size"]=="0") else "nonconvex"))
                     for r in read(ROOT/"extension/optimizer/results.csv") if r["method"]=="xavier"]
    else: raise FileNotFoundError(elig)
    records=original(eligibility)
    objective_path=ROOT/"focus/objective/scores.csv"
    if objective_path.exists(): records.extend(common_objective(objective_path))
    elif not args.provisional: raise FileNotFoundError(objective_path)
    arithmetic_path=ROOT/"focus/arithmetic/scores.csv"
    if arithmetic_path.exists(): records.extend(arithmetic(arithmetic_path))
    elif not args.provisional: raise FileNotFoundError(arithmetic_path)
    groups=index(records)
    status=attempts()
    save("literal_attempts.csv",status)
    if not args.provisional:
        assert len(groups)==87
        assert len(status)==174 and len({r["request"] for r in status})==60
        for r in status:
            if r["status"]=="missing":
                raise ValueError(f"Request outcome pending: {r['request']}")
            assert all(((r["method"],s) in groups[key(r)])==(r["status"]=="valid") for s in SPLITS)
        for g in groups.values():
            assert all((m,s) in g for m in ("xavier",*SOLVERS) for s in SPLITS)
    pairs=paired(records)
    agg=aggregate_pairs(pairs)
    sizes=summarize(records)
    for name,items in (("endpoints.csv",records),("paired_deltas.csv",pairs),("paired_summary.csv",agg),
                       ("sizes.csv",sizes),("by_method.csv",endpoint_summary(records)),("initialization_deltas.csv",compare_initial(records))):
        save(name,items)
    write_sizes(sizes)
    from endpoint_report import write_report
    write_report(records,agg)
    draws_figure(agg)
    print(f"{len(groups)} paired settings, {len(records)} checkpoint/split records")

if __name__=="__main__":
    main()
