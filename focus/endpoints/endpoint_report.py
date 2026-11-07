"""Produce self-contained table inputs, matched-pair summaries and certificate scores."""
import math
from collections import Counter
from statistics import median
from analyze import (HERE, ROOT, FITS, METHODS, SPLITS, WRITERS, SOLVERS,
                     fmt, index, key, read, save)

CAPTIONS = {
    "train": "observed training examples",
    "dev": "independent central examples or, for the two integer sequences, unseen indices 9 through 32",
    "outer": (r"the extrapolation test for weight-space programming (WSP) and gradient training. "
              r"Continuous inputs use magnitudes up to 3 for training and $[-10,-3]$ or $[3,10]$ for testing, "
              r"with positive self-power tested only on $[3,10]$. Fibonacci and triangular numbers use indices "
              r"0 through 8 for training and 33 through 64 for testing"),
}


def size_tables():
    pieces = []
    for split in ("outer", "dev", "train"):
        content = (HERE / f"sizes_{split}.tex").read_text()
        title = CAPTIONS[split]
        pieces.append(r"\begin{table}[htbp]" + "\n" + r"\centering" + "\n" +
                      r"\caption{Exact matches and raw root mean squared error (RMSE) on " + title +
                      ". Each cell contains the median across three starts. Matches count complete examples, including every output of a list or stream. An integer index counts as one example. RMSE includes every scalar output. Size is hidden width except in sorting, where it counts residual blocks. Slots count stored coefficients.}" +
                      f"\\label{{tab:sizes_{split}}}\n" + r"""\scriptsize
\setlength{\tabcolsep}{2pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrr *{8}{r}@{}}
\toprule
 & & & \multicolumn{2}{c}{\shortstack{WSP\\Examples only}} &
 \multicolumn{2}{c}{\shortstack{WSP\\Formula and examples}} &
 \multicolumn{2}{c}{\shortstack{Gradient\\Adam + BFGS}} &
 \multicolumn{2}{c}{\shortstack{Gradient\\Least squares}}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Target & Size & Slots & Exact & RMSE & Exact & RMSE & Exact & RMSE & Exact & RMSE\\
\midrule
""" + content + r"""\bottomrule
\end{tabular}}
\end{table}
""")
    (HERE / "endpoint_tables.tex").write_text("\n".join(pieces))


def pair_table(summary):
    rows=[]
    for w in WRITERS:
        for bp in SOLVERS:
            data=[r for r in summary if r["scope"]=="all" and r["wsp"]==w and r["gradient"]==bp]
            def match(split,metric):
                return next(r for r in data if (r["split"],r["metric"])==(split,metric))
            a=match("train","train_loss")
            b=match("outer","raw_rmse")
            counts=(" / ".join(str(a[k]) for k in ("wsp_smaller","equal","bp_smaller")),
                    " / ".join(str(a[k]) for k in ("wsp_smaller_1e_10","within_1e_10","bp_smaller_1e_10")),
                    " / ".join(str(b[k]) for k in ("wsp_smaller","equal","bp_smaller")))
            rows.append(" & ".join(map(str,(METHODS[w],METHODS[bp],a["settings"],*counts)))+r" \\")
    (HERE/"paired_table.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Within-setting comparisons for all four choices of endpoints. A setting specifies a task, architecture and starting weights. Each triple counts settings with a smaller WSP error, a tie, or a smaller gradient error, in that order. Strict comparisons use the unrounded recorded scores. The middle column treats absolute training-loss differences of at most $10^{-10}$ as ties. Each method received the same training examples and objective. The Formula and examples condition also received the rule.}
\label{tab:paired_methods}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llr rrr@{}}\toprule
 & & & \multicolumn{2}{c}{Observed training loss} & Outer RMSE\\
\cmidrule(lr){4-5}\cmidrule(l){6-6}
WSP condition & Gradient optimizer & Settings & Strict & $10^{-10}$ tolerance & Strict\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def optimum(records):
    source = {key(r):r for r in read(ROOT/"extension/optimizer/results.csv")
              if r["method"]=="certificate_write" and r["real_optimum"]!=""}
    rows=[]
    for k,g in index(records).items():
        # Evaluate the proposed endpoints against the same objective. Exploratory
        # starts with a model proposal are not used to choose a headline endpoint.
        current=[g[m,"train"] for m in METHODS if (m,"train") in g]
        if not current: continue
        best=min(x["train_loss"] for x in current)
        for r in current:
            cert=source.get(k) if k[0] not in ("add","subtract") else None
            real=float(cert["real_optimum"]) if cert is not None else None
            rows.append(dict(task=k[0],size=k[1],seed=k[2],method=r["method"],train_loss=r["train_loss"],
                             best_of_four_loss=best,gap_to_best_of_four=r["train_loss"]-best,
                             within_1e_10_of_best=r["train_loss"]-best<=1e-10,
                             real_certificate=cert["certificate"] if cert else "",
                             certified_real_loss=real if cert else "",
                             within_1e_10_of_real=abs(r["train_loss"]-real)<=1e-10 if real is not None else "",
                             observed_zero_loss_is_binary64_min=r["train_loss"]==0))
    save("optimality.csv",rows)
    return rows


def score_overview(records, opt):
    groups=index(records)
    lines=[]
    for m in ("xavier", "examples","formula","adam_bfgs","ls_xavier"):
        tr=[g[m,"train"] for g in groups.values() if (m,"train") in g]
        dev=[g[m,"dev"] for g in groups.values() if (m,"dev") in g]
        outer=[g[m,"outer"] for g in groups.values() if (m,"outer") in g]
        def perfect(rr):
            return sum(r["exact_count"]==r["n_examples"] for r in rr)
        bestrows=[r for r in opt if r["method"]==m]
        certified=[r for r in bestrows if r["real_certificate"] in ("affine","observed_knot","interior_intersection") or
                   (r["task"] in ("square","sin","selfpower","tanh") and r["size"]==1)]
        lines.append(dict(method=m,settings=len(tr),median_train_loss=median(r["train_loss"] for r in tr),
                          zero_training_loss=sum(r["train_loss"]==0 for r in tr),all_training_examples_exact=perfect(tr),
                          all_dev_examples_exact=perfect(dev),all_outer_examples_exact=perfect(outer),
                          within_1e_10_of_best=sum(r["within_1e_10_of_best"] for r in bestrows) if bestrows else "",
                          certified_one_hinge_settings=len(certified) if bestrows else "",
                          one_hinge_within_1e_10=sum(r["within_1e_10_of_real"] for r in certified) if bestrows else ""))
    save("score_overview.csv",lines)


def fig():
    (HERE/"figure1.tex").write_text(r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figure1.pdf}
\caption{The first WSP reply given examples only and the Adam--BFGS gradient fit start from the same weights on the same task and architecture. Each bar counts how often one method's endpoint has smaller error or the two unrounded recorded errors agree. Even a very small strict difference is counted. Panel A uses their common loss on the labeled training examples. Panel B uses raw root mean squared error (RMSE) on held-out outer inputs in each target's own units. Each task contributes three sizes with three starts, except running sum with two sizes and three starts. Every listed setting has a certified nonconvex training objective.}\label{fig:paired_endpoints}
\end{figure}
""")


def write_report(records, aggregates):
    pair_table(aggregates)
    size_tables()
    fig()
    scores=optimum(records)
    score_overview(records,scores)
