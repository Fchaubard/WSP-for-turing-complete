"""Draw the Table 3 task charts from the same verified per-size medians."""
import csv
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from report import METHODS, TASKS

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "output"
LABELS = ["One-shot, examples only", "Iterative, examples only",
          "One-shot, formula and examples", "Iterative, formula and examples",
          "Gradient descent", "Gradient descent then L-BFGS",
          "Adam then full BFGS", "Trust-region least squares"]
ARCHITECTURE = dict(square="hidden units", sin="hidden units", selfpower="hidden units",
                    tanh="hidden units", add="hidden units", subtract="hidden units",
                    sum="state coordinate", sort="residual blocks", fib="state coordinates",
                    tri="state coordinates")
INK = "#152A33"
MUTED = "#50626A"
EXACT = "#477D7A"
ERROR = "#55738E"
POSITIONS = [1.4, 2.4, 3.4, 4.4, 6.6, 7.6, 8.6, 9.6]


def number(value):
    if value == 0:
        return "0"
    if .001 <= value < 10000:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    coefficient = mantissa.rstrip("0").rstrip(".")
    superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{coefficient}×10{superscript}"


def draw_task(fig, rect, task, medians):
    left, bottom, width, height = rect
    title, _, size = TASKS[task]
    group = [medians[task, size, "outer", m] for m in METHODS]
    total = group[0]["n"]

    def label(x, y, text, **style):
        return fig.text(left + width*x, bottom + height*y, text,
                        family="DejaVu Sans", color=INK, **style)

    label(0, .956, title, size=14, weight="bold", va="center")
    description = f"{size} {ARCHITECTURE[task]}  ·  {group[0]['params']} parameters  ·  {total:,} outer test examples"
    label(0, .879, description, size=8.7, va="center")
    # Positions are shared across all three axes; labels read across both metrics.
    x_label, x_exact, x_error = 0, .423, .720
    points = []
    for x, w in [(x_label, .410), (x_exact, .252), (x_error, .254)]:
        ax = fig.add_axes([left + width*x, bottom + height*.068, width*w, height*.667])
        ax.set_ylim(10.25, -.55)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=MUTED, labelsize=7.2, length=0, pad=3)
        ax.spines[:].set_visible(False)
        points.append(ax)
    names, exact, error = points
    names.set_xlim(0,1)
    names.set_axis_off()
    for y, group_label in [(0.25,"Weight-space programming"),(5.45,"Gradient methods")]:
        names.text(0,y,group_label,size=8.35,weight="bold",va="center",color=INK)
    for y, text in zip(POSITIONS,LABELS):
        names.text(.025,y,text,size=7.9,va="center",color=INK)
    max_error = max(row["raw_rmse"] for row in group)
    ticks = [v for v in MaxNLocator(nbins=2, steps=[1,2,2.5,5,10]).tick_values(0,max_error)
             if 0 <= v <= max_error]
    # A common zero and 3 significant digits retain the distinction between exact
    # answers and tiny residuals even when the latter are below the printer resolution.
    for ax, x, field, color, heading, subheading, limit, tick_values in [
        (exact, x_exact,"exact_count",EXACT,"Exact predictions","Higher is better",total,[0,total]),
        (error, x_error,"raw_rmse",ERROR,"Raw RMSE","Lower is better",max_error,ticks),
    ]:
        label(x,.795,heading,size=9.2,weight="bold",va="center")
        label(x,.747,subheading,size=7.6,va="center")
        ax.set_xlim(0,limit*1.22 or 1)
        ax.set_xticks(tick_values)
        ax.set_xticklabels([f"{int(v):,}" if field=="exact_count" else number(v) for v in tick_values])
        ax.grid(axis="x", lw=.55, color="#DCE4E5", zorder=0)
        ax.axvline(0,lw=.7,color="#A9B9BD",zorder=1)
        for method, y, row in zip(METHODS, POSITIONS, group):
            val=row[field]
            bar = ax.barh(y,val,height=.47,color=color,linewidth=0,zorder=2)
            bar.patches[0].set_gid(f"{field}-{method}")
            txt=f"{int(val):,}" if field=="exact_count" else number(val)
            value_label = ax.annotate(txt,(val,y),xytext=(3,0),textcoords="offset points",ha="left",va="center",
                        color=INK,fontsize=7.5,annotation_clip=False,clip_on=False)
            value_label.set_gid(f"{field}-{method}-text")


def tex_wrapper():
    out=[]
    pairs=list(zip(list(TASKS)[::2],list(TASKS)[1::2]))
    for i,(first,second) in enumerate(pairs,1):
        title=f"{TASKS[first][0]} and {TASKS[second][0].lower()}"
        out += [r"\begin{figure}[htbp]\centering" if i==1 else r"\begin{figure}[p]\centering",
                rf"\includegraphics[width=\linewidth]{{task_panels_{i}.pdf}}",
                r"\caption{"+title+r" on the extrapolation test at the sizes in Table~\ref{tab:focusouter}. "
                r"Bars show medians over three starts. Exact predictions count whole examples that match the reference. "
                r"Raw root mean squared error (RMSE) is in the target's units. Each task has its own linear error axis starting at zero.}",
                rf"\label{{fig:taskbars{i}}}",r"\end{figure}"]
    out.append(r"\clearpage")
    (HERE/"task_panels.tex").write_text("\n".join(out)+"\n")


def write_html(svgs):
    from html import escape
    options="".join(f'<option value="{k}">{escape(TASKS[k][0])}</option>' for k in TASKS)
    cards="\n".join(f'<section data-task="{k}"'+('' if k=='square' else ' hidden')+f'>{svg}</section>' for k,svg in svgs.items())
    (OUT/"table-3-task-charts.html").write_text("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extrapolation results by task</title>
<style>*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#152a33;margin:0;background:#f3f6f6}main{max-width:1000px;margin:28px auto;background:white;padding:clamp(18px,4vw,52px);border:1px solid #dce4e5;border-radius:7px}h1{font-size:clamp(22px,3vw,34px);line-height:1.2;letter-spacing:-.035em;margin:0 0 20px}nav{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}select{font:inherit;border:1px solid #8da5a6;border-radius:4px;padding:5px 25px 5px 9px;color:#152a33;background:white}a{color:#285e65}nav a{margin-left:auto}svg{width:100%;height:auto;display:block}svg text{font-family:Arial,Helvetica,sans-serif!important}small{color:#50626a}section{min-width:580px}article{overflow-x:auto}footer{border-top:1px solid #dce4e5;padding-top:12px}footer p{margin:0 0 7px}section[hidden]{display:none}@media(max-width:650px){main{margin:0;border:none;border-radius:0}nav a{margin-left:0}}@media print{main{max-width:none;margin:0;border:0}nav{display:none}}</style>
</head><body><main><h1>Extrapolation results by task</h1><nav><label for="task">Task</label>
<select id="task">"""+options+"""</select><a href="table-3-task-charts.pdf">Open all ten charts (PDF)</a></nav>
<article aria-label="Selected task chart">"""+cards+"""</article><footer><p>Each bar shows the median across three random starts. Exact predictions match the 64-bit reference on every output of an example. Raw root mean squared error (RMSE) is in the target’s units.</p>
<small>Adam is adaptive moment estimation. BFGS is the Broyden–Fletcher–Goldfarb–Shanno method. L-BFGS is its limited-memory variant.</small></footer></main>
<script>const selector=document.getElementById('task');function showTask(){for(const chart of document.querySelectorAll('[data-task]'))chart.hidden=chart.dataset.task!==selector.value;}selector.addEventListener('change',showTask);showTask();</script></body></html>""")


def main():
    medians = {}
    with (HERE / "size_medians.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["task"], int(row["size"]), row["split"], row["method"]
            if key in medians:
                raise ValueError(f"Repeated chart measurement {key}")
            medians[key] = dict(n=int(row["n"]), params=int(row["params"]),
                                exact_count=int(row["exact_count"]), raw_rmse=float(row["raw_rmse"]))
    if len(medians) != 696:
        raise ValueError("Expected 696 published task-size-method-split medians")
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none","mathtext.fontset":"dejavusans"})
    pairs=list(zip(list(TASKS)[::2],list(TASKS)[1::2]))
    metadata={"Title":"Extrapolation results by task","Author":"Francois Chaubard"}
    with PdfPages(OUT/"table-3-task-charts.pdf",metadata=metadata) as archive:
        for i,(first,second) in enumerate(pairs,1):
            fig=plt.figure(figsize=(6.4,6.65),facecolor="white")
            for task,y in [(first,.524),(second,.030)]:
                draw_task(fig,(.018,y,.947,.447),task,medians)
            fig.add_artist(Line2D([.018,.980],[.505,.505],lw=.65,color="#B8CACC",transform=fig.transFigure))
            archive.savefig(fig)
            fig.savefig(HERE/f"task_panels_{i}.pdf",metadata=metadata)
            plt.close(fig)
    svgs={}
    for task in TASKS:
        fig=plt.figure(figsize=(6.4,3.12),facecolor="white")
        draw_task(fig,(.014,.030,.947,.95),task,medians)
        with StringIO() as buffer:
            fig.savefig(buffer,format="svg")
            svg=buffer.getvalue()
        plt.close(fig)
        svg = svg[svg.index("<svg"):]
        svg = svg.replace("<svg ", '<svg role="img" aria-label="'+TASKS[task][0]+' test results for eight methods" ',1)
        for method in METHODS:
            row = medians[task, TASKS[task][2], "outer", method]
            for field, label in [("exact_count","Exact predictions"),("raw_rmse","Raw root mean squared error")]:
                value = row[field]
                for suffix in ("", "-text"):
                    opening = f'<g id="{field}-{method}{suffix}">'
                    if opening not in svg:
                        raise ValueError(f"Missing chart artist {opening}")
                    svg = svg.replace(opening,opening+f"<title>{label}: {value}</title>",1)
        svgs[task] = (svg.replace('id="',f'id="{task}-')
                        .replace('url(#',f'url(#{task}-')
                        .replace('href="#',f'href="#{task}-'))
    write_html(svgs)
    tex_wrapper()


if __name__=="__main__":
    main()
