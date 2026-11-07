"""Plot shell accuracy against sweep size from archived integer hit counts."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).parent
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
fig.subplots_adjust(left=.092, right=.986, top=.85, bottom=.29, wspace=.15)
panels = [
    ("relu.csv", "A   Sine", "Rotation stages", 57, 8192),
    ("relu_power.csv", "B   Positive self-power", "Fractional bits retained", 72, 4096),
]
for ax, (filename, title, xlabel, threshold, denominator) in zip(axes, panels):
    with (root / filename).open() as source:
        rows = list(csv.DictReader(source))
    for variant, color, marker, line, label in [
        ("two_word", "#1d597f", "o", "-", "Two words"),
        ("single_word", "#9c4829", "s", "--", "One word"),
    ]:
        subset = [r for r in rows if r["method"].endswith(variant)
                  and r["split"] in ("left", "right")]
        sizes = sorted({int(r["order"]) for r in subset})
        fractions = []
        for size in sizes:
            group = [r for r in subset if int(r["order"]) == size]
            assert sum(int(r["n"]) for r in group) == denominator
            fractions.append(sum(int(r["correct"]) for r in group) / denominator)
        ax.plot(sizes, fractions, color=color, marker=marker, linestyle=line,
                markersize=3.7, linewidth=1.5, label=label, clip_on=False)
    ax.plot(threshold, 1, marker="D", markersize=6.2, markerfacecolor="white",
            markeredgecolor="#134264", markeredgewidth=1.4, zorder=4)
    ax.annotate(f"All correct at {threshold}", (threshold, 1), xytext=((7, -14) if ax is axes[0] else (0, -14)),
                textcoords="offset points", fontsize=8,
                ha="left" if ax is axes[0] else "center", color="#134264")
    ax.set(title=title, xlabel=xlabel, ylim=(-.05, 1.04))
    ax.title.set_fontsize(10)
    ax.set_xticks(sizes)
    ax.tick_params(labelsize=8, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce1e4", linewidth=.65)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axes[0].set_ylabel("Correctly rounded outputs")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.savefig(root.parents[1] / "paper" / "smooth_precision.pdf", bbox_inches="tight")
