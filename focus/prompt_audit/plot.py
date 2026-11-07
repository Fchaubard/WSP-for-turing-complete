"""Plot the saved square example used for the prompt/selection diagnosis."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"extension/writer"))
from design import predict


def main():
    state=json.loads((ROOT/"focus/iterative/trajectories/3b244afdc6df.json").read_text())["states"]["16"]
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"font.size":9})
    fig, axes=plt.subplots(1,2,figsize=(7.1,2.6))
    intervals=[(-10,10,"Across the full domain"),(-3,3,"Inside the feedback interval")]
    for ax,(left,right,title) in zip(axes,intervals):
        x=np.linspace(left,right,1001)
        ax.plot(x,x*x,color="#253843",lw=1.4,label="Target square")
        ax.plot(x,predict("square",16,np.asarray(state["initial"]),x),color="#477D7A",lw=1.8,label="One-shot WSP")
        ax.plot(x,predict("square",16,np.asarray(state["weights"]),x),color="#B2563E",lw=1.8,label="Iterative WSP")
        ax.set(xlabel="Input",ylabel="Prediction",title=title,xlim=(left,right))
        ax.axvspan(-3,3,color="#e7edf0",zorder=-1)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(HERE/"square_continuation.pdf")
    fig.savefig(HERE/"square_continuation.png",dpi=200)


if __name__=="__main__":
    main()
