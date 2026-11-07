"""Regenerate manuscript plots from the archived result files."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper'
def rows(path):
    with (ROOT/path).open() as f: return list(csv.DictReader(f))
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.axisbelow':True})
blue='#195782'; orange='#b34c22'; green='#178170'; gray='#676b73'
def style(ax):
    ax.grid(axis='y',color='#e1e5e8',linewidth=.65)
def fig_frontier():
    short=rows('frontier/short.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.45),layout='constrained')
    ax=axes[0]; H=np.arange(1,13)
    for d,c in [(2,gray),(4,green),(8,blue)]:
        ax.plot(H,np.minimum(H,d+1),'-o',ms=2.5,color=c,label=f'd = {d}')
    ax.set(xticks=[1,3,5,7,9,12],yticks=[1,3,5,7,9],xlabel='Number of steps to reproduce',ylabel='Fewest weights changed',title='a   Running sum: minimum edits')
    ax.legend(frameon=False,loc='upper left',fontsize=7);style(ax)
    ax=axes[1]; ds=[2,4,8,16,32]
    for k,c,l in [('fit_max',blue,'Largest error on fitted steps'),('next_abs',orange,'Next step, held out')]:
        groups=[[max(1e-17,float(r[k])) for r in short if int(r['d'])==d and int(r['H'])==d] for d in ds]
        mid,q1,q3=np.quantile(groups,[.5,.25,.75],axis=1)
        ax.fill_between(range(5),q1,q3,color=c,alpha=.16,linewidth=0)
        ax.plot(range(5),mid,'o-',ms=3,color=c,label=l)
    ax.set(yscale='log',xticks=range(5),xticklabels=ds,xlabel='State width (also the number of fitted steps)',ylabel='Absolute error on the unit impulse',title='b   Fit the prefix, test one step later')
    ax.set_yticks([1e-16,1e-12,1e-8,1e-4,1]);ax.legend(frameon=False,fontsize=7,loc='center left');style(ax)
    fig.savefig(OUT/'frontier.pdf',bbox_inches='tight');plt.close(fig)
def fig_precision():
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.42),layout='constrained')
    data=rows('scalar/compensated.csv')
    ax=axes[0]
    for met,c,label in [('write_corrected',blue,'Correction read out'),('correction_off',orange,'Two weights set to zero')]:
        for s,ls in [('in','--'),('shell','-')]:
            reg=['left','right'] if s=='shell' else ['in']
            arr=[r for r in data if r['method']==met and r['split'] in reg]
            ds=sorted({int(r['size']) for r in arr})
            ys=[np.mean([float(r['exact']) for r in arr if int(r['size'])==d]) for d in ds]
            ax.plot(ds,ys,ls,color=c,marker='o' if s=='shell' else None,ms=3,label=label if s=='shell' else None)
    ax.set(ylim=(-.06,1.06),xlabel='Number of grid halvings',ylabel='Exact square outputs',title='a   Compensation in a ReLU square circuit',xticks=[16,24,35,48,55]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=7,loc='center right');style(ax)
    ax=axes[1]; inp=rows('frontier/trajectory.csv');ts=[4,8,16,32,64,128]
    for meth,key,c,label in [('row','whole_exact',blue,'Isolate readout · exact'),('eigenvector','whole_close',green,'Align input · within tolerance'),('eigenvector','whole_exact',orange,'Align input · exact')]:
        vals=[np.mean([float(r[key]) for r in inp if r['method']==meth and r['d']=='8' and r['t']==str(t) and r['split']=='shell']) for t in ts]
        ax.plot(range(len(ts)),vals,'o-',color=c,ms=3,label=label)
    ax.set(xticks=range(len(ts)),xticklabels=ts,xlabel='Length of test stream',ylabel='Complete streams passing',ylim=(-.06,1.06),title='b   Two equally sparse accumulator circuits')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=6.7,loc='center left');style(ax)
    fig.savefig(OUT/'precision.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':fig_frontier();fig_precision()
