"""Plot outcomes from archived, per-start train losses and held-out errors."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
src=base.parent/'extension/optimizer'
summary=json.loads((src/'summary.json').read_text())
m={r['method']:r for r in summary['methods']}
fig,(a,b)=plt.subplots(1,2,figsize=(7.5,2.75),gridspec_kw={'width_ratios':[1.02,1.]})
fig.subplots_adjust(left=.255,bottom=.20,right=.98,top=.78,wspace=.32)
methods=[('adam800','Adam'),('adam_bfgs','Adam + BFGS'),('ls_xavier','Least squares, Xavier'),('ls_formula','Least squares, WSP formula and examples'),('ls_examples','Least squares, WSP examples only')]
y=np.arange(len(methods)); counts=[m[k]['hinge_certified_near'] for k,_ in methods]
a.barh(y,counts,color=['#75828a','#75828a','#39719a','#a85d35','#a85d35'],height=.57)
a.set_yticks(y,[l for k,l in methods],fontsize=8);a.invert_yaxis();a.set_xlim(0,12.4);a.set_xticks([0,3,6,9,12]);a.set_xlabel('Fits within tolerance (of 12)',fontsize=8)
for i,v in enumerate(counts):a.text(v+.15,i,str(v),va='center',fontsize=8)
a.set_title('A   Fits near the global minimum',loc='left',fontsize=9,fontweight='bold',pad=9)
p=summary['paired']['formula->ls_formula']; y=np.arange(2)
left=np.zeros(2)
for term,color in [('improved','#327d74'),('unchanged','#dfe3e4'),('worsened','#b55d45')]:
    values=np.array([p[k].get(term,90-p[k]['improved']-p[k]['worsened']) for k in ['train_loss','outer_scaled']])
    b.barh(y,values,left=left,color=color,height=.44,label='Within tolerance' if term=='unchanged' else term.title())
    for i,v in enumerate(values):
        if v>0:b.text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=8,color='#222' if term=='unchanged' else 'white')
    left+=values
b.set_yticks(y,['Training','Outer'],fontsize=8);b.invert_yaxis();b.set_xlim(0,90);b.set_xticks([0,30,60,90]);b.set_xlabel('Task / size / start settings (of 90)',fontsize=8)
b.set_title('B   Refining WSP (formula and examples)',loc='left',fontsize=9,fontweight='bold',pad=9)
b.legend(loc='lower center',bbox_to_anchor=(.5,-.53),ncol=3,frameon=False,fontsize=6.4,handlelength=.85,columnspacing=.9)
for ax in [a,b]:
    ax.spines[['right','top','left']].set_visible(False);ax.tick_params(axis='both',length=0,labelsize=7);ax.spines['bottom'].set_color('#9ba7ab')
fig.savefig(base/'optimizer.pdf',bbox_inches='tight',pad_inches=.035)
