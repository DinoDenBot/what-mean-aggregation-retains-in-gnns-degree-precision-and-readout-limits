"""Descriptive figures; does not change the frozen analysis or select arms."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
with (ROOT/'results-v1/contrasts.csv').open() as f:rows=list(csv.DictReader(f))
lookup={(r['condition'],r['channel'],r['contrast']):r for r in rows if r['metric']=='log_loss'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
conditions=[('lambda_low','Mean degree 12'),('source','Mean degree 24 (source)'),('lambda_high','Mean degree 48')]
channels=['native','gaussian_025','gaussian_05','gaussian_1']
fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
for ax,(cond,title) in zip(axes,conditions):
    ax.axhline(0,color='#777777',lw=.8)
    for contrast,label,color,marker in [('mean_change','Mean','#23689b','o'),('degree_change','Mean + degree','#be6537','s')]:
        vals=[lookup[(cond,ch,contrast)] for ch in channels]
        y=np.array([float(v['estimate']) for v in vals]);lo=np.array([float(v['lower']) for v in vals]);hi=np.array([float(v['upper']) for v in vals])
        ax.errorbar([0,.25,.5,1],y,yerr=[y-lo,hi-y],marker=marker,color=color,label=label,capsize=3)
    ax.set(title=title,xlabel='Gaussian noise multiplier c',ylabel='Change in log loss (nats/node)')
    ax.set_xticks([0,.25,.5,1]);ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False)
fig.suptitle('Frozen classifiers: first-layer aggregate noise',fontsize=13)
fig.supxlabel('Paired nested-bootstrap intervals: 97.5% for primary mean changes at c=1 and shifted degrees; 95% otherwise. Panels use different y-axis scales.',fontsize=8)
fig.savefig(ROOT/'noise_response.png',dpi=180);fig.savefig(ROOT/'noise_response.pdf');plt.close(fig)

channels=['float16','bfloat16','float16_both','bfloat16_both','gaussian_025','gaussian_05','gaussian_1','uniform_1']
labels=['fp16: layer 1','bf16: layer 1','fp16: both','bf16: both','Gaussian .25','Gaussian .5','Gaussian 1','Uniform 1']
fig,axes=plt.subplots(1,3,figsize=(12,5.1),layout='constrained')
for ax,(cond,title) in zip(axes,conditions):
    vals=[lookup[(cond,ch,'degree_protection')] for ch in channels]
    y=np.array([float(v['estimate']) for v in vals]);lo=np.array([float(v['lower']) for v in vals]);hi=np.array([float(v['upper']) for v in vals])
    ax.axvspan(-.001,.001,color='#eeeeee',label='+/- .001 nats')
    ax.axvline(0,color='#777777',lw=.8)
    ax.errorbar(y,np.arange(len(channels)),xerr=[y-lo,hi-y],fmt='o',color='#23689b',capsize=3)
    ax.set_yticks(np.arange(len(channels)),labels);ax.invert_yaxis()
    ax.set(title=title,xlabel='Degree protection (nats/node)')
    ax.grid(axis='x',alpha=.2)
fig.suptitle('Positive values: the intervention increases mean-only loss more',fontsize=12)
fig.supxlabel('Paired nested-bootstrap intervals: 97.5% for Gaussian 1 at shifted degrees; 95% otherwise. Shading denotes +/- .001 nats. Panels use different x-axis scales.',fontsize=8)
fig.savefig(ROOT/'degree_protection.png',dpi=180);fig.savefig(ROOT/'degree_protection.pdf');plt.close(fig)
print('Saved noise_response and degree_protection as PNG and PDF')
