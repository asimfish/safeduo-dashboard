"""Standalone measured comparison plots, without IID error bars."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    r=json.loads((HERE/'holdout_results.json').read_text());d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())
    modes=d['modes'];labels=['Reference','Velocity','Empirical PD','Velocity + PD'];colors=['#3573a6','#26795d','#b97724','#7651a3']
    fig,ax=plt.subplots(1,2,figsize=(12,4.9),constrained_layout=True)
    x=np.arange(4);width=.18
    for j,m in enumerate(modes):
        rows=[next(row for row in r['rows'] if row['mode']==m and row['seed']==block['command_seed']) for block in d['rows']]
        strict=[row['violations']/64*100 for row in rows]+[sum(row['violations'] for row in rows)/192*100]
        deep=[row['deep']/64*100 for row in rows]+[sum(row['deep'] for row in rows)/192*100]
        xx=x+(j-1.5)*width
        ax[0].bar(xx,strict,width,color=colors[j],label=labels[j])
        ax[0].bar(xx,deep,width,color='none',edgecolor='#162433',hatch='///',linewidth=.7)
    ax[0].set_xticks(x,['Block1 (64)','Block2 (64)','Block3 (64)','All (192)'])
    ax[0].set_ylabel('Windows with measured violation (%)');ax[0].set_title('Strict geometry < 0; hatch: deeper than 5 mm')
    ax[0].legend(fontsize=8,loc='upper left',ncols=2);ax[0].grid(axis='y',alpha=.18);ax[0].set_axisbelow(True)
    coverage={c['mode']:c['measured'] for c in r['coverage']};base=coverage['joint_reference']
    paths=[coverage[m]['mean_joint_path_rad']/base['mean_joint_path_rad'] for m in modes]
    ranges=[coverage[m]['mean_within_window_joint_range']/base['mean_within_window_joint_range'] for m in modes]
    ax[1].bar(x-.18,paths,.36,color='#3573a6',label='Actual joint path')
    ax[1].bar(x+.18,ranges,.36,color='#b97724',label='Within-window normalized range')
    ax[1].axhline(1,color='#344b5c',linestyle='--',linewidth=1)
    ax[1].set_xticks(x,labels,rotation=12);ax[1].set_ylabel('Measured ratio to reference');ax[1].set_title('Movement measured on each actual trajectory')
    ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.18);ax[1].set_axisbelow(True)
    fig.suptitle('SafeDuo: 192 fresh paired cases x 4 methods, strict six-step FIFO',fontsize=12)
    for suffix in ('png','pdf','svg'):fig.savefig(HERE/('comparison.'+suffix),dpi=180)
    print('PLOT_FROM_CLOSED_RAW_RESULTS',flush=True)
