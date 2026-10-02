"""Export auditable research figures from the same data shown in the panel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text())
plt.rcParams.update({'font.size':10, 'svg.fonttype':'none'})
fig, ax=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
labels=[['Legacy link prediction','COM prediction','Body velocity'],['Legacy lag guard','Combined correction']]
for i,x in enumerate(r['traces'][:2]):
    for s,label in zip(x['series'],labels[i]):
        ax[i].plot(x['time_s'],s['values'],color=s['color'],label=label,lw=1.7)
    ax[i].axhline(0,color='.3',ls='--',lw=.8)
    ax[i].set(xlim=(x['x_min'],x['x_max']),ylim=(x['y_min'],x['y_max']),
              xlabel='Simulation time (s)',ylabel='Margin rate (mm/s)' if i==0 else 'Measured self_U margin (mm)')
    ax[i].grid(alpha=.2)
    ax[i].legend(fontsize=8)
ax[0].set_title('Same legacy state: row7276 / env2')
ax[1].set_title('Matched seed / initial pose: env2')
fig.savefig(OUT/'kinematics_and_margin.png',dpi=180)
fig.savefig(OUT/'kinematics_and_margin.svg')
plt.close(fig)
x=r['traces'][2]
fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
for s,label in zip(x['series'],['Raw','System 0']):
    ax.plot(x['time_s'],s['values'],color=s['color'],label=label,lw=1.8)
ax.axhline(0,color='.3',ls='--',lw=.8)
ax.set(xlim=(x['x_min'],x['x_max']),ylim=(x['y_min'],x['y_max']),xlabel='Simulation time (s)',
       ylabel='Measured F_L–F_R sphere surface distance (mm)',title='Identical command tape and initial pose: held random seed9')
ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'random_pair_safety.png',dpi=180)
fig.savefig(OUT/'random_pair_safety.svg')
plt.close(fig)
rows=[x for x in r['rows'] if x['hold_steps']<=30]
fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
names={'raw':'Raw','backstop_only':'Backstop','system0':'System 0'}
for i,flow in enumerate(['uniform_random','held_random']):
    cells=[next(x for x in rows if x['flow']==flow and x['method']==m) for m in names]
    ax.bar(np.arange(3)+(i-.5)*.35,[100*x['mean_joint_range_fraction'] for x in cells],
           width=.35,label='IID (each step)' if i==0 else 'Random hold (0.5 s)')
ax.set_xticks(np.arange(3),[names[x] for x in names])
ax.set(ylabel='Mean measured joint span / soft-limit span (%)',title='New seeds 9/10, 32 environments, 5 s, amplitude 0.015 rad')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'random_coverage.png',dpi=180)
fig.savefig(OUT/'random_coverage.svg')
plt.close(fig)
