"""Closed proposal/qualification coverage; no volume or complete-safety claim."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
R=Path(json.loads((H/'REGISTRATION_HAND_VALID_V2.json').read_text())['raw'])
result=json.loads((H/'PHYSICAL_PREFIX_RESULT_V1.json').read_text())
with np.load(R/'geometric_bank_v2/bank.npz') as b:
    limits=b['soft_limits'];proposal=b['all_proposal_q'];geo=b['accepted_q'];ids=result['selected_input_ids'];selected=geo[ids]
    spans={name:np.ptp(q,axis=0)/(limits[:,1]-limits[:,0]) for name,q in [('proposals',proposal),('geometric512',geo),('selected128',selected)]}
    names=[f'{a}{j+1}' for a,n in [('FL',7),('FR',7),('UL',6),('UR',6)] for j in range(n)]
    eef={a:b[a+'_body_pos_local'][ids,list(b[a+'_body_names']).index('left_hand_base' if a=='F_L' else 'right_hand_base' if a=='F_R' else 'wrist_3_link')] for a in ['F_L','F_R','U_L','U_R']}
    selected_proposals=b['selected_indices'][ids]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'safeduo-prefix512-v1'})
def save(fig,name):
    for ext in ['png','svg','pdf']:
        kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        fig.savefig(H/f'{name}.{ext}',dpi=170,**kwargs)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
for i,(k,label,color) in enumerate([('proposals','18,880 full-limit proposals','#a7b0bc'),('geometric512','512 geometry-qualified poses','#4d89ab'),('selected128','128 prefix-qualified selections','#18846b')]):
    ax.bar(np.arange(26)+(i-1)*.26,spans[k]*100,.26,label=label,color=color)
ax.set_xticks(range(26),names,rotation=55);ax.set_ylim(0,105);ax.set_ylabel('Marginal joint span / native soft-limit range (%)');ax.set_title('Wide joint proposals with explicit conditioning; this is not 26D volume coverage');ax.legend(fontsize=9)
save(fig,'proposal512_joint_coverage')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for (a,p),color in zip(eef.items(),['#155d96','#88b4d4','#c26222','#e5ae79']):
    axes[0].scatter(p[:,0],p[:,1],s=13,label=a,color=color);axes[1].scatter(p[:,0],p[:,2],s=13,label=a,color=color)
for ax in axes:ax.set_xlabel('World-local x (m)');ax.grid(alpha=.2)
axes[0].set_ylabel('World-local y (m)');axes[1].set_ylabel('World-local z (m)');axes[0].legend(fontsize=9)
fig.suptitle('Native hand-base / wrist positions of the prospectively selected 128 inputs')
save(fig,'selected128_eef_points')
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,key,title in zip(axes,['prefix_geometry_qualified','prefix_force_qualified','admitted'],['All12 microstep raw gap >= 0','All82-owner point force <= 0.1 N','Both primary prefix conditions']):
    matrix=np.asarray([c[key] for c in result['cells']]).reshape(4,4);im=ax.imshow(matrix,vmin=0,vmax=32,cmap='YlGnBu')
    for y in range(4):
        for x in range(4):ax.text(x,y,f'{matrix[y,x]}/32',ha='center',va='center',fontsize=11,color='white' if matrix[y,x]>20 else 'black')
    ax.set_xticks(range(4),['.05','.10','.20','.35']);ax.set_yticks(range(4),['0','.10','.25','.50']);ax.set_xlabel('Random target amplitude (rad)');ax.set_title(title,fontsize=9)
axes[0].set_ylabel('Initial velocity / native limit');fig.colorbar(im,ax=axes,shrink=.8,label='Admitted states per cell')
save(fig,'prefix512_cell_admission')
summary=dict(status='CLOSED_COVERAGE_FROM_NATIVE_BANK_AND_512_LEDGER',proposals=18880,geometric512=512,prefix_qualified=270,selected=128,
    marginal_joint_span_fractions={k:v.tolist() for k,v in spans.items()},selected_proposal_indices=selected_proposals.tolist(),
    selected_eef_extent_m={a:np.ptp(p,axis=0).tolist() for a,p in eef.items()},full26D_volume_claim=False,cartesian_path_reachability_claim=False,hand_randomized=False,
    selected128_postprefix_all74_hard_breaches_1e_minus5_rad=sum(r['supplementary_max_hard_limit_violation_rad']>1e-5 for r in result['rejection_ledger'] if r['input_id'] in set(ids)),safety_acceptance=False)
(H/'COVERAGE128_INPUT_RESULT_V1.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(H/'COVERAGE128_PLOT_DATA_V1.npz',**{k+'_joint_span':v for k,v in spans.items()},**{a+'_eef':p for a,p in eef.items()})
print('COVERAGE', {k:(float(v.min()),float(v.max())) for k,v in spans.items()})
