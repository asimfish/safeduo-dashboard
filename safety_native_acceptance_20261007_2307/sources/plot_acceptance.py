"""Standalone scientific figures, preserving registered denominators and motion floors."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
OUT=Path('/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/figures')
def main():
    r=json.loads((H/'ACCEPTANCE_RESULT.json').read_text());c=json.loads((H/'COVERAGE_RESULT.json').read_text());rows=r['aggregated'];OUT.mkdir(exist_ok=False)
    names=[v['mode'] for v in rows];x=np.arange(4);fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for ax,key,title in zip(axs,['strict_windows','deep_windows','native_hand_raw_over_windows'],['Strict represented-sphere violations (<0 m)','Deep represented-sphere violations (<-5 mm)','Raw hand normal over registered 0.1 N diagnostic']):
        vals=[v[key] for v in rows];bars=ax.bar(x,vals,color=['#758598','#e09c42','#68a4bb','#7958aa']);ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,128);ax.set_ylabel('Windows / 128 paired cases');ax.set_title(title,fontsize=10)
        for b,v in zip(bars,vals):ax.annotate(str(v),(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
    fig.suptitle(f"Complete 512-window experiment: {r['candidate_decision']} (bounded simulation)",fontsize=13);fig.savefig(OUT/'outcomes.png',dpi=180);fig.savefig(OUT/'outcomes.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');base=next(v for v in rows if v['mode']=='zero_inclusive');cand=next(v for v in rows if v['mode']=='adaptive_joint')
    ratios=[v['mean_q_l2_path']/base['mean_q_l2_path'] if base['mean_q_l2_path'] else np.nan for v in rows];axs[0].bar(x,ratios,color='#4087ab');axs[0].axhline(.9,color='#ac3c45',linestyle='--',label='Registered floor 0.9');axs[0].set_xticks(x,names,rotation=25,ha='right');axs[0].set_ylabel('Measured path / zero baseline');axs[0].legend(fontsize=8)
    sr=[v['path_ratio_candidate_vs_zero'] for v in r['strata']];axs[1].bar(range(7),[np.nan if a is None else a for a in sr],color='#7958aa');axs[1].axhline(.9,color='#ac3c45',linestyle='--');axs[1].set_xticks(range(7),[str(v['label']) for v in r['strata']]);axs[1].set_xlabel('Seven risk strata (-1 = global)');axs[1].set_ylabel('Candidate measured path / zero baseline')
    axs[2].bar(range(26),cand['mean_joint_ranges'],color='#338e83');axs[2].axhline(.001,color='#ac3c45',linestyle='--');axs[2].set_xlabel('Controlled joint (7 + 7 + 6 + 6)');axs[2].set_ylabel('Mean measured range (rad)');fig.suptitle('Actual movement is scored separately from collision outcomes');fig.savefig(OUT/'actual_motion.png',dpi=180);fig.savefig(OUT/'actual_motion.pdf');plt.close(fig)
    hist=sum(np.asarray(b['initial_joint_histogram_12_bins'],int) for b in c['blocks']);fig,ax=plt.subplots(figsize=(12,5),layout='constrained');im=ax.imshow(hist,aspect='auto',origin='lower',cmap='Blues');ax.set_xlabel('Normalized joint-limit bin (12 bins)');ax.set_ylabel('26 requested initial controlled joints');ax.set_title('128 distinct conditioned initial poses; not unconditional IID safety trials');fig.colorbar(im,ax=ax,label='Initial cases');fig.savefig(OUT/'initial_coverage.png',dpi=180);fig.savefig(OUT/'initial_coverage.pdf');plt.close(fig)
    print('SCIENTIFIC_FIGURES_COMPLETE',len(list(OUT.iterdir())),flush=True)
if __name__=='__main__':main()
