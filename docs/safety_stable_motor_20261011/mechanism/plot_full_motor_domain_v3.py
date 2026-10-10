"""Show marginal proposal support, explicitly excluding native coverage claims."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
O=H/'full_motor_operating_domain_v1'


def main():
    report=json.loads((O/'FULL_MOTOR_DOMAIN_V1.json').read_text())
    gate=json.loads((O/'INDEPENDENT_PROPOSAL_ORACLE_ACTUAL_EXIT_V1.json').read_text())
    assert gate['actual_outer_exit']==0
    with np.load(O/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz',allow_pickle=False) as z:
        low,high=z['independent_lower_rad'],z['independent_upper_rad']
        normalized=(z['independent_q'][:16000]-low)/(high-low)
        names=z['independent_names'].tolist()
        counts=np.stack([np.histogram(normalized[:,i],bins=np.linspace(0,1,11))[0] for i in range(50)])
    fig,(a,b)=plt.subplots(2,1,figsize=(14,12),gridspec_kw={'height_ratios':[4,1]},constrained_layout=True)
    im=a.imshow(counts,aspect='auto',origin='lower',extent=[0,1,-.5,49.5],cmap='Blues')
    labels=[n.replace('left_','').replace('_joint','') for n in names]
    a.set_yticks(np.arange(50));a.set_yticklabels(labels,fontsize=7)
    a.set_xlabel('Normalized feasible independent coordinate interval; 10 marginal bins')
    a.set_ylabel('All 26 arm and 24 independent hand motor coordinates')
    fig.colorbar(im,ax=a,label='Uniform development proposal count')
    span=np.asarray(report['uniform_marginal_span_fraction'])
    hand_names={row['arm']+'/'+motor['name'] for row in report['rows'] for motor in row['motor_domain']}
    colors=['#e28743' if name in hand_names else '#3b70a3' for name in names]
    assert colors.count('#e28743')==24 and colors.count('#3b70a3')==26
    b.bar(np.arange(50),span*100,color=colors)
    b.set_ylim(0,100);b.set_xlabel('Independent coordinate index; blue arms, orange hand motors')
    b.set_ylabel('Uniform marginal span (%)');b.grid(axis='y',alpha=.3)
    fig.suptitle('Full50 motor domain: 16,000 uniform + 4,000 boundary development proposals\nOriginal74 limit and24 relation software checks PASS; native coverage NOT TESTED\nMarginal proposal support only; high-dimensional reachable or safe volume not measured; formal holdout0',fontsize=14)
    outputs=[]
    for ext in ['png','svg']:
        target=O/('full50_proposal_support_v3.'+ext);assert not target.exists();fig.savefig(target,dpi=160)
        outputs.append(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    plt.close(fig)
    (O/'FIGURE_CUSTODY_V3.json').write_text(json.dumps(dict(outputs=outputs,source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_trials=0,fullSystem0_accepted=False),indent=2)+'\n')
    print('PASS_FULL50_PROPOSAL_ONLY_MARGINAL_FIGURE',flush=True)


if __name__=='__main__':
    main()
