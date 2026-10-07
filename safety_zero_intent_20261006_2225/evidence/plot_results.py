"""Standalone scientific plots from complete original-window scoring."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
path=HERE/'holdout_results.json';blob=path.read_bytes();result=json.loads(blob)
assert result['completed_method_windows']==576 and result['invalid_method_windows']==0
rows=result['totals'];labels=['Reference .050','Fixed .010','Zero-inclusive .010'];x=np.arange(3)
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
axes[0,0].bar(x-.17,[r['violations'] for r in rows],width=.34,label='strict <0m')
axes[0,0].bar(x+.17,[r['deep'] for r in rows],width=.34,label='deep <-.005m')
axes[0,0].set_ylabel('Failed original windows /192 per method');axes[0,0].legend()
primary=[rows[0],rows[2]];primary_x=np.arange(2)
axes[0,1].bar(primary_x-.17,[r['strict_env_steps'] for r in primary],width=.34,label='strict env-steps')
axes[0,1].bar(primary_x+.17,[r['deep_env_steps'] for r in primary],width=.34,label='deep env-steps')
axes[0,1].set_ylabel('Duration in correlated environment frames');axes[0,1].set_title('Primary reference/candidate comparison');axes[0,1].legend()
axes[1,0].bar(x,[r['joint_path_l2_mean_rad'] for r in rows]);axes[1,0].set_ylabel('Mean actual joint L2 path (rad)')
axes[1,1].bar(x,[r['four_arms_moving_fraction']*100 for r in rows]);axes[1,1].set_ylabel('All four arms moving >.001rad /frame (%)')
for ax in axes.flat:
    if ax is axes[0,1]:ax.set_xticks(primary_x,[labels[0],labels[2]],rotation=12)
    else:ax.set_xticks(x,labels,rotation=12)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.suptitle('Fresh paired simulation: 192 states x3 methods, actual6-step FIFO\nCandidate rejected: 2 new paired strict failures; no physical-safety certification')
for ext in ['png','pdf','svg']:fig.savefig(HERE/('comparison.'+ext),dpi=180)
assert path.read_bytes()==blob
(HERE/'PLOT_EXECUTION.json').write_text(json.dumps(dict(status='PASS_STANDALONE_ORIGINAL_DATA_PLOTS',result_sha256=hashlib.sha256(blob).hexdigest(),matplotlib=matplotlib.__version__,numpy=np.__version__),indent=2)+'\n')
print('SCIENTIFIC_PLOTS_COMPLETE')
