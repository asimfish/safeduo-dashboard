from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent / 'full_motor_operating_domain_v2'


def main():
    source = P / 'JOINT_SAMPLING_SUPPORT_AUDIT_V2.json'
    assert json.loads((P / 'JOINT_SAMPLING_SUPPORT_ACTUAL_EXIT_V2.json').read_text())['actual_outer_exit'] == 0
    d = json.loads(source.read_text())
    assert d['native_trajectories_added'] == d['formal_holdout_count'] == 0
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), layout='constrained')
    fig.suptitle('50-motor sampling: independent speed proposals and remaining coverage gaps', fontsize=17, fontweight='bold')
    for key, label, color in [('previous_v1_uniform_with_shared_row_scales', 'V1: shared row scale (16,000)', '#b65c45'),
                              ('new_v2_pure_independent_velocities', 'V2: independent motor speeds (32,000)', '#24758d')]:
        values = np.array(d[key]['squared_velocity_correlations'])
        ax[0].hist(values, bins=np.linspace(-.1, .6, 71), alpha=.7, color=color, label=label)
        ax[0].axvline(values.mean(), color=color, linestyle='--')
    ax[0].set(xlabel='Pearson correlation of squared velocities', ylabel='Motor pairs', title='All 1,225 motor pairs: magnitude dependence')
    ax[0].legend(fontsize=8)
    keys = ['positions', 'velocities', 'commands']
    values = [d[k]['minimum_pair_bin_occupancy'] for k in keys]
    ax[1].bar(['Position', 'Velocity', 'Target'], values, color='#24758d', width=.55)
    ax[1].set(ylim=(0, 1.2), ylabel='Minimum occupied fraction across motor pairs', title='32,000 pure IID proposals: each pair has 8 x 8 bins')
    for i, key in enumerate(keys):
        ax[1].text(i, values[i]+.025, f"64/64 bins\nMin cell n={d[key]['minimum_pair_cell_count']}", ha='center', fontsize=9)
    distance = np.array(d['unexecuted_development_probe_distances'])
    ax[2].hist(distance, bins=np.linspace(0, 1, 21), color='#24758d', edgecolor='white')
    ax[2].axvline(.25, color='#b65c45', linestyle='--', label='25% of each coordinate range')
    ax[2].set(xlim=(0, 1), xlabel='Nearest of 40,000 proposals, normalized Chebyshev', ylabel='Unexecuted development probes',
              title=f"128 diagnostic probes: median={np.median(distance):.3f}")
    ax[2].legend(fontsize=8, loc='upper left')
    fig.supxlabel('Native proposal trajectories: 0 | Formal acceptance: 0 | Safe or reachable volume: unmeasured\nFull marginal and pairwise support does not establish full 50-dimensional coverage.', fontsize=11)
    outputs = []
    for ext in ['png', 'svg']:
        target = P / f'full50_joint_sampling_diagnostics_v2.{ext}'
        assert not target.exists()
        fig.savefig(target, dpi=170, facecolor='white')
        outputs.append(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    plt.close(fig)
    target = P / 'JOINT_SAMPLING_FIGURE_CUSTODY_V2.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(status='PASS_FIGURE_FROM_CLOSED_DEVELOPMENT_DISTRIBUTION_AUDIT',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), outputs=outputs,
        native_trajectories_added=0, formal_holdout_count=0, fullSystem0_accepted=False), indent=2)+'\n')
    print('FULL50_V2_SCIENTIFIC_FIGURE_CREATED', flush=True)


if __name__ == '__main__':
    main()
