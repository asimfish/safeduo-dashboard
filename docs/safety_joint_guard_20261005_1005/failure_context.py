"""Post-registered descriptive contexts for every regression against baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_v2 import HERE, load, sha

RAW = Path('/mnt/nas/data/lyf/double_hand') / HERE.name


def frame(path, step, env):
    receipts = json.loads((path / 'forecast_receipts.json').read_text())
    chunk = next(c for c in receipts['chunks'] if c['start'] <= step < c['stop'])
    file = path / chunk['path']; assert sha(file) == chunk['sha256']
    with np.load(file, allow_pickle=False) as z:
        i = step - chunk['start']
        out = {k: z[k][i, env].copy() for k in ['forecast', 'measured_d', 'dmin', 'selected_ids']}
        out['exempt'] = np.unpackbits(z['exempt'][i, env])[:9021].astype(bool)
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    assert not args.out.exists(), 'preserve the previous descriptive receipt'
    campaign = json.loads((RAW / 'holdout/campaign.json').read_text())
    jobs = {j['id']: j for j in campaign['jobs'] if j['status'] == 'complete'}
    design = json.loads((HERE / 'DESIGN.json').read_text())
    rows = []; inputs = {}
    for registration in design['rows']:
        seed = registration['command_seed']
        if args.seed is not None and args.seed != seed: continue
        base_id = f'baseline_guard_{seed}'
        if base_id not in jobs: continue
        base = load(RAW / 'holdout' / base_id / 'cell_001.npz')
        base_bad = (base['official_margins'] < 0).any((0, 2))
        for mode in ['admission_guard', 'envelope_guard', 'joint_guard']:
            job_id = f'{mode}_{seed}'
            if job_id not in jobs: continue
            path = RAW / 'holdout' / job_id
            protocol = json.loads((path / 'protocol.json').read_text())
            assert protocol['status'] == 'complete' and protocol['steps'] == 960
            data = load(path / 'cell_001.npz'); diag = load(path / 'project_diagnostics.npz')
            assert np.array_equal(data['q_initial'], base['q_initial']) and np.array_equal(data['cmd'], base['cmd'])
            bad = (data['official_margins'] < 0).any((0, 2))
            inputs[job_id] = {f: sha(path / f) for f in ['cell_001.npz', 'project_diagnostics.npz', 'full_row_identity.json', 'forecast_receipts.json']}
            identity = json.loads((path / 'full_row_identity.json').read_text())
            for env in np.flatnonzero(~base_bad & bad):
                e = int(env); t = int(np.flatnonzero((data['official_margins'][:, e] < 0).any(-1))[0])
                issued = data['q_initial'][e] if t == 0 else data['controller_target'][t-1, e]
                q_pre = data['q_initial'][e] if t == 0 else data['q'][t-1, e]
                pending = data['pre_pending_actuator_targets'][t, :, e]
                before = frame(path, t, e)
                record = dict(seed=seed, env=e, mode=mode, first_failure_step=t,
                    negative_classes=np.flatnonzero(data['official_margins'][t, e] < 0).tolist(),
                    first_post_margin_m=data['official_margins'][t, e].tolist(),
                    window_min_margin_m=data['official_margins'][:, e].min(0).tolist(),
                    maximum_pre_target_debt_rad=float(np.abs(issued-q_pre).max()),
                    maximum_pending_target_displacement_rad=float(np.abs(pending-q_pre).max()),
                    q_pre=q_pre.tolist(), pre_qd=data['pre_qd_compact'][t,e].tolist(),
                    issued_target=issued.tolist(), pending_targets=pending.tolist(),
                    raw_command=data['cmd'][t,e].tolist(), returned_command=data['exec'][t,e].tolist(),
                    actual_target_increment=data['effective_target_delta'][t,e].tolist(),
                    diagnostics={k:np.asarray(v[t,e]).tolist() for k,v in diag.items()
                        if k.startswith(('original_safety_residual_','returned_safety_residual_','target_safety_residual_',
                                         'returned_alpha_residual_','target_alpha_residual_',
                                         'returned_bound_residual_','target_bound_residual_',
                                         'individual_infeasibility_lower_bound_'))},
                    context_scope='observed temporal context, not a unique causal attribution')
                if t < 959:
                    after = frame(path, t+1, e)
                    nonexempt = np.flatnonzero(~after['exempt'])
                    worst = nonexempt[np.argsort(after['measured_d'][nonexempt])[:3]]
                    selected = set(before['selected_ids'][before['selected_ids'] >= 0].tolist())
                    record['nearest_nonexempt_post_rows'] = [dict(row_id=int(i),
                        pair_id=identity['pair_id'][i], physical_class=identity['class_id'][i],
                        sphere_indices=identity['pair_sphere_idx'][i],
                        post_geometric_margin_m=float(after['measured_d'][i]),
                        previous_geometric_margin_m=float(before['measured_d'][i]),
                        previous_raw_linear_forecast_m=float(before['forecast'][i]),
                        previous_braking_dmin_m=float(before['dmin'][i]),
                        selected_previous=bool(i in selected),exempt_previous=bool(before['exempt'][i])) for i in worst]
                else:
                    record['post_full_rows_unavailable'] = 'terminal dense margin exists; no next full pre frame'
                rows.append(record)
    result = dict(schema='safeduo.posthoc_regression_context.v1', rows=rows, input_sha256=inputs,
        analysis_sha256=sha(__file__), endpoint='strict original geometric margin<0',
        study_status=campaign['status'], selector='all completed-cell regressions versus same-seed baseline; no outcome exclusions',
        interpretation='descriptive supplement after outcomes; no candidate/tape/sampling/threshold changes',
        hardware_approved=False)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(contexts=len(rows),out=str(args.out))))


if __name__ == '__main__': main()
