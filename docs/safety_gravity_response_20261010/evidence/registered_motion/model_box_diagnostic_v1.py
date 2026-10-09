"""CPU-only nominal one-step box audit; not a calibration or safety certificate."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
REG = json.loads((P / 'REGISTRATION_V1.json').read_text())
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bank = load(REG['bank'])
    goals = load(REG['common_goal_tape'])
    rows = []
    for batch in range(2):
        leaf = Path(json.loads((H / 'gravity_hold128_v1/REGISTRATION_V1.json').read_text())['raw']) / f'paired_batch{batch}_on_v8'
        params = load(leaf / 'resolved_native_parameters.npz')
        dyn = load(leaf / 'dynamics_stream.npz')
        selected = np.asarray(REG['batch_bank_positions'][batch])
        initial = goals['initial_controlled_q'][selected].astype(np.float64)
        mode_results = {}
        for name, reference, centred, compensate in [
                ('old_far_reference_box_no_support', bank['reference_target'][0, selected], False, False),
                ('bounded_goal_queue_box_with_support', goals['reference_target'][0, selected], True, True)]:
            matrices, rhs, lower, upper = [], [], [], []
            offset = 0
            for arm in ARMS:
                def p(field): return params[arm + '_' + field].astype(np.float64)
                def d(field): return dyn[arm + '_' + field][0].astype(np.float64)
                q, v, mass = d('q'), d('qd'), d('mass_matrix')
                kp, kd, armature = p('stiffness'), p('damping'), p('armature')
                idx = params[arm + '_controlled_joint_indices']
                width, dof = len(idx), q.shape[-1]
                diag = np.arange(dof)
                mass[:, diag, diag] += armature
                bias = d('coriolis')
                extra = np.zeros_like(bias)
                if not bool(params[arm + '_disable_gravity_cfg'][0]):
                    gravity = d('gravity')
                    if compensate:
                        extra = np.clip(gravity, -p('max_force'), p('max_force'))
                    bias += gravity - extra
                target = params[arm + '_full_hold_target'].astype(np.float64).copy()
                target[:, idx] = reference[:, offset:offset + width]
                dt = .008333
                c = mass.copy()
                c[:, diag, diag] += dt * kd + dt * dt * kp
                velocity = np.linalg.solve(c, (np.einsum('nij,nj->ni', mass, v) + dt * kp * (target - q) - dt * bias)[..., None])[..., 0]
                basis = np.broadcast_to(np.eye(dof)[:, idx], (64, dof, width))
                sensitivity = np.linalg.solve(c, dt * kp[..., None] * basis)
                velocity_a = np.zeros((64, width, 26))
                velocity_a[:, :, offset:offset + width] = sensitivity[:, idx]
                vmax = p('max_velocity')[:, idx]
                matrices += [velocity_a, -velocity_a]
                rhs += [-vmax - velocity[:, idx], velocity[:, idx] - vmax]
                tau = kp * (target - q) - (dt * kp + kd) * velocity + extra
                tau_local = kp[..., None] * basis - (dt * kp + kd)[..., None] * sensitivity
                tau_a = np.zeros((64, dof, 26))
                tau_a[:, :, offset:offset + width] = tau_local
                effort = p('max_force')
                matrices += [tau_a, -tau_a]
                rhs += [-effort - tau, tau - effort]
                soft = p('soft_limits')
                centre = initial[:, offset:offset + width] if centred else reference[:, offset:offset + width]
                cap = .02 if centred else .05
                lower.append(np.maximum(soft[..., 0] - reference[:, offset:offset + width], centre - cap - reference[:, offset:offset + width]))
                upper.append(np.minimum(soft[..., 1] - reference[:, offset:offset + width], centre + cap - reference[:, offset:offset + width]))
                offset += width
            a, b, lo, hi = [np.concatenate(x, axis=-1 if x is lower or x is upper else 1) for x in [matrices, rhs, lower, upper]]
            maximum = (a * np.where(a >= 0, hi[:, None], lo[:, None])).sum(-1)
            impossible = (b > maximum + 1e-6).any(-1)
            invalid = (lo > hi).any(-1)
            mode_results[name] = dict(individual_constraint_impossible_lanes=int(impossible.sum()),
                invalid_boxes=int(invalid.sum()), impossible_input_ids=params['global_input_id'][impossible].tolist(),
                geometry_constraints_included=False, global_joint_feasibility_proven=False,
                force_is_nominal_total_estimate_not_native_PD_readback=True)
        rows.append(dict(batch=batch, modes=mode_results, native_source_sha256=sha(leaf / 'dynamics_stream.npz')))
    result = dict(status='CLOSED_CPU_MODEL_BOX_DIAGNOSTIC', inputs=128, rows=rows,
        previously_registered_action_probe_unchanged=True, actual_native_action_outcomes_used=False,
        model_calibrated=False, safety_acceptance=False, fullSystem0_accepted=False,
        scope='Individual nominal linear constraint/box impossibility only. Absence does not prove joint geometry/dynamics feasibility.',
        source_sha256=sha(Path(__file__)))
    (P / 'MODEL_BOX_DIAGNOSTIC_V1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
