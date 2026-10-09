"""Queue-centred response prototype with local gravity effort; no calibrated arrival-state bound."""
from dataclasses import replace
import torch


def bounded_projection(A, b, lower, upper, iterations):
    if not all(torch.isfinite(x).all() for x in (A, b, lower, upper)):
        raise ValueError('nonfinite projection inputs')
    b, lower, upper = b.to(A), lower.to(A), upper.to(A)
    invalid_box = (lower > upper).any(-1)
    norm = A.square().sum(-1).sqrt()
    missing = ((norm < 1e-9) & (b > 1e-6)).any(-1)
    maximum_in_box = (A * torch.where(A >= 0, upper[:, None, :], lower[:, None, :])).sum(-1)
    impossible_in_box = b > maximum_in_box + 1e-6
    normalized = A / norm.clamp_min(1e-9)[..., None]
    rhs = b / norm.clamp_min(1e-9)
    delta = torch.zeros_like(lower)
    lanes = torch.arange(len(A), device=A.device)
    for _ in range(iterations):
        violation = rhs - torch.einsum('nmd,nd->nm', normalized, delta)
        violation = violation.masked_fill((norm < 1e-7) | impossible_in_box, 0.)
        worst = violation.argmax(-1)
        change = violation[lanes, worst].clamp_min(0)
        delta += normalized[lanes, worst] * change[:, None]
        delta = torch.minimum(torch.maximum(delta, lower), upper)
    residual = b - torch.einsum('nmd,nd->nm', A, delta)
    return delta, dict(residual=residual, missing_direction=missing, invalid_box=invalid_box, impossible_in_box=impossible_in_box)


def make_multirow_issue(env, full_provider, data, profiles, reference, settings):
    arms = ('F_L', 'F_R', 'U_L', 'U_R')
    lanes = torch.arange(64, device=env.device)
    selected = torch.topk(data.dists, settings.get('candidate_rows', 512), largest=False).indices
    one = replace(data, active_idx=selected, active_mask=torch.ones_like(selected, dtype=torch.bool),
        active_pairs=torch.stack([data.dists.gather(1, selected), data.closing.gather(1, selected),
            env._sph.class_id[selected], env._sph.pair_id[selected]], -1),
        active_dmin=data.full_dmin.gather(1, selected), viol_exempt=data.full_viol_exempt.gather(1, selected))
    nearby = full_provider.rows_from(one, env._body_pos_cache)
    controlled_norm = torch.zeros_like(nearby.d)
    for robot, aa in [('F', arms[:2]), ('U', arms[2:])]:
        cols = torch.cat([env._joint_idx[aa[0]], env._joint_idx[aa[1]] + len(env._arms[aa[0]].joint_names)])
        controlled_norm += nearby.J[robot][:, :, cols].square().sum(-1)
    controllable = controlled_norm.sqrt() >= 1e-5
    rank = torch.where(controllable, nearby.d, torch.full_like(nearby.d, 1000.))
    picked = torch.topk(rank, settings['rows'], largest=False).indices
    chosen = selected.gather(1, picked)
    valid = controllable.gather(1, picked)
    one = replace(data, active_idx=chosen, active_mask=valid,
        active_pairs=torch.stack([data.dists.gather(1, chosen), data.closing.gather(1, chosen),
            env._sph.class_id[chosen], env._sph.pair_id[chosen]], -1),
        active_dmin=data.full_dmin.gather(1, chosen), viol_exempt=data.full_viol_exempt.gather(1, chosen))
    rows = full_provider.rows_from(one, env._body_pos_cache)
    selected = chosen
    # Uncontrollable nearby rows remain an admission concern and in raw scoring.
    uncontrolled_negative = ((nearby.d < 0) & ~controllable).any(-1)
    predicted_rate = torch.zeros(64, settings['rows'], device=env.device)
    measured_rate = torch.zeros_like(predicted_rate)
    coefficients, velocity_A, velocity_b, torque_A, torque_b = [], [], [], [], []
    lower, upper = [], []
    cap = torch.full((64,), .02, device=env.device)
    cap[(profiles == 3) | (profiles == 5)] = .02
    offset = 0
    for a in arms:
        art, view, idx = env._arms[a], env._arms[a].root_physx_view, env._joint_idx[a]
        def get(name): return getattr(view, name)().to(env.device).clone()
        q, v = get('get_dof_positions'), get('get_dof_velocities')
        dof, width = q.shape[-1], len(idx)
        robot = a[0]; off = 0 if a.endswith('_L') else len(env._arms[robot + '_L'].joint_names)
        J = rows.J[robot][:, :, off:off + dof].clone()
        ignored = profiles == 7
        hand_cols = torch.ones(dof, dtype=torch.bool, device=env.device); hand_cols[idx] = False
        J[ignored] *= (~hand_cols)[None, None]
        measured_rate += torch.einsum('nmd,nd->nm', J, v)
        kp, kd = get('get_dof_stiffnesses'), get('get_dof_dampings')
        mass = get('get_generalized_mass_matrices')
        di = torch.arange(dof, device=env.device)
        mass[:, di, di] += get('get_dof_armatures')
        bias = get('get_coriolis_and_centrifugal_compensation_forces')
        extra = torch.zeros_like(bias)
        if not art.cfg.spawn.rigid_props.disable_gravity:
            gravity = get('get_gravity_compensation_forces')
            effort = get('get_dof_max_forces')
            extra = torch.minimum(torch.maximum(gravity, -effort), effort)
            bias += gravity - extra
        target = get('get_dof_position_targets'); target[:, idx] = reference[a]
        dt = float(env.cfg.sim.dt)
        C = mass.clone(); C[:, di, di] += dt * kd + dt**2 * kp
        rhs = torch.einsum('nij,nj->ni', mass, v) + dt * kp * (target - q) - dt * bias
        vnext = torch.linalg.solve(C, rhs[:, :, None])[:, :, 0]
        predicted_rate += torch.einsum('nmd,nd->nm', J, vnext)
        basis = torch.eye(dof, device=env.device)[:, idx][None].expand(64, -1, -1)
        v_sensitivity = torch.linalg.solve(C, dt * kp[:, :, None] * basis)
        coefficients.append(torch.einsum('nmd,ndk->nmk', J, v_sensitivity))
        full_vA = torch.zeros(64, width, 26, device=env.device)
        full_vA[:, :, offset:offset+width] = v_sensitivity[:, idx]
        vmax = get('get_dof_max_velocities')[:, idx]
        velocity_A.extend([full_vA, -full_vA]); velocity_b.extend([-vmax-vnext[:, idx], vnext[:, idx]-vmax])
        tau0 = kp * (target-q) - (dt*kp+kd)*vnext + extra
        tau_local = kp[:, :, None]*basis - (dt*kp+kd)[:, :, None]*v_sensitivity
        full_tauA = torch.zeros(64, dof, 26, device=env.device)
        full_tauA[:, :, offset:offset+width] = tau_local
        effort = get('get_dof_max_forces')
        torque_A.extend([full_tauA, -full_tauA]); torque_b.extend([-effort-tau0, tau0-effort])
        limits = art.data.soft_joint_pos_limits[:, idx]
        centre = settings['queued_centre'][a]
        lower.append(torch.maximum(limits[..., 0]-reference[a], centre-cap[:, None]-reference[a]))
        upper.append(torch.minimum(limits[..., 1]-reference[a], centre+cap[:, None]-reference[a]))
        offset += width
    d = rows.d
    delay = torch.full((64, 1), settings['arrival_delay_s'], device=env.device)
    delay[profiles == 6] = 0
    arrival = d + measured_rate.clamp_max(0) * delay
    desired = (-(arrival-settings['reserve_m'])/settings['response_timescale_s']).clamp(
        -settings['inward_velocity_cap_m_s'], settings['outward_velocity_cap_m_s'])
    desired = torch.where(valid, desired, torch.full_like(desired, -.15))
    geometry_A = torch.cat(coefficients, -1)
    A = torch.cat([geometry_A, *velocity_A, *torque_A], 1)
    b = torch.cat([desired-predicted_rate, *velocity_b, *torque_b], 1)
    delta, info = bounded_projection(A, b, torch.cat(lower, -1), torch.cat(upper, -1), settings['projection_passes'])
    geometry_residual = info['residual'][:, :settings['rows']].max(-1).values.clamp_min(0)
    velocity_end = settings['rows'] + 52
    velocity_residual = info['residual'][:, settings['rows']:velocity_end].max(-1).values.clamp_min(0)
    effort_residual = info['residual'][:, velocity_end:].max(-1).values.clamp_min(0)
    output, offset = {}, 0
    for a in arms:
        width = len(env._joint_idx[a]); output[a] = reference[a] + delta[:, offset:offset+width]; offset += width
    return output, dict(selected_rows=selected, controllable_selected_rows=valid.sum(-1),
        uncontrolled_nearby_raw_negative=uncontrolled_negative, geometry_velocity_residual=geometry_residual,
        controlled_velocity_limit_residual=velocity_residual, full_drive_effort_residual=effort_residual,
        missing_direction=info['missing_direction'], bounded_impossible_constraints=info['impossible_in_box'].sum(-1), invalid_box=info['invalid_box'],
        model_constraints_satisfied=(geometry_residual <= 1e-4) & (velocity_residual <= 1e-3) &
            (effort_residual <= 1e-3) & ~info['missing_direction'] & ~info['invalid_box'] & ~uncontrolled_negative,
        predicted_geometry_velocity=predicted_rate, measured_full_geometry_velocity=measured_rate,
        passive_arrival_gap=arrival, delta=delta)
