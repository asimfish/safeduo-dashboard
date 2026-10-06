"""Evaluation-only reference governor. Does not touch queues or physical state."""
import torch


def reference_bounds(q, target, lower, upper, box, gap=None):
    """Slew toward the nearest reachable point in a tracking envelope.

    A moving q can put the envelope outside this step's reachable target interval.
    In that case return a singleton nearest endpoint, rather than snapping targets,
    exceeding the original speed box, or constructing an empty interval.
    """
    if q.shape != target.shape or lower.shape != q.shape or upper.shape != q.shape:
        raise ValueError('original measured state and target-limit shape required')
    for value in (q, target, lower, upper, torch.as_tensor(box)):
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite original reference input')
    if box <= 0 or (lower > upper).any():
        raise ValueError('positive original box and ordered limits required')
    lo = torch.maximum(lower - target, torch.full_like(target, -box))
    hi = torch.minimum(upper - target, torch.full_like(target, box))
    if not torch.isfinite(q).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite reference state')
    if (lo > hi).any():
        raise ValueError('original target/limit interval is unreachable')
    if gap is None:
        return lo, hi
    gap = torch.as_tensor(gap,device=q.device,dtype=q.dtype)
    if not torch.isfinite(gap).all() or (gap <= 0).any():
        raise ValueError('reference gap must be positive')
    return ((q - gap - target).maximum(lo).minimum(hi),
            (q + gap - target).maximum(lo).minimum(hi))


def install(env, mode, gap_provider=None):
    from safeduo.safety.types import ARM_KEYS, DeltaCmd
    if mode not in ('box_only', 'envelope_050'):
        raise ValueError('unregistered mechanism')
    original = env._backstop.project
    gap = None if mode == 'box_only' else .050

    def project(cmd, rows, alpha, p, dt, **kwargs):
        state = env.scene_state()
        actual_gap = gap if gap_provider is None else gap_provider()
        bounds = {a: reference_bounds(state.q[a], env._targets[a],
                                     env._q_soft_limits[a][..., 0],
                                     env._q_soft_limits[a][..., 1],
                                     env._backstop.cfg.vmax * dt, actual_gap)
                  for a in ARM_KEYS}
        # The original R19 bypass returns cmd verbatim. Its input must therefore
        # satisfy the same envelope as the projection, without disabling R19.
        limited = {a: cmd.delta_q[a].maximum(bounds[a][0]).minimum(bounds[a][1])
                   for a in ARM_KEYS}
        kwargs['delta_bounds'] = bounds
        result, active, info = original(DeltaCmd(delta_q=limited), rows, alpha, p, dt, **kwargs)
        changed = torch.stack([(limited[a] - cmd.delta_q[a]).abs().amax(-1) > 1e-6
                               for a in ARM_KEYS], -1)
        for a in ARM_KEYS:
            if ((result.delta_q[a] < bounds[a][0] - 1e-6) |
                (result.delta_q[a] > bounds[a][1] + 1e-6)).any():
                raise ValueError('governor output exceeds declared reachable reference bounds')
        info['reference_governor_changed'] = changed
        return result, active | changed, info

    env._backstop.project = project
    return original
