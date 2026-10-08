"""Evaluation-only reference governor. Does not touch queues or physical state."""
import torch


def reference_bounds(q, target, lower, upper, box, gap=None, *, zero_inclusive=False):
    """Slew toward the nearest reachable point in a tracking envelope.

    A moving q can put the envelope outside this step's reachable target interval.
    In that case return a singleton nearest endpoint, rather than snapping targets,
    exceeding the original speed box, or constructing an empty interval.
    """
    if q.shape != target.shape or lower.shape != q.shape or upper.shape != q.shape:
        raise ValueError('original measured state and target-limit shape required')
    if q.ndim != 2 or not q.is_floating_point():
        raise ValueError('reference state must be a floating batch-by-joint tensor')
    if any(v.dtype != q.dtype or v.device != q.device for v in (target, lower, upper)):
        raise ValueError('reference inputs require identical floating dtype and device')
    if not isinstance(zero_inclusive, bool):
        raise ValueError('zero-inclusive flag must be boolean')
    box_tensor = torch.as_tensor(box)
    if box_tensor.ndim != 0:
        raise ValueError('original reference box must be scalar')
    for value in (q, target, lower, upper, box_tensor):
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
    if zero_inclusive and ((lo > 0).any() or (hi < 0).any()):
        raise ValueError('holding target is outside original reachable limits')
    if gap is None:
        return lo, hi
    gap = torch.as_tensor(gap,device=q.device,dtype=q.dtype)
    if gap.shape not in (torch.Size([]), torch.Size([q.shape[0], 1]), q.shape):
        raise ValueError('reference gap requires scalar, batch-by-one or exact state shape')
    if not torch.isfinite(gap).all() or (gap <= 0).any():
        raise ValueError('reference gap must be positive')
    reference_lo = (q - gap - target).maximum(lo).minimum(hi)
    reference_hi = (q + gap - target).maximum(lo).minimum(hi)
    if zero_inclusive:
        # Only the reference governor is relaxed. The original safety projector
        # may still choose a nonzero correction, inside the same rate/limit box.
        reference_lo = reference_lo.minimum(torch.zeros_like(reference_lo))
        reference_hi = reference_hi.maximum(torch.zeros_like(reference_hi))
    return reference_lo, reference_hi


def install(env, mode, gap_provider=None, *, zero_inclusive=False):
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
                                     env._backstop.cfg.vmax * dt, actual_gap,
                                     zero_inclusive=zero_inclusive)
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
