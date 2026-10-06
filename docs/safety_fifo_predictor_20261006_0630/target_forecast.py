"""All-row checks, including intermediates that a minimum could hide."""
import torch

from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS


def require_finite(label, tensors):
    for key, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite {label}.{key}; abort before projection/physics')


def full_forecast(full, rows, state, issued, pending, cmd, limits, box):
    require_finite('control_box', dict(box=torch.as_tensor(box)))
    if box <= 0 or len(pending) != 6:
        raise ValueError('positive speed box and exact six pending targets required')
    require_finite('geometry', dict(d=full.dists, dmin=full.full_dmin, closing=full.closing))
    require_finite('jacobian', rows.J)
    require_finite('q', state.q)
    require_finite('qd', state.qd)
    require_finite('issued', issued)
    require_finite('command', cmd.delta_q)
    require_finite('limits', limits)
    proposal = {a: (issued[a] + cmd.delta_q[a].clamp(-box, box)).clamp(
        limits[a][..., 0], limits[a][..., 1]) for a in ARM_KEYS}
    require_finite('boxed_proposal', proposal)
    forecast = full.dists.clone()
    for index, target in enumerate(list(pending) + [issued, proposal]):
        require_finite(f'target_{index}', target)
        displacement = {a: target[a] - state.q[a] for a in ARM_KEYS}
        require_finite(f'displacement_{index}', displacement)
        delta = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(displacement, r))
                    for r in ('F', 'U'))
        require_finite(f'linear_delta_{index}', dict(delta=delta))
        predicted = full.dists + delta
        require_finite(f'predicted_{index}', dict(distance=predicted))
        forecast = torch.minimum(forecast, predicted)
    require_finite('forecast', dict(all_rows=forecast))
    return forecast
