"""Pure one-step models; no access to future command, targets or measured next q."""
from typing import Protocol
import numpy as np
import torch

NAMES = ('static', 'target_snap', 'velocity', 'empirical')

class MotionPredictor(Protocol):
    def predict(self, q: torch.Tensor, qd: torch.Tensor,
                delivered: torch.Tensor, dt: float) -> torch.Tensor: ...

class Predictor:
    def __init__(self, name, coefficients=None):
        if name not in NAMES: raise ValueError('unknown predictor')
        self.name = name
        self.coefficients = None
        if name == 'empirical':
            value = np.asarray(coefficients, dtype=np.float64)
            if value.shape != (26, 2) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('invalid frozen coefficients')
            self.coefficients = torch.from_numpy(value.copy())

    def predict(self, q, qd, delivered, dt):
        if q.shape != qd.shape or q.shape != delivered.shape or q.shape[-1] != 26:
            raise ValueError('26 matched controlled joints required')
        if not np.isfinite(dt) or dt <= 0: raise ValueError('positive finite dt required')
        if any(not torch.isfinite(v).all() for v in (q, qd, delivered)):
            raise ValueError('nonfinite motion input')
        if self.name == 'static': return torch.zeros_like(q)
        if self.name == 'target_snap': return delivered - q
        if self.name == 'velocity': return qd * dt
        c = self.coefficients.to(device=q.device, dtype=q.dtype)
        out = c[:, 0] * (qd * dt) + c[:, 1] * (delivered - q)
        if not torch.isfinite(out).all(): raise ValueError('nonfinite prediction')
        return out

def nonnegative_fit(gram, xy, yy):
    """Exact two-feature nonnegative least squares via all active sets."""
    gram, xy = np.asarray(gram, float), np.asarray(xy, float)
    if gram.shape != (2,2) or xy.shape != (2,) or not np.isfinite(gram).all() or not np.isfinite(xy).all():
        raise ValueError('invalid sufficient statistics')
    if np.linalg.det(gram) <= 1e-12 * np.prod(np.diag(gram)) or (np.diag(gram) <= 0).any():
        raise ValueError('rank deficient joint')
    candidates = [np.zeros(2), np.array([max(xy[0]/gram[0,0],0),0]),
                  np.array([0,max(xy[1]/gram[1,1],0)])]
    unconstrained = np.linalg.solve(gram, xy)
    if (unconstrained >= 0).all(): candidates.append(unconstrained)
    return min(candidates, key=lambda c: float(yy - 2*c@xy + c@gram@c))
