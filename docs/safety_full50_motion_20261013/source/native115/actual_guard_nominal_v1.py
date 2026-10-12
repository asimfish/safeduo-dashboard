"""Guard target reference anchored to the previous submitted command.

This changes command construction only. Native q/qd, original URDF coupling,
the existing rate filter, and all physical acceptance gates remain separate.
"""
import numpy as np
from coupled_hand_targets_v1 import CoupledHand, Relation


def actual_guard_nominal(names, nominal, previous, opened, far, hard, relations,
                         mode, thumb_guard, finger_guard, control_dt_s,
                         relax_rate):
    names = tuple(names)
    vectors = [np.asarray(x, dtype=float) for x in (nominal, previous, opened, far)]
    nominal, previous, opened, far = vectors
    if not all(x.shape == (len(names),) and np.isfinite(x).all() for x in vectors):
        raise ValueError('Invalid command reference')
    if mode not in ('SEEK', 'HOLD', 'RELAX', 'JOINT_GUARD'):
        raise ValueError('Unknown feedback mode')
    if not np.isfinite([control_dt_s, relax_rate]).all() or min(control_dt_s, relax_rate) <= 0:
        raise ValueError('Invalid feedback clock or opening rate')
    hand = CoupledHand(names, hard, [Relation(r['slave'], r['master'], r['urdf_multiplier'], r['urdf_offset_rad']) for r in relations])
    roots = np.array([names.index(n) for n in hand.motors], dtype=int)
    for x in vectors:
        if np.max(abs(hand.expand(x[roots]) - x)) > 1e-6:
            raise ValueError('Command does not follow original coupling')
    if not ((previous >= hand.limits[:, 0] - 1e-6).all() and (previous <= hand.limits[:, 1] + 1e-6).all()):
        raise ValueError('Previous submitted target outside original limits')
    thumb = np.array(['thumb' in n for n in names])
    for r in hand.relations:
        if thumb[names.index(r.slave)] != thumb[names.index(r.master)]:
            raise ValueError('Coupling crosses independently guarded groups')
    if mode == 'SEEK':
        return nominal.copy()
    if mode == 'HOLD':
        return previous.copy()
    if mode == 'JOINT_GUARD' and not (thumb_guard or finger_guard):
        raise ValueError('Joint guard has no guarded group')
    active = np.ones(len(roots), dtype=bool) if mode == 'RELAX' else np.where(thumb[roots], thumb_guard, finger_guard)
    delta = relax_rate * control_dt_s * (far[roots] - opened[roots])
    desired = previous[roots] - np.where(active, delta, 0.)
    # Opening stops at the registered open endpoint; it cannot pass through it.
    lower = np.minimum(previous[roots], opened[roots])
    upper = np.maximum(previous[roots], opened[roots])
    desired = np.where(active, np.clip(desired, lower, upper), previous[roots])
    return hand.expand(desired)
