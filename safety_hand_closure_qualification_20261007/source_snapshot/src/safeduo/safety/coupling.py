"""A31 task-state coupling and R29-compatible policy row filtering.

Object IDs must come from a task-state provider. Proximity plus hand closure
is an operational co-holding estimate, not a contact-force certificate.
No task state means no coupling; a family name alone never enables it.
"""
from __future__ import annotations

import torch

ARM_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def coupling_from_grasps(object_ids, closed, near):
    if object_ids.ndim != 2 or object_ids.shape[-1] != 4:
        raise ValueError("object_ids must have shape (N, 4); -1 means empty")
    if closed.shape != object_ids.shape or near.shape != object_ids.shape:
        raise ValueError("closed/near must match object_ids")
    holding = (object_ids >= 0) & closed.bool() & near.bool()
    same = object_ids.unsqueeze(-1) == object_ids.unsqueeze(-2)
    return (same & holding.unsqueeze(-1) & holding.unsqueeze(-2)
            & ~torch.eye(4, dtype=torch.bool, device=object_ids.device))


def group_min_alpha(alpha, graph):
    if graph.shape != (*alpha.shape, 4) or alpha.shape[-1] != 4:
        raise ValueError("alpha (N, 4) and graph (N, 4, 4) required")
    if graph.dtype != torch.bool:
        raise ValueError("graph must be boolean")
    # Symmetrize and take transitive closure: overlapping groups must never
    # depend on their iteration order. Four arms require at most 3 hops.
    reach = graph | graph.transpose(-1, -2)
    reach = reach | torch.eye(4, dtype=torch.bool, device=graph.device)
    for k in range(4):
        reach = reach | (reach[..., :, k:k+1] & reach[..., k:k+1, :])
    return torch.where(reach, alpha.unsqueeze(-2), torch.inf).amin(-1)


def coupling_features(graph):
    return torch.stack([graph[..., i, j] for i, j in ARM_PAIRS], -1).float()


def synchronization_loss(alpha, features):
    differences = torch.stack([(alpha[..., i] - alpha[..., j]).abs()
                               for i, j in ARM_PAIRS], -1)
    return (differences * features).sum() / features.sum().clamp_min(1)


def structural_observation_mask(pairs, valid, dmin, class_dmin,
                                *, lookahead_s=0.0, struct_engage_dist=None):
    cls = pairs[..., 2].long().clamp(0, 2)
    structural = valid & ((dmin - class_dmin[cls]).abs() > 1e-9)
    # Active-pair closing velocity is positive on approach (opposite J qd).
    predicted = pairs[..., 0] - lookahead_s * pairs[..., 1].clamp_min(0)
    drop = structural & (predicted >= dmin)
    if struct_engage_dist is not None:
        drop = drop & (pairs[..., 0] >= struct_engage_dist)
    return valid & ~drop
