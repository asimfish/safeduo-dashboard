"""Evaluation-only union of predicted rows and all physically critical rows."""
from dataclasses import replace

import torch


def merge_critical_rows(full, selected, class_id, pair_id, band=.010, capacity=None):
    if band <= 0:
        raise ValueError('critical-row band must be positive')
    if any(x is None for x in (full.dists, full.closing, full.full_dmin, full.full_viol_exempt)):
        raise ValueError('critical-row union requires complete measurements')
    n, p = full.dists.shape
    admission = torch.zeros((n, p), device=full.dists.device, dtype=torch.long)
    admission.scatter_add_(1, selected.active_idx.clamp_min(0), selected.active_mask.long())
    critical = full.dists <= full.full_dmin + band
    union = (admission > 0) | critical
    required = int(union.sum(-1).max().item())
    if capacity is not None and required > capacity:
        raise ValueError(f'critical-row union exceeds declared capacity: {required} > {capacity}')
    width = required if capacity is None else min(capacity, p)
    if width == 0:
        width = 1
    ids = torch.arange(p, device=full.dists.device).expand(n, -1)
    chosen = ids.masked_fill(~union, p).topk(width, dim=1, largest=False).values
    valid = chosen < p
    idx = chosen.clamp_max(p-1)
    features = torch.stack([full.dists.gather(1, idx), full.closing.gather(1, idx),
                            class_id[idx], pair_id[idx]], -1)
    features = features.masked_fill(~valid.unsqueeze(-1), 0)
    features[..., 3] = features[..., 3].masked_fill(~valid, -1)
    return replace(selected, active_pairs=features, active_mask=valid,
                   active_idx=idx.masked_fill(~valid, -1),
                   active_dmin=full.full_dmin.gather(1, idx).masked_fill(~valid, 0),
                   viol_exempt=full.full_viol_exempt.gather(1, idx) & valid)
