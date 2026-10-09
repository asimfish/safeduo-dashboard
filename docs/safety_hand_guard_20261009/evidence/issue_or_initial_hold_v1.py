"""Experimental disposition of an invalid nominal proposal; not a safe backup proof."""
import torch


def issue_or_initial_hold(proposal, initial_hold, model_satisfied):
    assert model_satisfied.dtype == torch.bool and model_satisfied.ndim == 1
    assert proposal.keys() == initial_hold.keys()
    output = {}
    for arm in proposal:
        candidate, hold = proposal[arm], initial_hold[arm]
        assert candidate.shape == hold.shape and candidate.shape[0] == len(model_satisfied)
        assert torch.isfinite(candidate).all() and torch.isfinite(hold).all()
        output[arm] = torch.where(model_satisfied[:, None], candidate, hold)
    return output, ~model_satisfied
