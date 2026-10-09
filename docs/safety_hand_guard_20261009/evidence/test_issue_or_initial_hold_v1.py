"""Regression oracle for atomic per-lane proposal/hold disposition."""
from collections import deque
import json
from pathlib import Path

import torch

from issue_or_initial_hold_v1 import issue_or_initial_hold


def main():
    arms = ('F_L', 'F_R', 'U_L', 'U_R')
    proposal = {a: torch.arange(3 * n, dtype=torch.float32).reshape(3, n) + 10 for a, n in zip(arms, (7, 7, 6, 6))}
    hold = {a: torch.full_like(proposal[a], -2.) for a in arms}
    frozen = {a: proposal[a].clone() for a in arms}
    flags = torch.tensor([False, True, False])
    out, blocked = issue_or_initial_hold(proposal, hold, flags)
    assert torch.equal(blocked, ~flags)
    for a in arms:
        assert torch.equal(out[a][[0, 2]], hold[a][[0, 2]])
        assert torch.equal(out[a][1], proposal[a][1])
        assert torch.equal(proposal[a], frozen[a])
    queue = deque({a: hold[a].clone() for a in arms} for _ in range(6))
    for step in range(12):
        issued, _ = issue_or_initial_hold(proposal, hold, flags)
        applied = queue.popleft()
        queue.append(issued)
        for a in arms:
            assert torch.equal(applied[a], hold[a] if step < 6 else out[a])
    bad = {a: proposal[a].clone() for a in arms}
    bad['U_L'][0, 0] = float('nan')
    try:
        issue_or_initial_hold(bad, hold, flags)
    except AssertionError:
        pass
    else:
        raise AssertionError('A nonfinite proposal must invalidate the experiment')
    result = dict(status='PASS_ATOMIC_FOUR_ARM_LANE_DISPOSITION_FIFO6_AND_NONFINITE_ORACLE',
        analytic_checks=3, backup_safety_proven=False, no_production_policy_changed=True)
    path = Path(__file__).resolve().parent / 'HOLD_FALLBACK_ANALYTIC_ORACLE_V1.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
