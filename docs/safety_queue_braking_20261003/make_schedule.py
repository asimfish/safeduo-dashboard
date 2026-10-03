"""Lock post-hoc trigger times from the new six-step control, before hold probes."""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / 'delay_controls/protocol.json').read_text())
assert protocol['status'] == 'complete'
assert protocol['effective_backstop']['backlog_aware'] is False
baseline = OUT / 'delay_controls/cell_003.npz'
with np.load(baseline) as d:
    meta = json.loads(str(d['meta_json']))
    assert meta['actuator_delay_steps'] == 6
    bad = (d['official_margins'] < 0).any(-1)
    first = [int(np.flatnonzero(bad[:, e])[0]) if bad[:, e].any() else None
             for e in range(bad.shape[1])]
assert any(t is not None for t in first), 'no failed control to diagnose'
recipe = dict(schema='safeduo.posthoc_stop_schedule.v1',
              reference=str(baseline.relative_to(OUT)),
              reference_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
              first_failure_steps=first, lead_steps=12,
              trigger_steps=[max(t - 12, 0) if t is not None else -1 for t in first],
              scope='All failed environments in frozen new control; no intervention in others',
              interpretation='Hindsight diagnostic triggers, not a prospective safety detector')
(OUT / 'stop_schedule.json').write_text(json.dumps(recipe, indent=2) + '\n')
print(json.dumps(recipe))
