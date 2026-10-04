import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/zero_cells_v2')
manifest=json.loads((ROOT/'campaign.json').read_text());assert manifest['status']=='complete'
rows=[]
for job in manifest['jobs']:
    path=ROOT/job['id'];d=np.load(path/'cell_001.npz');z=np.load(path/'zero_time_geometry.npz')
    assert d['q'].shape==(60,64,26) and not d['cmd'].any() and not d['exec'].any()
    target=np.repeat(d['q_initial'][None],60,axis=0)
    assert np.array_equal(d['controller_target'],target) and np.array_equal(d['actuator_target'],target)
    assert not d['initial_violation'].any()
    bad=(d['official_margins']<0).any((0,2));early=(d['official_margins'][:6]<0).any((0,2))
    hidden=(z['before_table_margin']<0)&z['before_table_exempt']
    withdrawal=(hidden[None]&(d['table_margin_all']<0)&~d['table_exempt_mask']).any((0,2))
    rows.append(dict(id=job['id'],windows=64,violations=int(bad.sum()),early_violations=int(early.sum()),
                     failed_envs=np.flatnonzero(bad).tolist(),initial_exempt_table_overlap=int(hidden.any(-1).sum()),
                     negative_table_exemption_withdrawal=int(withdrawal.sum()),
                     class_violations=(d['official_margins']<0).any(0).sum(0).tolist(),
                     zero_time_q_exact=bool(np.array_equal(z['before_q'],z['after_q'])),
                     zero_time_centers_exact=bool(np.array_equal(z['before_centers'],z['after_centers'])),
                     zero_time_flags_exact=bool(np.array_equal(z['before_violation'],z['after_violation'])),
                     max_q_drift_rad=float(np.abs(d['q']-d['q_initial']).max()),
                     source_sha256=hashlib.sha256((path/'protocol.json').read_bytes()).hexdigest(),
                     commands_zero=True,targets_constant_exact=True))
result=dict(schema='safeduo.zero_initial_diagnostic.v1',rows=rows,windows=192,new_random_windows=0,
            violations=sum(r['violations'] for r in rows),initial_exempt_table_overlap=sum(r['initial_exempt_table_overlap'] for r in rows),
            negative_table_exemption_withdrawal=sum(r['negative_table_exemption_withdrawal'] for r in rows),
            inference='zero-time refresh did not change measured geometry; zero-input dynamics and exemption withdrawal reproduce failures; neither proves all remaining mechanisms')
(HERE/'zero_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
