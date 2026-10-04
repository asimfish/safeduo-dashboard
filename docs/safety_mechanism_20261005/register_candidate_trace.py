import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
plan=copy.deepcopy(json.loads((HERE/'trace_plan.json').read_text()))
frozen=json.loads((HERE/'development_plan.json').read_text())
plan['research_source_sha256'].update(frozen['research_source_sha256'])
for name in ('candidate_trace_runner.py','CANDIDATE_DIAGNOSIS_PLAN.md','register_candidate_trace.py'):
    plan['research_source_sha256'][str(HERE/name)]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
j=plan['jobs'][0];j['id']='causal_envelope_60317411';j['argv'][1]=str(HERE/'candidate_trace_runner.py')
plan.update(output_root='/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/candidate_trace_cells',
            registered_utc=datetime.now(timezone.utc).isoformat(),
            design='passive old-development candidate replay only; no tuning; no new independent commands')
path=HERE/'candidate_trace_plan.json';assert not path.exists()
path.write_text(json.dumps(plan,indent=2)+'\n')
print('registered64 repeated observation windows; no candidate parameter changes')
