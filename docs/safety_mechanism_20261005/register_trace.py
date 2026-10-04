import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'safety_risk_strata_20261004'
path=HERE/'trace_plan.json';assert not path.exists()
base=json.loads((OLD/'campaign_plan.json').read_text())
job=next(x for x in base['jobs'] if x['id']=='risk_60317411_system0')
argv=job['argv'].copy();argv[1]=str(HERE/'trace_runner.py');argv+=['--causal-trace','--kinematic-trace']
helpers={**base['research_source_sha256'],**{str(HERE/n):hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['PLAN.md','trace_runner.py','register_trace.py']}}
plan={**base,'jobs':[{**job,'id':'causal_baseline_60317411','argv':argv}],
      'research_source_sha256':helpers,'registered_utc':datetime.now(timezone.utc).isoformat(),
      'output_root':'/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/trace_cells',
      'planned_cells':1,'planned_windows':64,'new_unique_command_windows':0,'design':'observation-only full16s replay; actual dynamic row width unchanged'}
for name,sha in helpers.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
for name,sha in base['source_sha256'].items():assert hashlib.sha256((Path(base['cwd'])/name).read_bytes()).hexdigest()==sha,name
path.write_text(json.dumps(plan,indent=2)+'\n');print('Registered one observational replay; new independent windows0.')
