import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'safety_risk_strata_20261004'
path=HERE/'holdout_registration.json';assert not path.exists()
base=json.loads((OLD/'risk_bank_registration.json').read_text())
helpers={**base['research_source_sha256'],**{str(HERE/n):hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['HOLDOUT_PLAN.md','holdout_bank.py','register_holdout.py']}}
plan={**base,'registered_utc':datetime.now(timezone.utc).isoformat(),'seeds':[152684921,198470327,237901613],
      'research_source_sha256':helpers,'gpu':0,'output_root':'/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/holdout_banks',
      'policy_outcomes_used':False,'safety_windows':0,'candidate_not_selected_yet':True}
for n,sha in helpers.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==sha,n
path.write_text(json.dumps(plan,indent=2)+'\n');print('Independent holdout seeds registered before mechanism selection.')
