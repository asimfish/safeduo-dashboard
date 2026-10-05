import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=json.loads((HERE.parent/'safety_risk_strata_20261004/risk_bank_registration.json').read_text())
path=HERE/'bank_registration.json';assert not path.exists()
helpers={**old['research_source_sha256'],**{str(HERE/n):hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['PLAN.md','holdout_bank.py','register_bank.py']}}
for n,s in helpers.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==s,n
plan={**old,'registered_utc':datetime.now(timezone.utc).isoformat(),'seeds':[331047829,389116237,451902773],
      'research_source_sha256':helpers,'gpu':0,'output_root':'/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005/holdout_banks',
      'policy_outcomes_used':False,'safety_windows':0,'candidate_outcomes_observed':False}
path.write_text(json.dumps(plan,indent=2)+'\n');print('Three fresh seeds and initial qualification registered before policy outcomes.')
