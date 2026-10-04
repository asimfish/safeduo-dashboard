import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
base=json.loads((HERE/'zero_plan_v2.json').read_text())
plan=dict(registered_utc=datetime.now(timezone.utc).isoformat(),seeds=[60317411,80692357,109441003],
          risk_quota_per_seed_pair=8,general_per_seed=16,cem_max_restarts=12,cem_max_iterations=40,num_envs=64,
          source_sha256=base['source_sha256'],checkpoint_path=base['checkpoint_path'],checkpoint_sha256=base['checkpoint_sha256'],
          research_source_sha256={str(HERE/n):hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['risk_bank.py','RISK_PLAN.md','register_risk_bank.py']},
          output_root='/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/risk_banks',
          policy_outcomes_used=False,safety_windows=0,gpu=1)
path=HERE/'risk_bank_registration.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n');print('registered risk-bank geometry search and zero-input admission')
