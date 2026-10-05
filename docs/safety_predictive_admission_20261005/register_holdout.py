"""Register all new bank hashes before executing any held-out random policy."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_risk_strata_20261004'))
from risk_recipe import validate_bank

def main():
    # Import is delayed because candidate registration must already exist.
    from register_candidate import job
    path=HERE/'holdout_plan.json';assert not path.exists()
    p=json.loads((HERE/'candidate_registration.json').read_text())
    root=Path('/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005/holdout_banks')
    assert (root/'sampling_summary.json').exists()
    helpers=dict(p['research_source_sha256'])
    helpers[str(HERE/'candidate_registration.json')]=hashlib.sha256((HERE/'candidate_registration.json').read_bytes()).hexdigest()
    for seed in p['seeds']:
        b=root/str(seed);m=json.loads((b/'metadata.json').read_text())
        with np.load(b/'bank.npz') as z:validate_bank(z['accepted_q'],z['risk_pair_index'],m)
        assert m['risk_quotas']==[8]*6 and m['general_count']==16
        for n in ('bank.npz','metadata.json'):helpers[str(b/n)]=hashlib.sha256((b/n).read_bytes()).hexdigest()
    for n,s in helpers.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==s,n
    jobs=[job(seed,mode,root/str(seed),'cuda:0') for seed in p['seeds'] for mode in p['modes']]
    plan={**copy.deepcopy(p),'jobs':jobs,'research_source_sha256':helpers,
          'registered_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(root.parent/'holdout_cells'),
          'planned_cells':15,'planned_windows':960,'new_unique_command_windows':192}
    path.write_text(json.dumps(plan,indent=2)+'\n');print('All192 initial references and15 conditions registered before independent pressure outcomes.')

if __name__=='__main__':main()
