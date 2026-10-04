import hashlib
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
original=json.loads((OUT/'campaign_plan.json').read_text())
continuation=json.loads((OUT/'campaign_plan_continuation.json').read_text())
assert not (OUT/'campaign_plan_dynamic.json').exists()
jobs=[]
for j in original['jobs']:
    if j['expected']['method']!='system0':continue
    argv=j['argv'].copy();argv[1]=str(OUT/'wide_runner_dynamic.py')
    jobs.append({**j,'id':j['id']+'_dynamic','argv':argv})
plan={**continuation,'jobs':jobs,'output_root':str(Path(original['output_root']).parent/'cells_dynamic'),
      'candidate':'dynamic critical rows, budget512, actor32 frozen; evaluation-only',
      'prior_campaign_paths':[str(Path(original['output_root'])/'campaign.json'),str(Path(continuation['output_root'])/'campaign.json')],
      'planned_cells':9,'planned_windows':576}
plan['research_source_sha256']=continuation['research_source_sha256'].copy()
for name in ['wide_runner_dynamic.py','CAPACITY_PLAN.md','register_dynamic.py']:
    p=OUT/name;plan['research_source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'campaign_plan_dynamic.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'registered_dynamic_cells':len(jobs),'new_unique_tapes':0,'budget':512}))
