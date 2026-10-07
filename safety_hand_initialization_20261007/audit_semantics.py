import json,hashlib
from pathlib import Path
from dataclasses import asdict
from safeduo.configs import load_config, repo_root
from safeduo.safety.semantics import ContactSemantics
p=Path(__file__).resolve().parent
r=json.loads((p/'REGISTRATION_old_default.json').read_text())
y=load_config(r['env_yaml']);yp=repo_root()/y['assets']['semantics_yaml'];sem=ContactSemantics(str(yp))
checks=[]
for arm,side in [('U_L','left'),('U_R','right')]:
    qa=f'{arm}/hand/{side}_thumb_2';qb=f'{arm}/hand/wrist_3_link'
    verdict=asdict(sem.judge(qa,qb))
    assert verdict['keep'] is False and verdict['verdict']=='adjacency_exempt'
    checks.append(dict(arm=arm,semantic_sensor=qa,semantic_partner=qb,classes=[sem.classify(qa),sem.classify(qb)],verdict=verdict))
(p/'SEMANTICS_GAP.json').write_text(json.dumps(dict(status='CONFIRMED_MODEL_HAND_INTERNAL_EXEMPTION_FOR_MEASURED_CONTACT_PAIR',
    analysis_timing='Source-path analysis after native results; not an added prespecified physical endpoint',resolved_semantics_yaml=str(yp),
    semantics_sha256=hashlib.sha256(yp.read_bytes()).hexdigest(),checks=checks,
    interpretation='Both measured thumb2/palm pairs are excluded from official distance hazard rows by hand_internal: all. This confirms a model coverage gap, not a physical collision exemption or proof that the guarded controller was exercised in these static probes.',
    recommendation='Qualify a separate hand closure/self-contact layer with native states and pre-admitted collision-free hand targets; do not simply unexempt all hand pairs in an arm-only action/Jacobian controller.',production_sources_unchanged=True),indent=2)+'\n')
print(yp);print(checks)
