"""Posthoc contact-policy attribution; never changes registered raw-force gate."""
from pathlib import Path
import json,sys,importlib.util,hashlib,yaml
import numpy as np
H=Path(__file__).resolve().parent
R=Path('/mnt/nas/data/lyf/double_hand/safety_feasible_response_20261009_0316')/(sys.argv[1]+'_v1')
source=Path('/home/liyufeng/safeduo/src/safeduo/safety/semantics.py')
spec=importlib.util.spec_from_file_location('contact_semantics_readonly',source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
cfgroot=Path('/home/liyufeng/safeduo/src/safeduo/configs')
def merge(a,b):
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(a.get(k),dict):merge(a[k],v)
        else:a[k]=v
    return a
def loadcfg(n):
    y=yaml.safe_load((cfgroot/n).read_text())
    return merge(loadcfg(y['extends']) if y.get('extends') else {},{k:v for k,v in y.items() if k!='extends'})
resolved_cfg=loadcfg('duo_env_a31_pending_guard.yaml')
policy=Path('/home/liyufeng/safeduo')/resolved_cfg['assets']['semantics_yaml']
sem=mod.ContactSemantics(str(policy))
identity=json.loads((R/'native_contact_identity.json').read_text())
hand_native_names={}
for a in ['F_L','F_R','U_L','U_R']:
    side='left' if a.endswith('L') else 'right'
    f=Path('/home/liyufeng/safeduo/assets_src/real')/(f'inspire_rh56f2_{side}_v5.yaml' if a.startswith('F') else f'rh56dfx_{side}_v7.yaml')
    y=yaml.safe_load(f.read_text());hand_native_names[a]=set(y['collision_spheres'])
def qualified(p):
    for a in ['F_L','F_R','U_L','U_R']:
        if '/'+a+'/' in p:
            link=p.rsplit('/',1)[-1]
            return a+'/'+('hand/' if link in hand_native_names[a] or '/f2_hand/' in p or (a.startswith('U') and link.startswith(('left_','right_'))) else '')+link
    if '/TableF/' in p:return 'table_F'
    if '/TableU/' in p:return 'table_U'
    if '/ground' in p:return 'ground'
    return 'UNKNOWN:'+p
peaks={};exceed={};examples={};observations={};matrices={}
for v in identity['views']:
    first_env=next(i for i,p in enumerate(v['sensors']) if '/env_0/' in p)
    local_sensors=[p for p in v['sensors'] if '/env_0/' in p]
    template={p.rsplit('/',1)[-1]:i for i,p in enumerate(local_sensors)}
    base=np.asarray([[sem.judge(qualified(sensor),qualified(partner)).verdict for partner in v['filters'][first_env]] for sensor in local_sensors])
    verdict=base[[template[p.rsplit('/',1)[-1]] for p in v['sensors']]]
    matrices[v['arm']]=verdict
    for cat in np.unique(verdict):
        peaks.setdefault(str(cat),np.zeros(64));exceed.setdefault(str(cat),np.zeros(64,dtype=bool));observations.setdefault(str(cat),0)
for p in sorted((R/'native_contacts').glob('physics_events*.npz')):
    with np.load(p) as z:
        for v in identity['views']:
            a=v['arm'];scalar=z[a+'_partner_scalar_abs_N'];ids=np.asarray(v['env_ids']);matrix=matrices[a]
            for cat in np.unique(matrix):
                cat=str(cat);masked=np.where(matrix[None]==cat,scalar,0)
                per_sensor=masked.max((0,2));np.maximum.at(peaks[cat],ids,per_sensor)
                observations[cat]+=int((masked>.1).sum())
                if masked.max()>examples.get(cat,{}).get('peak_N',0):
                    t,si,pi=np.unravel_index(masked.argmax(),masked.shape)
                    examples[cat]=dict(peak_N=float(masked[t,si,pi]),lane=int(ids[si]),macro=int(z['frame'][t]),substep=int(z['substep'][t]),sensor=v['sensors'][si],partner=v['filters'][si][pi],qualified_sensor=qualified(v['sensors'][si]),qualified_partner=qualified(v['filters'][si][pi]))
result=dict(status='POSTHOC_RAW_CONTACT_STATIC_SEMANTICS_ATTRIBUTION',policy_path=str(policy),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
    resolved_sphere_mode=resolved_cfg['assets']['spheres'], inherited_config_resolved=True, primary_force_gate_unchanged=True,secondary_only=True,whole_run_closed=(R/'point_contact_receipts.json').exists(),
    scope='Static semantic verdict only. Conditional permission is not evaluated; an adjacency exemption is not a force or physical assembly certificate.',
    by_verdict={cat:dict(lanes_over_0p1N=int((p>.1).sum()),lane_peak_scalar_N=p.tolist(),sensor_partner_micro_observations_over_gate=observations[cat],maximum=examples.get(cat)) for cat,p in peaks.items()})
(H/('CONTACT_SEMANTICS_'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],{k:(v['lanes_over_0p1N'],v['maximum']) for k,v in result['by_verdict'].items()})
