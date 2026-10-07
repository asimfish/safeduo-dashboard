"""Independent registered native-data scorer; no runtime driver import."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--registration',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
r=json.loads(a.registration.read_text());raw=a.raw
rec=json.loads((raw/'recording_receipt.json').read_text())
assert rec['status']=='PASS_COMPLETE_STATIC_DIAGNOSTIC' and not (raw/'failure.txt').exists()
assert rec['registration_sha256']==sha(a.registration)
for name, key in [('native_metadata.json','metadata_sha256'),('contact_identities.json','identities_sha256'),('initial_states.json','initial_states_sha256'),('sparse_contact_points.json','sparse_contact_points_sha256')]:
    assert sha(raw/name)==rec[key], name
rows={};next_step=0
for c in rec['chunks']:
    assert c['first_step']==next_step and sha(raw/c['file'])==c['sha256']
    with np.load(raw/c['file'],allow_pickle=False) as d:
        assert np.array_equal(d['step'],np.arange(c['first_step'],c['last_step']+1))
        for k in d.files: rows.setdefault(k,[]).append(d[k])
    next_step=c['last_step']+1
assert next_step==r['steps']==rec['steps']
d={k:np.concatenate(v) for k,v in rows.items()}
assert all(np.isfinite(v).all() for v in d.values())
assert np.allclose(d['time_s'],(d['step']+1)*r['physics_dt_s'],atol=1e-10,rtol=0)
for img in rec['images']: assert sha(raw/img['file'])==img['sha256']
assert len(rec['images'])==len(r['capture_steps'])*len(r['capture_envs'])*len(r['views'])
assert all(all(v==0 for v in c['max_abs_diff'].values()) for c in rec['render_checks'])
meta=json.loads((raw/'native_metadata.json').read_text())
ident=json.loads((raw/'contact_identities.json').read_text())
legacy=json.loads(Path(r['legacy_metadata']).read_text())
cr=r['criteria'];cases=[]
for case in r['cases']:
    e=case['env'];hands=[]
    for arm, m in meta['arms'].items():
        ids=m['hand_ids'];old=np.asarray(legacy['arms'][arm]['hand_default_rad'][0],dtype=np.float32)
        lim=np.asarray(m['soft_limits_rad'],dtype=np.float32)[e,ids]
        far=np.where(np.abs(lim[:,1]-old)>=np.abs(lim[:,0]-old),lim[:,1],lim[:,0])
        opened=old.copy()
        for j,jid in enumerate(ids):
            if arm.startswith('U') and 'thumb_1_joint' in m['joint_names'][jid]: opened[j]=r['target_open_thumb_rad']
        q=d[arm+':q'][:,e][:,ids];tgt=d[arm+':target'][:,e][:,ids]
        net=d[f'e{e}:{arm}:hand_net'];pairs=d[f'e{e}:{arm}:hand_partner_normal']
        pn=np.linalg.norm(pairs,axis=-1);bn=np.linalg.norm(net,axis=-1)
        acc=float(np.abs(net-pairs.sum(-2)).max())
        peak_index=np.unravel_index(int(np.argmax(pn)),pn.shape)
        hi=next(v for v in ident['hands'] if v['env']==e and v['arm']==arm)
        peak=dict(normal_n=float(pn[peak_index]),step=int(peak_index[0]),time_s=float(d['time_s'][peak_index[0]]),sensor=hi['sensors'][peak_index[1]],partner=hi['filters'][peak_index[1]][peak_index[2]])
        limits=np.asarray(m['native_limits_rad'])[e,ids]
        within=bool(np.all(tgt>=limits[:,0]-1e-6) and np.all(tgt<=limits[:,1]+1e-6))
        windows={}
        for name,(start,end) in r['evaluation_windows_s'].items():
            mask=(d['time_s']>=start)&(d['time_s']<end);assert mask.sum()>0
            expected=opened
            if name.startswith('closed'):
                event=next(v for v in r['hand_events'] if v['time_s']==(3 if name=='closed_calibration' else 9))['arms'][arm]
                f=np.asarray([event['fraction_thumb'] if event.get('fraction_thumb') is not None and 'thumb' in m['joint_names'][jid] else event['fraction'] for jid in ids],dtype=np.float32)
                expected=old+f*(far-old)
            err=float(np.abs(q[mask]-expected).max());terr=float(np.abs(tgt[mask]-expected).max())
            pair=float(pn[mask].max());body=float(bn[mask].max())
            # Startup pose can begin at zero in deliberately unsafe controls; attainment judged separately.
            angle_ok=err<=(cr['closed_tracking_error_max_rad'] if name.startswith('closed') else cr['open_error_max_rad'])
            passed=angle_ok and terr<=cr['target_error_max_rad'] and pair<=cr['pair_normal_max_n'] and body<=cr['body_net_max_n'] and acc<=cr['net_filter_accounting_max_abs_n'] and within
            windows[name]=dict(states=int(mask.sum()),max_error_rad=err,target_error_rad=terr,reference_rad=expected.tolist(),pair_normal_max_n=pair,body_net_max_n=body,qualified=bool(passed))
        hands.append(dict(arm=arm,windows=windows,net_filter_accounting_max_abs_n=acc,whole_path_peak=peak,all_targets_inside_native_limits=within,
             whole_path_contacts_qualified=bool(pn.max()<=cr['pair_normal_max_n'] and bn.max()<=cr['body_net_max_n'] and acc<=cr['net_filter_accounting_max_abs_n']),
             first_step=dict(q_rad=q[0].tolist(),target_rad=tgt[0].tolist(),pair_normal_max_n=float(pn[0].max()))))
    cases.append(dict(**case,hands=hands,startup_qualified=all(v['windows']['startup']['qualified'] for v in hands),
                      all_windows_qualified=all(all(w['qualified'] for w in h['windows'].values()) for h in hands),
                      whole_path_contacts_qualified=all(h['whole_path_contacts_qualified'] for h in hands)))
result=dict(status='COMPLETE_STATIC_INITIALIZATION_DIAGNOSTIC_NOT_TASK_ACCEPTANCE',run=r['run'],constructor_thumb_rad=r['constructor_thumb_rad'],registration_sha256=sha(a.registration),receipt_sha256=sha(raw/'recording_receipt.json'),
            cases=cases,qualified_startup=sum(c['startup_qualified'] for c in cases),qualified_all_windows=sum(c['all_windows_qualified'] for c in cases),qualified_whole_path_contacts=sum(c['whole_path_contacts_qualified'] for c in cases),
            static_environments=len(cases),environment_states=len(cases)*r['steps'],verified_original_images=len(rec['images']),criteria=cr,new_task_trials=0,new_final_trials=0,constructor_internal_contact_forces_measured=False,limitations=r['limitations'])
a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
