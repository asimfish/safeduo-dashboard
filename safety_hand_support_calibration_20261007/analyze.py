"""Independent NumPy reduction of sealed static diagnostic records."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze(raw, registration):
    r=json.loads(registration.read_text()); receipt=json.loads((raw/'recording_receipt.json').read_text())
    assert receipt['status']=='PASS_COMPLETE_STATIC_DIAGNOSTIC' and not (raw/'failure.txt').exists()
    assert receipt['registration_sha256']==sha(registration)
    for file, key in [('native_metadata.json','metadata_sha256'),('contact_identities.json','identities_sha256'),('sparse_contact_points.json','sparse_contact_points_sha256')]:
        assert sha(raw/file)==receipt[key]
    values={}; expected=0
    for chunk in receipt['chunks']:
        assert chunk['first_step']==expected and sha(raw/chunk['file'])==chunk['sha256']
        with np.load(raw/chunk['file'],allow_pickle=False) as data:
            assert np.array_equal(data['step'],np.arange(chunk['first_step'],chunk['last_step']+1))
            for key in data.files:values.setdefault(key,[]).append(data[key])
        expected=chunk['last_step']+1
    assert expected==r['steps']==receipt['steps']
    d={key:np.concatenate(val) for key,val in values.items()}
    assert np.allclose(d['time_s'],(d['step']+1)*r['physics_dt_s'],atol=1e-10,rtol=0)
    for image in receipt['images']: assert sha(raw/image['file'])==image['sha256']
    meta=json.loads((raw/'native_metadata.json').read_text());ident=json.loads((raw/'contact_identities.json').read_text())
    arms=list(meta['arms']);results=[]
    for case in r['cases']:
        e=case['env']; summaries=[]
        for arm in arms:
            m=meta['arms'][arm];hid=m['hand_ids'];aid=m['arm_ids'];default=np.asarray(m['hand_default_rad'])[e]
            q=d[arm+':q'][:,e,hid];target=d[arm+':target'][:,e,hid]
            lim=np.asarray(m['native_limits_rad'])[e,hid]
            base=dict(arm=arm,hand_joint_names=[m['joint_names'][j] for j in hid],
                default_inside_native_limits=bool(np.all(default>=lim[:,0]-1e-6) and np.all(default<=lim[:,1]+1e-6)),
                getter_cache_max_abs_rad=float(np.max(np.abs(d[arm+':q'][:,e]-d[arm+':cache_q'][:,e]))),
                actual_arm_reference_max_error_rad=float(np.max(np.abs(d[arm+':q'][:,e,aid]-np.asarray(m['arm_reference_rad'])[e]))),windows={})
            for label,(start,end) in r['evaluation_windows_s'].items():
                idx=(d['time_s']>=start)&(d['time_s']<end)
                err=np.abs(q[idx]-default); terr=np.abs(target[idx]-default)
                maxerr=err.max(-1);net=d[f'e{e}:{arm}:hand_net'][idx]
                base['windows'][label]=dict(states=int(idx.sum()),default_open_states=int((maxerr<=r['unchanged_previous_gate']['open_max_error_rad']).sum()),
                    max_error_to_default_rad=float(maxerr.max()),median_error_to_default_rad=float(np.median(maxerr)),
                    target_max_error_to_default_rad=float(terr.max()),max_tracking_error_rad=float(np.max(np.abs(q[idx]-target[idx]))),
                    mean_actual_joint_rad=q[idx].mean(0).tolist(),mean_target_joint_rad=target[idx].mean(0).tolist(),
                    mean_hand_net_sum_n=float(np.linalg.norm(net,axis=-1).sum(-1).mean()),
                    max_hand_net_sum_n=float(np.linalg.norm(net,axis=-1).sum(-1).max()))
            summaries.append(base)
        results.append(dict(**case,hands=summaries))
    support=[];reconstruction_max=0.;net_normal_max=0.;net_total_max=0.
    for item in ident['objects']:
        e=item['env'];obj=item['object'];key=f'e{e}:{obj}:'
        nf=d[key+'normal'];fr=d[key+'friction'];net=d[key+'net'];nrec=d[key+'normal_reconstructed']
        nr=float(np.max(np.abs(nf-nrec))); nn=float(np.max(np.abs(net-nf.sum(1))));nt=float(np.max(np.abs(net-(nf+fr).sum(1))))
        reconstruction_max=max(reconstruction_max,nr);net_normal_max=max(net_normal_max,nn);net_total_max=max(net_total_max,nt)
        if r['cases'][e]['kind']!='support': continue
        idx=(d['time_s']>=9.)&(d['time_s']<11.999)
        pose=d[key+'pose'][idx];vel=d[key+'velocity'][idx]
        mass=np.asarray(meta['masses_kg'][obj]).reshape(-1)[e];weight=mass*9.81
        components=[]
        for j,path in enumerate(item['filters']):
            mean_n=nf[idx,j].mean(0);mean_f=fr[idx,j].mean(0)
            if np.linalg.norm(nf[idx,j],axis=-1).max()>.001 or np.linalg.norm(fr[idx,j],axis=-1).max()>.001:
                components.append(dict(path=path,mean_normal_force_n=mean_n.tolist(),mean_friction_force_n=mean_f.tolist(),
                    mean_normal_magnitude_n=float(np.linalg.norm(nf[idx,j],axis=-1).mean()),
                    max_normal_magnitude_n=float(np.linalg.norm(nf[idx,j],axis=-1).max())))
        table=np.asarray([j for j,path in enumerate(item['filters']) if '/Table' in path],dtype=int)
        table_z=nf[idx][:,table,2].sum(-1)
        support.append(dict(env=e,label=r['cases'][e]['label'],object=obj,mass_kg=float(mass),weight_n=float(weight),
            final_center_local_m=(pose[-1,:3]-np.asarray(meta['origins'])[e]).tolist(),
            max_linear_speed_m_s=float(np.linalg.norm(vel[:,:3],axis=-1).max()),max_angular_speed_rad_s=float(np.linalg.norm(vel[:,3:],axis=-1).max()),
            mean_net_contact_n=net[idx].mean(0).tolist(),mean_normal_sum_n=nf[idx].sum(1).mean(0).tolist(),mean_friction_sum_n=fr[idx].sum(1).mean(0).tolist(),
            table_only_support_weight_fraction=float(table_z.mean()/weight),
            net_z_weight_relative_error=float(np.max(np.abs(net[idx,2]-weight)/weight)),
            normal_reconstruction_max_abs_n=nr,net_minus_normal_sum_max_abs_n=nn,net_minus_normal_and_friction_max_abs_n=nt,
            components=components))
    gates=r['support_diagnostic_limits']
    full_normal_qualified=reconstruction_max<=gates['normal_reconstruction_max_abs_n'] and net_normal_max<=gates['net_filter_sum_max_abs_n']
    static_support_qualified=full_normal_qualified and all(s['net_z_weight_relative_error']<=gates['stationary_weight_relative_error'] for s in support)
    return dict(status='COMPLETE_STATIC_DIAGNOSTIC_NOT_TASK_ACCEPTANCE',registration_sha256=sha(registration),
        raw_receipt_sha256=sha(raw/'recording_receipt.json'),new_task_trials=0,new_final_trials=0,cases=results,support=support,
        measurement=dict(full_normal_contact_accounting_qualified=bool(full_normal_qualified),static_support_weight_qualified=bool(static_support_qualified),
            normal_reconstruction_max_abs_n=reconstruction_max,net_minus_filtered_normal_max_abs_n=net_normal_max,
            net_minus_filtered_normal_and_friction_max_abs_n=net_total_max,
            max_normal_count=receipt['max_normal_count'],max_friction_count=receipt['max_friction_count'],capacity=receipt['contact_capacity'],
            collision_shapes=len(meta['collision_inventory']),instance_proxy_collision_shapes=sum(c['instance_proxy'] for c in meta['collision_inventory']),
            partner_count_env0=len(meta['partners_env0']),verified_images=len(receipt['images']),native_render_checks=len(receipt['render_checks'])),
        limitations=['No task trial, System0/policy bypassed; no new grasp/safety performance result.',
            'Native contact API last physics step sampled, not all internal solver iterations.',
            'Contact net may represent normal-only; normal and friction accounting hypotheses reported separately.',
            'Static normal/support qualification does not qualify dynamic friction, complete wrench or tactile sensing.',
            'Fixed arm targets can track imperfectly; actual arm error retained. Four narrow task poses are not workspace coverage.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=analyze(a.raw,a.registration);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],measurement=result['measurement']),indent=2))
