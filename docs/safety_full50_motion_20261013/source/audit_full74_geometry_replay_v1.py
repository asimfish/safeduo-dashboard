"""Every recorded pair compared with frozen original native geometry fields."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from full74_geometry_v1 import Full74Geometry,ARMS
from hand_kinematic_replay_v1 import quaternion_matrix

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
E=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def original_openhand_gaps(provider,full_q,armids):
    centers=provider.fk_all({a:torch.as_tensor(full_q[a][:,armids[a]],dtype=torch.float32) for a in ARMS})['centers']
    m=provider.sph;r=m.radii;parts=[]
    for pairs in (m.pairs_cross,m.pairs_self):parts.append((centers[:,pairs[:,0]]-centers[:,pairs[:,1]]).norm(dim=-1)-(r[pairs[:,0]]+r[pairs[:,1]]))
    pair=m.pairs_table;box=(centers[:,pair[:,0]]-torch.tensor(provider.layout.table_centers)[pair[:,1]]).abs()-torch.tensor(provider.layout.table_half_extents)[pair[:,1]]
    parts.append(box.clamp_min(0).norm(dim=-1)+box.amax(-1).clamp_max(0)-r[pair[:,0]])
    return torch.cat(parts,dim=-1).numpy()


def main():
    output=E/'FULL74_GEOMETRY_REPLAY_AUDIT_V1.json';assert not output.exists()
    plan=read(E/'FULL74_GEOMETRY_PROSPECTIVE_PLAN_V1.json');rows=[];sources={str(E/'full74_geometry_v1.py'):sha(E/'full74_geometry_v1.py'),str(Path(__file__)):sha(__file__)}
    for v in plan['native_replay_versions']:
        p=H/f'fourhand_native_probe_v{v}';reg=read(p/'REGISTRATION_V1.json');resource=read(p/'RESOURCE_EXECUTION_V1.json')
        assert resource['actual_exit']==resource['actual_native_exit']==0 and resource['native_owned_process_reaped_and_absent']
        for f,h in reg['source_sha256'].items():assert sha(f)==h,f
        raw=Path(resource['out']);assets=read(reg['coupled_assets'])['assets']
        with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
        fk=Full74Geometry(params,assets);gap_max=0.;legacy_max=0.;events=0;body_max=0.;rotation_max=0.;worst=None
        rows_by_arm={a:{'arm':a,'maximum_position_error_m':0.,'maximum_rotation_error_rad':0.} for a in ARMS}
        independent_geometry_reconstruction_max=0.;sign_disagreements=0;unsafe_states=0
        for c in read(raw/'physics_chunks.json'):
            f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
            with np.load(f) as z:
                full={a:z[a+'_native_q'][:,0] for a in ARMS};pred=fk.transforms(full)
                gaps=fk.gaps_from_centers(fk.centers_from_transforms(pred));actual=z['all9021_raw_geometry'][:,0]
                delta=abs(gaps-actual);m=float(delta.max())
                if m>gap_max:
                    gap_max=m;i,j=np.unravel_index(delta.argmax(),delta.shape);worst={'event':int(z['physics_event'][i]),'pair':int(j),'predicted_m':float(gaps[i,j]),'native_m':float(actual[i,j])}
                old=original_openhand_gaps(fk.provider,full,fk.armids);legacy_max=max(legacy_max,float(abs(old-actual).max()))
                sign_disagreements+=int(((gaps.min(axis=1)<0)!=(actual.min(axis=1)<0)).sum());unsafe_states+=int((actual.min(axis=1)<0).sum())
                native_transforms={}
                for a in ARMS:
                    native=z[a+'_native_body_transforms_after_xyzw'][:,0];native_transforms[a]={}
                    for j,b in enumerate(fk.bodies[a]):
                        t=np.broadcast_to(np.eye(4),(len(native),4,4)).copy();t[:,:3,:3]=quaternion_matrix(native[:,j,3:]);t[:,:3,3]=native[:,j,:3]
                        native_transforms[a][b]=t
                        if b not in pred[a]:continue
                        pe=np.linalg.norm(pred[a][b][:,:3,3]-t[:,:3,3],axis=-1)
                        rel=pred[a][b][:,:3,:3].transpose(0,2,1)@t[:,:3,:3]
                        ae=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))
                        row=rows_by_arm[a];row['maximum_position_error_m']=max(row['maximum_position_error_m'],float(pe.max()));row['maximum_rotation_error_rad']=max(row['maximum_rotation_error_rad'],float(ae.max()))
                ng=fk.gaps_from_centers(fk.centers_from_transforms(native_transforms))
                independent_geometry_reconstruction_max=max(independent_geometry_reconstruction_max,float(abs(ng-actual).max()))
                events+=len(actual)
        body_max=max(x['maximum_position_error_m'] for x in rows_by_arm.values());rotation_max=max(x['maximum_rotation_error_rad'] for x in rows_by_arm.values())
        passed=gap_max<=plan['maximum_absolute_pair_error_m'] and body_max<=plan['maximum_body_position_error_m'] and rotation_max<=plan['maximum_body_rotation_error_rad']
        rows.append({'version':v,'events':events,'pairs_each_event':9021,'pair_comparisons':events*9021,'maximum_full74_pair_error_m':gap_max,
             'maximum_legacy_openhand_pair_error_m':legacy_max,'maximum_native_transform_geometry_reconstruction_error_m':independent_geometry_reconstruction_max,
             'maximum_body_position_error_m':body_max,'maximum_body_rotation_error_rad':rotation_max,'rows_by_arm':list(rows_by_arm.values()),
             'worst_pair_difference':worst,'native_negative_minimum_states_retained':unsafe_states,'predicted_minimum_sign_disagreements':sign_disagreements,
             'recorded_state_accuracy_pass':passed})
        for a in ARMS:sources[assets[a]['original_URDF']]=sha(assets[a]['original_URDF'])
        print('FULL74_REPLAY',v,events,'pair_error_m',gap_max,'legacy_error_m',legacy_max,'accuracy_pass',passed,flush=True)
    report={'status':'CLOSED_FULL74_NATIVE_GEOMETRY_DEVELOPMENT_REPLAY','plan':plan,'rows':rows,
            'all_recorded_state_accuracy_gates_pass':all(r['recorded_state_accuracy_pass'] for r in rows),
            'total_native_events_reused':sum(r['events'] for r in rows),'total_pair_comparisons':sum(r['pair_comparisons'] for r in rows),
            'native_wrist_anchors_used':False,'native_geometry_offsets_or_pair_fit_used':False,
            'full74_actual_coordinates_used':True,'all24_passive_followers_measured_not_synthesized':True,
            'future_tracking_or_stopping_bound_qualified':False,'random_full50_native_trajectories_executed':0,
            'new_native_jobs':0,'formal_holdouts':0,'new_training_jobs':0,'fullSystem0_accepted':False,'source_sha256':sources}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('FULL74_NATIVE_REPLAY_CLOSED',report['all_recorded_state_accuracy_gates_pass'],report['total_pair_comparisons'],flush=True)


if __name__=='__main__':main()
