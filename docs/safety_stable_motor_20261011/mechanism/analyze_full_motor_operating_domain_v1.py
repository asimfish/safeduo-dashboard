"""Analyze all50 independent coordinates; proposals are not native trials."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from audit_coupled_hand_targets_native_v1 import matrix

H=Path(__file__).resolve().parent
OUT=H/'full_motor_operating_domain_v1'
ARMS=('F_L','F_R','U_L','U_R')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not OUT.exists()
    baseline=H/'fourhand_native_probe_v31'
    assert read(baseline/'ACTUAL_COMPLETION_EXIT_V31.json')['actual_outer_exit']==0
    assert read(baseline/'INDEPENDENT_AUDIT_ACTUAL_EXIT_V1.json')['actual_outer_exit']==0
    reg=read(baseline/'REGISTRATION_V1.json')
    meta=read(reg['coupled_assets'])
    paramsfile=Path(read(baseline/'RESOURCE_EXECUTION_V1.json')['out'])/'native_parameters.npz'
    data=np.load(paramsfile,allow_pickle=False)
    labels=[];lower=[];upper=[];vcaps=[];blocks=[];rows=[]
    for arm in ARMS:
        asset=meta['assets'][arm]
        names=data[arm+'_joint_names'].tolist()
        hard=data[arm+'_hard_limits'][0].astype(np.float64)
        vmax=data[arm+'_max_velocity'][0].astype(np.float64)
        ids=data[arm+'_controlled_indices'].reshape(-1).tolist()
        n=len(ids)
        assert n==7 if arm.startswith('F') else n==6
        start=len(labels)
        labels.extend(arm+'/'+names[i] for i in ids)
        lower.extend(hard[ids,0]);upper.extend(hard[ids,1]);vcaps.extend(vmax[ids])
        roots,A,b=matrix(asset['hand_joint_names'],asset['original_URDF'])
        assert len(roots)==6 and set(roots)==set(asset['independent_motor_joint_names'])
        handids=[names.index(name) for name in asset['hand_joint_names']]
        lo=np.full(6,-np.inf);hi=np.full(6,np.inf);speed=np.full(6,np.inf)
        for i in range(len(handids)):
            nonzero=np.flatnonzero(A[i]);assert len(nonzero)==1
            j=int(nonzero[0]);s=A[i,j]
            bound=(hard[handids[i]]-b[i])/s
            lo[j]=max(lo[j],bound.min());hi[j]=min(hi[j],bound.max())
            speed[j]=min(speed[j],vmax[handids[i]]/abs(s))
        assert np.isfinite([lo,hi,speed]).all() and (hi>lo).all() and (speed>0).all()
        motorstart=len(labels)
        labels.extend(arm+'/'+name for name in roots)
        lower.extend(lo);upper.extend(hi);vcaps.extend(speed)
        blocks.append(dict(arm=arm,armids=ids,start=start,motorstart=motorstart,handids=handids,A=A,b=b,asset=asset,names=names))
        rows.append(dict(arm=arm,arm_coordinates=n,independent_hand_motors=6,dependent_hand_coordinates=6,
                         motor_domain=[dict(name=name,lower_rad=float(lo[j]),upper_rad=float(hi[j]),
                                            maximal_relation_consistent_speed_rad_s=float(speed[j])) for j,name in enumerate(roots)]))
    assert len(labels)==50 and len(set(labels))==50
    lower=np.asarray(lower);upper=np.asarray(upper);vcaps=np.asarray(vcaps)
    assert np.isfinite([lower,upper,vcaps]).all() and (upper>lower).all()
    # This public development seed will never be a formal holdout seed.
    seed=320250;count=16000;rng=np.random.default_rng(seed)
    unit=rng.random((count,50));velocity_unit=rng.uniform(-1,1,(count,50))
    q=lower+unit*(upper-lower)
    fractions=np.array([0,.25,.5,1])[np.arange(count)%4]
    v=velocity_unit*fractions[:,None]*vcaps
    target=lower+rng.random((count,50))*(upper-lower)
    # Boundary challenges kept in addition to the original uniform proposals.
    boundary_unit=rng.choice([1e-5,1-1e-5],size=(4000,50))
    boundary_q=lower+boundary_unit*(upper-lower)
    boundary_v=rng.choice([-1.,1.],size=(4000,50))*.99999*vcaps
    q=np.concatenate([q,boundary_q]);v=np.concatenate([v,boundary_v])
    target=np.concatenate([target,lower+(1-boundary_unit)*(upper-lower)])
    total=len(q);assert total==20000
    arrays=dict(independent_q=q,independent_qd=v,independent_target=target,
                independent_lower_rad=lower,independent_upper_rad=upper,
                independent_speed_cap_rad_s=vcaps,independent_names=np.asarray(labels),
                proposal_id=np.arange(total),proposal_stratum=np.array(['uniform']*count+['boundary_and_opposite_target']*4000))
    sourcefiles=[Path(__file__),baseline/'REGISTRATION_V1.json',baseline/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
                 baseline/'ACTUAL_COMPLETION_EXIT_V31.json',paramsfile,Path(reg['coupled_assets'])]
    for block,row in zip(blocks,rows):
        a=block['arm'];names=block['names'];fullq=np.zeros((total,len(names)));fullv=np.zeros_like(fullq);fulltarget=np.zeros_like(fullq)
        s=block['start'];n=len(block['armids']);m=block['motorstart']
        for full,independent in [(fullq,q),(fullv,v),(fulltarget,target)]:
            full[:,block['armids']]=independent[:,s:s+n]
            full[:,block['handids']]=independent[:,m:m+6]@block['A'].T
        fullq[:,block['handids']]+=block['b'];fulltarget[:,block['handids']]+=block['b']
        hard=data[a+'_hard_limits'][0];maxv=data[a+'_max_velocity'][0]
        assert np.all(fullq>=hard[:,0]-1e-12) and np.all(fullq<=hard[:,1]+1e-12)
        assert np.all(fulltarget>=hard[:,0]-1e-12) and np.all(fulltarget<=hard[:,1]+1e-12)
        assert np.all(abs(fullv)<=maxv+1e-12)
        residual=velocity_residual=0.
        for joint in ET.parse(block['asset']['original_URDF']).findall('joint'):
            mimic=joint.find('mimic')
            if mimic is None or joint.get('name') not in names:continue
            slave=names.index(joint.get('name'));master=names.index(mimic.get('joint'))
            mul=float(mimic.get('multiplier','1'));off=float(mimic.get('offset','0'))
            residual=max(residual,float(abs(fullq[:,slave]-mul*fullq[:,master]-off).max()),
                         float(abs(fulltarget[:,slave]-mul*fulltarget[:,master]-off).max()))
            velocity_residual=max(velocity_residual,float(abs(fullv[:,slave]-mul*fullv[:,master]).max()))
        assert residual<=1e-12 and velocity_residual<=1e-12
        row.update(proposal_position_relation_max_rad=residual,proposal_velocity_relation_max_rad_s=velocity_residual)
        for key,value in [('q',fullq),('qd',fullv),('target',fulltarget)]:arrays[a+'_'+key]=value
        sourcefiles.append(Path(block['asset']['original_URDF']))
    span=(q[:count].max(axis=0)-q[:count].min(axis=0))/(upper-lower)
    bins=[len(set(np.minimum(9,np.floor((q[:count,i]-lower[i])/(upper[i]-lower[i])*10)).astype(int))) for i in range(50)]
    assert min(bins)==10
    OUT.mkdir();np.savez_compressed(OUT/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz',**arrays)
    report=dict(status='PASS_FULL50_SOFTWARE_DOMAIN_AND_ORIGINAL74_MAPPING_ONLY',
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),development_seed=seed,
                development_uniform_proposals=count,additional_boundary_proposals=4000,
                total_proposals=total,independent_motor_coordinates=50,arm_coordinates=26,
                independent_hand_motor_coordinates=24,mapped_hand_coordinates=48,native_joint_coordinates=74,
                rows=rows,uniform_marginal_span_fraction=span.tolist(),uniform_bins_each_coordinate=bins,
                uniform_min_marginal_span_fraction=float(span.min()),
                source_SHA256={str(file):sha(file) for file in sourcefiles},
                proposal_SHA256=sha(OUT/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz'),
                formal_holdout_count=0,native_proposals_executed=0,
                all_native_hand_initial_support_remains_unvalidated=True,
                physical_geometry_reachability_contacts_or_safe_volume_not_measured=True,
                object_layout_noise_fault_randomization_not_in_this_joint_domain_dataset=True,
                maximum_native_safe_or_reachable_coverage_not_claimed=True,
                future_execution_requires_frozen_qualified_candidate_and_explicit_initial_geometry_rejection_log=True,
                fullSystem0_accepted=False)
    (OUT/'FULL_MOTOR_DOMAIN_V1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],total,'proposals',float(span.min()),flush=True)


if __name__=='__main__':
    main()
