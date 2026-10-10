"""Independent raw observation of env2/F_R; no simulation or view reselection."""
import itertools
import math
from pathlib import Path
import numpy as np
from evidence_io import require,parse_json,write_new,contained
import camera_adapter as h
import camera_oracle as o
from actual_angle_oracle import measured,circular_error
from receipt_schema import case_contract,group_metadata,strict_equal

CASE=dict(env_id=2,arm='F_R',native_point=0,proposal_ids=list(range(32)),physical_steps=0,
          policy_actions=0,prefix_qualification=False)


def bounded_case(receipt):
    return case_contract(receipt)


def framing_points(bounds,baseline,inventory):
    expected={p for p,m in inventory.items() if m['env_id']==2}
    require(set(bounds)==expected and len(expected)==82,'all82 native bounds inventory')
    worlds={}
    for path,row in bounds.items():
        m=inventory[path];pose=baseline[m['arm']+'_native_link_transforms_xyzw'][2,m['body_index']].astype(np.float64)
        require(np.array_equal(pose,np.asarray(row['actual_native_pose_xyzw'])),'bounds actual native pose')
        lo,hi=np.asarray(row['local_lower']),np.asarray(row['local_upper'])
        require(lo.shape==hi.shape==(3,) and np.isfinite([lo,hi]).all() and np.all(lo<=hi),'authored bound shape/finite/order')
        q=pose[3:];norm=np.linalg.norm(q);require(abs(norm-1)<=1e-4,'bounds quaternion norm')
        x,y,z,w=q/norm
        rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        worlds[path]=np.array(list(itertools.product(*zip(lo,hi))))@rot.T+pose[:3]
    rows={p:m for p,m in inventory.items() if m['env_id']==2 and m['arm']=='F_R' and m['hand']}
    require(bool(rows),'nonempty F_R hand')
    origins=np.array([baseline['F_R_native_link_transforms_xyzw'][2,m['body_index'],:3] for m in rows.values()],float)
    points=np.concatenate([origins,*[worlds[p] for p in rows]])
    return points,rows,{p:worlds[p].mean(0) for p in rows},(origins.min(0)+origins.max(0))/2


def camera_pose(rec,points,centers,center):
    candidate=rec['prospective_camera_candidate'];target=np.asarray(candidate['target_world_m'],float)
    d=np.asarray(candidate['direction'],float);require(target.shape==d.shape==(3,) and np.isfinite([target,d]).all(),'candidate shape/finite')
    require(abs(np.linalg.norm(d)-1)<1e-12 and candidate['geometry_changed'] is False,'unit camera-only direction')
    phase=candidate['phase'];path=candidate['target_native_path']
    if phase in ('legacy24','fixed_full_sphere'):require(path is None and np.allclose(target,center,atol=1e-12,rtol=0),'whole-hand aim')
    else:require(phase=='adaptive_low_observed_body_aim' and path in centers and np.allclose(target,centers[path],atol=1e-12,rtol=0),'native body aim')
    require(np.array_equal(target,rec['target_world_m']) and np.array_equal(d,rec['world_direction']),'candidate record mismatch')
    saved=np.asarray(rec['hand_framing_points_world_m'],float)
    require(saved.shape==points.shape and np.allclose(saved,points,atol=1e-12,rtol=0),'origins plus authored corners reconstruction')
    result=h.frustum(points,rec['camera']);world=np.asarray(rec['camera']['actual_camera_to_world_row_matrix'],float)
    eye=np.asarray(rec['requested_eye_world_m'],float);require(eye.shape==(3,) and np.isfinite(eye).all(),'camera eye')
    direction=eye-target;require(np.linalg.norm(direction)>0,'zero eye-target distance');direction/=np.linalg.norm(direction)
    require(np.allclose(world[3,:3],eye,atol=2e-5,rtol=0) and np.allclose(world[2,:3],direction,atol=2e-5,rtol=0) and
            np.allclose(direction,d,atol=2e-5,rtol=0),'actual/requested camera pose')
    require(type(rec['fit_operations']) is int and 1<=rec['fit_operations']<=3,'bounded analytic fit')
    # FINAL native matrix and fixed framing-points AABB center define the view.
    # The optical aim target is independent and may change during adaptive search.
    observation=measured(rec['camera'],points);angle=observation['actual_azimuth_deg']
    planned=float(np.degrees(np.arctan2(d[1],d[0]))%360)
    require(type(rec['requested_azimuth_deg']) in (float,int) and abs(circular_error(planned,rec['requested_azimuth_deg']))<1e-9,'requested direction azimuth')
    require(type(rec['actual_azimuth_deg']) in (float,int) and type(rec['azimuth_deg']) in (float,int) and
        math.isfinite(rec['actual_azimuth_deg']) and rec['azimuth_deg']==rec['actual_azimuth_deg'] and
        abs(circular_error(angle,rec['actual_azimuth_deg']))<1e-4,'FINAL actual angle identity1e-4 and selection alias')
    require(np.allclose(rec['azimuth_reference_world_m'],observation['azimuth_reference_world_m'],atol=1e-12,rtol=0),'fixed native framing reference')
    require(strict_equal(rec['reference_center_world_m'],rec['azimuth_reference_world_m']) and
        rec['angle_source']=='FINAL_ACTUAL_USD_MATRIX_FIXED_NATIVE_FRAMING_CENTER' and
        rec['original_reader_identity_tolerance_deg']==1e-4 and rec['minimum_pairwise_separation_deg']==30.,'fixed angle aliases/protocol')
    return result,angle,planned


def single_counts(outputs,info,inventory,context,palms):
    result=o.mask_counts(outputs,info,inventory,context,palms,2)
    objects,ids,maps,parts,errors=o.raw_maps(outputs,info,inventory,context,palms,2)
    foreign_hand=False;foreign_context=0
    for name,mapping in maps.items():
        for key,path in mapping.items():
            if path is None or objects[path]['env_id']==2:continue
            n=int(np.count_nonzero(ids[name]==key))
            if objects[path].get('hand'):foreign_hand|=n>0
            else:foreign_context+=n
    # This prospective collector leaves all environments visible. Its hand gate
    # forbids foreign HAND pixels; other context is counted and disclosed.
    result['errors']=[x for x in result['errors'] if x!='foreign_native_pixels']
    if foreign_hand:result['errors'].append('foreign_hand_pixels')
    result['foreign_context_stream_pixels']=foreign_context
    return result


def coverage_agreement(rec,counts,rows,audit):
    value=rec['coverage_v6'];require(value==parse_json(audit.ref(rec['anatomical_mask_evidence'])),'saved anatomy metadata mismatch')
    expected={p:dict(body_name=m['body_name'],body_index=m['body_index'],pixels=counts['per_native_object_pixels'][p]) for p,m in rows.items()}
    require(strict_equal(value['per_native_body'],expected) and strict_equal(value['exact_same_rigid_hand_pixels'],sum(x['pixels'] for x in expected.values())) and
        strict_equal(value['palm_visual_pixels'],counts['palm_by_arm']['F_R']) and strict_equal(value['finger_family_pixels'],counts['family_by_arm']['F_R']),'typed raw anatomy counts disagree')
    saved=parse_json(audit.ref(rec['mask_evidence']))
    require({k:v for k,v in saved.items() if k!='streams'}==rec['mask'],'saved base mask metadata mismatch')
    total=sum(x['pixels'] for x in expected.values())
    require(strict_equal(rec['mask']['observed_hand_pixels'],total),'base raw hand pixel count')
    require(value['base_mask_status']==rec['mask']['status'],'base mask/anatomy status disagreement')


def selected_group(records,selected,inventory):
    require(isinstance(selected,list) and len(selected)==3 and all(type(x) is int for x in selected) and len(set(selected))==3,'three distinct selected attempts')
    byid={r['attempt']:r for r in records};require(len(byid)==len(records) and all(x in byid for x in selected),'selected original attempt missing')
    trio=[byid[x] for x in selected]
    require(all(r['status']=='qualified' and '_counts' in r for r in trio),'selected attempt lacks qualified raw pixels')
    angles=[r['_independent_azimuth'] for r in trio]
    separated=all(abs((a-b+180)%360-180)>=30 for a,b in itertools.combinations(angles,2))
    result=o.hand_group(trio,inventory,2,'F_R')
    result['qualified']=bool(result['qualified'] and result['all_native_hand_bodies_observed'] and separated)
    result['pairwise_actual_azimuths_separated']=separated
    result['actual_azimuths_degrees']=angles
    result['angle_reference']='FINAL actual eye minus fixed native framing-points AABB center'
    return result


def audit_camera(audit,receipt,reference,asset,output):
    refs=bounded_case(receipt);baseline=audit.snapshot(receipt['native_before'])
    h.exact_snapshot(reference,baseline);h.exact_snapshot(baseline,audit.snapshot(receipt['native_after']))
    inventory=audit.load_inventory(receipt['native_path_mapping']);setup=parse_json(audit.ref(receipt['semantic_setup']))
    context,palms=o.validate_palms_and_context(setup,inventory,asset)
    points,rows,centers,center=framing_points(parse_json(audit.ref(receipt['native_body_bounds'])),baseline,inventory)
    expected=[dict(m,rigid_path=p,target_center_world_m=centers[p].tolist()) for p,m in rows.items()]
    actual=receipt['hand_native_paths'];require(len(actual)==len(expected),'hand path denominator')
    for got,want in zip(actual,expected):
        require(set(got)==set(want) and all(got[k]==v for k,v in want.items() if k!='target_center_world_m') and
                np.allclose(got['target_center_world_m'],want['target_center_world_m'],atol=1e-12,rtol=0),'hand native rows/targets')
    reports=[];records=[];pngs=set();errors=[]
    for number,ref in enumerate(refs,1):
        rec=parse_json(audit.ref(ref));problems=[];decoded=False
        require(rec['attempt']==number and rec['arm']=='F_R' and rec['status'] in ('qualified','rejected'),'attempt identity/status')
        try:
            outputs,info=o.native_and_pixels(audit,rec,baseline)
            if 'camera' in rec:
                framing,angle,planned=camera_pose(rec,points,centers,center)
                rec['_framing_pass']=framing['pass_gate'];rec['_independent_azimuth']=angle
                rec['_fixed_center_azimuth']=framing['azimuth_deg'];rec['_planned_azimuth']=planned
            if outputs is not None:
                decoded=True;rec['_counts']=single_counts(outputs,info,inventory,context,palms)
                coverage_agreement(rec,rec['_counts'],rows,audit)
            require(rec['status']!='qualified' or (decoded and rec.get('_framing_pass') and not rec['_counts']['errors']),'qualified attempt fails raw gate')
        except Exception as exc:problems.append(type(exc).__name__+': '+str(exc))
        if 'rgb' in rec:
            require(rec['rgb']['path'] not in pngs,'duplicate PNG');pngs.add(rec['rgb']['path'])
        progress=parse_json(audit.ref(dict(path=f'camera_repair_env002_F_R/progress_{number:03d}.json',
            sha256=audit.evidence.hash(audit.root/f'camera_repair_env002_F_R/progress_{number:03d}.json'))))
        require(progress==dict(attempt=number,status=rec['status'],reasons=rec['reasons']),'durable progress mismatch')
        report=dict(attempt=number,producer_status=rec['status'],raw_pixels_decoded=decoded,counts=rec.get('_counts'),
            actual_fixed_center_azimuth=rec.get('_independent_azimuth'),requested_azimuth=rec.get('_planned_azimuth'),
            integrity_errors=problems,acceptance=False)
        write_new(output/'attempts'/f'{number:03d}.json',report);reports.append(report);records.append(rec)
        errors.extend(f'attempt{number}: '+x for x in problems)
    group=selected_group(records,receipt['selected_attempts'],inventory)
    declared=receipt['strict_group']
    try:group_metadata(declared,group)
    except ValueError as exc:errors.append(str(exc))
    folder=contained(audit.root/'camera_repair_env002_F_R',audit.root)
    require({r['path'] for r in refs}=={str(p.relative_to(audit.root)) for p in folder.glob('*.attempt.json')},'unlisted original attempt')
    require(pngs=={str(p.relative_to(audit.root)) for p in folder.glob('*.png')},'unlisted original PNG')
    search=receipt['camera_search'];require(search['attempt_cap']==96 and search['attempts_used']==len(refs) and
        search['physical_occlusion_impossible_proven'] is False and search['finite_view_search_never_proves_universal_visibility'] is True,'bounded search metadata')
    return dict(scope='ENV2_F_R_INITIAL_ZERO_STEP_ONLY',attempts=reports,selected_attempts=receipt['selected_attempts'],strict_group=group,
        all_attempts_visited=len(records),raw_pixels_decoded=sum(r['raw_pixels_decoded'] for r in reports),original_png_count=len(pngs),
        held_snapshot_checks=audit.snapshots,held_native_fields=27,integrity_errors=errors,
        independent_observation_sufficient=group['qualified'] and not errors,view_reselection=False,
        full32_qualified=False,prefix_qualified=False,wide_coverage='NOT_CAPTURED',final_capture='NOT_CAPTURED',
        native_pass=False,stage_pass=False,safety_acceptance=False,full_mesh_coverage_certified=False)
