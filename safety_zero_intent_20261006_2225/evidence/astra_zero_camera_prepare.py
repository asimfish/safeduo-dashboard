"""All-group own-state/math audit and fixed image queue. Does NOT inspect vision.

The queued original PNG files require subsequent actual view_image tool calls.
No image is edited, generated, resized, composited or written by this program.
"""
import argparse
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from astra_zero_camera_core import ARMS, VIEWS, PLANES, compact, exact, frustum_planes, native_state_binding, select_groups, png_header
from astra_zero_raw_score import H, Ledger, float_array, now, relative_file, sha, verify_fifo
from astra_zero_score_core import evaluation_classes, fingerprint, native_flags, reduce_geometry, require, score_windows, aggregate


def write(name, value):
    require(Path(name).name == name and name.startswith('astra_zero_camera_'), 'unowned camera output')
    target = H/name
    temporary = H/('astra_zero_camera_tmp_'+name)
    temporary.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def closure():
    path = H/'visual_execution.json'
    if not path.exists():
        return dict(status='WAITING_CAMERA_EXECUTION_NOT_STARTED', terminal=False)
    record = json.loads(path.read_text())
    return dict(status=record['status'], terminal=record['status'] in ('complete', 'complete_with_failures', 'failed'),
        jobs=[{k:j.get(k) for k in ('mode','status','exit_code')} for j in record['jobs']])


def registered(ledger):
    own = ledger.json(H/'astra_zero_camera_plan.json')
    for name, digest in own['helper_source_sha256'].items():
        ledger.bind(H/name, digest)
    plan = ledger.json(H/'visual_plan.json', own['visual_plan_sha256'])
    selection = ledger.json(H/'astra_zero_camera_selection.json', own['selection_sha256'])
    require(selection['status'] == 'PRESELECTED_BEFORE_ANY_CAMERA_PIXELS', 'selection not preregistered')
    require(selection['modes'] == ['joint_reference','zero_inclusive'], 'selected mode set changed')
    require(selection['scheduled_groups'] == [{'env':0,'step':75},{'env':16,'step':480},{'env':40,'step':959},{'env':56,'step':480}], 'selected scheduled cases changed')
    return plan, selection


def all_source_bindings(ledger, plan):
    for name, digest in plan['sources'].items():
        ledger.bind(name, digest)
    for name, digest in plan['original_source_sha256'].items():
        ledger.bind(Path(plan['cwd'])/name, digest)
    for job in plan['jobs']:
        ledger.bind(job['argv'][job['argv'].index('--ckpt')+1], plan['actor_sha256'])


def inspect_group(ledger, root, cap, dense, labels, classes, arm_ids, mode, dt):
    state_path = relative_file(root, cap['state'])
    s = ledger.json(state_path, cap['sha256'])
    e, t = cap['env_id'], cap['step']
    require((s['env_id'],s['step'],s['capture_kind'],s['visual_mode']) == (e,t,cap['capture_kind'],mode), 'capture identity differs')
    require(s['environment_count'] == 64 and s['risk_stratum'] == int(labels[e]) and s['state_time_s'] == (t+1)*dt, 'capture state time/environment/stratum')
    require(s['first_negative_capture'] == (cap['capture_kind'] == 'first_failure'), 'capture kind marker')
    require(set(s['views']) == set(VIEWS) and len(cap['images']) == 9 and s['images'] == cap['images'], 'nine-view inventory')
    before_path = relative_file(root, s['fresh_native_snapshot'])
    after_path = relative_file(root, s['fresh_native_after_snapshot'])
    before = ledger.npz(before_path, expected=s['fresh_native_before_sha256'])
    after = ledger.npz(after_path, expected=s['fresh_native_after_sha256'])
    binding = native_state_binding(s, before, after, dense)
    centers = np.asarray(s['sphere_centers_world_m'], np.float32)
    radii = np.asarray(s['sphere_radii_m'], np.float32)
    require(centers.shape == (142,3) and radii.shape == (142,) and (radii >= 0).all(), 'sphere archive shape')
    exact(centers, dense['sphere_centers'][t,e], 'own dense sphere centers')
    d = np.asarray(s['full_distances_m'], np.float32)
    exempt = np.asarray(s['full_exempt'])
    require(d.shape == (9021,) and exempt.shape == d.shape and exempt.dtype == np.bool_, 'full raw row shape/type')
    for name in ('full_braking_dmin_m','full_closing_m_s'):
        float_array(np.asarray(s[name],np.float32), (9021,), name)
    margins = reduce_geometry(d,exempt,classes)
    exact(margins, dense['official_margins'][t,e], 'own post geometry/class minima')
    view_records = {}
    images = {Path(item['path']).stem.removeprefix(f'step_{t:04d}_'): item for item in cap['images']}
    require(set(images) == set(VIEWS) and len(images) == len(cap['images']), 'view image names/duplicates')
    for name in VIEWS:
        view=s['views'][name]
        W=np.asarray(view['actual_camera_to_world_row_matrix'],float)
        K=np.asarray(view['actual_intrinsic_matrix'],float)
        require(W.shape==(4,4) and K.shape==(3,3) and np.isfinite(W).all() and np.isfinite(K).all(), 'nonfinite camera matrix')
        require(np.allclose(W[3,:3],view['actual_position_world_m'],rtol=0,atol=1e-8), 'actual USD position differs')
        require(np.allclose(W[:3,:3]@W[:3,:3].T,np.eye(3),rtol=0,atol=1e-6), 'nonrigid actual camera transform')
        require(np.allclose(W[:,3],[0,0,0,1],rtol=0,atol=1e-8), 'nonaffine USD transform')
        require(np.allclose(W[3,:3],np.asarray(view['eye'])+np.asarray(s['origin_world_m']),rtol=0,atol=1e-5), 'requested/actual camera position mismatch')
        optics=view['actual_usd_optics']
        require(optics['horizontal_aperture_offset']==0 and optics['vertical_aperture_offset']==0, 'unmodeled aperture offsets')
        require(optics['focal_length']>0 and optics['horizontal_aperture']>0 and optics['vertical_aperture']>0, 'invalid optics')
        expected_K=np.diag([1280*optics['focal_length']/optics['horizontal_aperture'],720*optics['focal_length']/optics['vertical_aperture'],1.])
        expected_K[:2,2]=[640,360]
        require(np.allclose(K,expected_K,rtol=1e-5,atol=2e-4), 'USD optics/intrinsics mismatch')
        mask=arm_ids<2 if name.startswith('f_') else arm_ids>=2 if name.startswith('u_') else np.ones(len(arm_ids),bool)
        plane=frustum_planes(centers[mask],radii[mask],W,K,view['actual_clipping_range_m'])
        require((plane>0).all(), 'represented target spheres cropped by actual frustum: '+name)
        claim=view['observed_sphere_frustum']
        require(claim['spheres']==int(mask.sum()) and tuple(claim['plane_names'])==PLANES, 'frustum identity/order')
        require(np.allclose(plane.min(0),claim['minimum_by_plane_m'],rtol=0,atol=1e-7), 'frustum plane readback mismatch')
        require(abs(float(plane.min())-claim['minimum_plane_margin_m'])<=1e-7, 'frustum global minimum mismatch')
        path=relative_file(root,images[name]['path'])
        payload=ledger.bytes(path,images[name]['sha256'])
        header=png_header(payload)
        view_records[name]=dict(path=str(path),sha256=ledger.hashes[str(path)],
            png_header=header,
            actual_camera_position_world_m=W[3,:3].tolist(),minimum_by_plane_m=plane.min(0).tolist(),
            minimum_sphere_plane_margin_m=float(plane.min()),sphere_count=int(mask.sum()),
            pixel_inspection_performed=False)
    return dict(group_id=f'{mode}:{cap["capture_kind"]}:e{e}:t{t}',mode=mode,env=e,step=t,kind=cap['capture_kind'],
        state_path=str(state_path),state_sha256=ledger.hashes[str(state_path)],native_before_path=str(before_path),
        native_after_path=str(after_path),native_binding=binding,
        own_margin_by_class_m=[float(x) if np.isfinite(x) else None for x in margins],
        native_strict=bool((margins<np.float32(0)).any()),native_deep=bool((margins<np.float32(-.005)).any()),
        views=view_records,pixel_inspection_performed=False)


def audit_mode(ledger, plan, selection, record, mode):
    job=next(j for j in plan['jobs'] if j['mode']==mode)
    root=Path(plan['actual_visual_root'])/mode
    require(record['status']=='complete' and record['exit_code']==0 and Path(record['out'])==root, 'camera job incomplete/wrong root')
    require(record['argv']==job['argv'], 'camera actual argv differs from registered job')
    proto=ledger.json(root/'visual_protocol.json',record.get('protocol_sha256'))
    require(proto['status']=='complete' and proto['completed_windows']==64 and proto['steps']==960 and proto['dt']==.016666, 'camera protocol incomplete/timebase')
    require(proto['actor_sha256']==plan['actor_sha256'] and proto['observer_sha256']==plan['sources'][str(H/'visual_runner.py')], 'actor/observer mismatch')
    for arg,key in [('--seeds','seeds'),('--device','device'),('--env-yaml','env_yaml'),('--duration-s','duration_s'),('--methods','methods')]:
        actual=proto['args'][key];expected=job['argv'][job['argv'].index(arg)+1]
        require(actual==type(actual)(expected),'camera argument mismatch: '+arg)
    meta=ledger.json(root/'guard_metadata.json')
    require(meta['reference_zero_inclusive'] is (mode=='zero_inclusive'),'camera zero-inclusive flag differs')
    require(meta['mode']==mode and meta['capacity']==9021 and meta['strict_fifo_steps']==6 and meta['reference_gap_rad']=={'joint_reference':.050,'zero_inclusive':.010}[mode] and not meta['joint_repair'] and not meta['queue_preemption'], 'camera control contract')
    for path,digest in meta['sources'].items():
        require(plan['sources'].get(path)==digest,'camera guard source binding')
    keys=['q','q_initial','pre_qd_compact','cmd','exec','official_margins','official_deep','ee','sphere_centers',
          'controller_target','actuator_target','pre_pending_actuator_targets','pre_pending_project_history',
          'pre_target_debt','effective_target_delta','joint_soft_limits','external_unscaled_cmd']
    dense=ledger.npz(root/'cell_001.npz',keys)
    for k in ('q','pre_qd_compact','cmd','exec','controller_target','actuator_target','pre_target_debt','effective_target_delta','external_unscaled_cmd'):
        float_array(dense[k],(960,64,26),k)
    float_array(dense['q_initial'],(64,26),'q_initial')
    float_array(dense['sphere_centers'],(960,64,142,3),'sphere_centers')
    float_array(dense['official_margins'],(960,64,4),'official_margins',True)
    for k in ('pre_pending_actuator_targets','pre_pending_project_history'):float_array(dense[k],(960,6,64,26),k)
    float_array(dense['joint_soft_limits'],(64,26,2),'joint_soft_limits')
    verify_fifo(dense)
    strict,deep=native_flags(dense['official_margins'])
    exact(dense['official_deep'],deep,'native visual deep flags')
    labels=ledger.npz(root/'bank_assignment.npz',['risk_pair_index'])['risk_pair_index']
    bank_path=Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])
    bank=ledger.npz(bank_path,['accepted_q','risk_pair_index'],plan['sources'][str(bank_path)])
    recipe=ledger.npz(root/'input_recipe.npz',['q_initial','sampled_initial','tape'])
    exact(labels,bank['risk_pair_index'],'camera registered risk labels')
    exact(recipe['sampled_initial'],bank['accepted_q'],'camera registered sampled q')
    exact(recipe['q_initial'],dense['q_initial'],'camera recipe own q0')
    float_array(recipe['tape'],(962,64,26),'camera full command tape')
    exact(recipe['tape'][:60],np.zeros((60,64,26),np.float32),'camera full zero prefix')
    exact(recipe['tape'][:960],dense['cmd'],'camera actual command tape')
    exact(recipe['tape'][:960],dense['external_unscaled_cmd'],'camera external raw tape')
    identity=ledger.json(root/'full_row_identity.json')
    classes=evaluation_classes(identity);arms=np.asarray(identity['sphere_arm_id'])
    require(len(classes)==9021 and len(arms)==142,'camera geometry identity')
    receipt=ledger.json(root/'camera_receipts.json');caps=receipt['receipts']
    identities=[(c['capture_kind'],c['env_id'],c['step']) for c in caps]
    require(len(set(identities))==len(caps),'duplicated camera group')
    scheduled={(c['env_id'],c['step']) for c in caps if c['capture_kind']=='scheduled'}
    require(scheduled=={(e,t) for e in (0,8,16,24,32,40,56) for t in (75,480,959)},'scheduled group inventory')
    expected_failures=set()
    bad=strict.any(-1)
    for label in (-1,0,1,2,3,4,5):
        hits=np.argwhere(bad & (labels[None,:]==label))
        if len(hits):expected_failures.add((int(hits[0,1]),int(hits[0,0])))
    observed_failures={(c['env_id'],c['step']) for c in caps if c['capture_kind']=='first_failure'}
    require(observed_failures==expected_failures,'own-state earliest stratum failure capture')
    require(len(caps)==21+len(expected_failures) and receipt['groups']==len(caps) and receipt['PNG']==9*len(caps)
            and receipt['scheduled_groups']==21 and receipt['first_failure_groups']==len(expected_failures),'camera denominator')
    expected_pngs={relative_file(root,image['path']).resolve() for cap in caps for image in cap['images']}
    actual_pngs={p.resolve() for p in root.rglob('*.png')}
    require(actual_pngs==expected_pngs,'unreceipted or missing PNG in closed camera tree')
    require(len(expected_pngs)==9*len(caps),'PNG path reused by camera groups')
    expected_states={relative_file(root,c['state']).resolve() for c in caps}
    require({p.resolve() for p in (root/'multiview').rglob('*_state.json')}==expected_states,'unreceipted or missing state group')
    chosen=select_groups(selection,caps,mode)
    selected_ids={(c['capture_kind'],c['env_id'],c['step']) for c in chosen}
    groups=[]
    for cap in caps:
        group=inspect_group(ledger,root,cap,dense,labels,classes,arms,mode,proto['dt'])
        group['selected_for_actual_view_image']=(group['kind'],group['env'],group['step']) in selected_ids
        groups.append(group)
    design=ledger.json(H/'RANDOM_EXPERIMENT_DESIGN.json')
    numeric_root=root.parent.parent/'holdout_0'/f'{mode}_{design["rows"][0]["command_seed"]}'
    numeric_keys=['q','q_initial','ee','cmd','exec','official_margins','controller_target','actuator_target','pre_qd_compact','pre_pending_actuator_targets']
    numeric_plan=ledger.json(H/'plans'/'holdout_0_plan.json')
    from astra_zero_raw_score import canonical_closed
    numeric_records=canonical_closed(ledger,numeric_plan)
    require(numeric_records is not None,'numeric block0 not closed for camera forward binding')
    nr=numeric_records[numeric_root.name]
    require(nr['status']=='complete' and nr['exit_code']==0,'numeric forward source condition incomplete')
    ledger.json(numeric_root/'protocol.json',nr['protocol_sha256'])
    numeric=ledger.npz(numeric_root/'cell_001.npz',numeric_keys)
    numeric_recipe=ledger.npz(numeric_root/'input_recipe.npz',['tape'])
    forward={k:bool(fingerprint(dense[k])==fingerprint(numeric[k])) for k in numeric_keys}
    forward['full_tape']=bool(fingerprint(recipe['tape'])==fingerprint(numeric_recipe['tape']))
    qdiff=(dense['q']!=numeric['q']).any((1,2));changed=np.flatnonzero(qdiff)
    metrics=score_windows(dense['official_margins'])
    return dict(mode=mode,groups=groups,own_visual_counts=aggregate([dict(status='VERIFIED',metrics=m) for m in metrics]),
        actual_group_count=len(groups),actual_image_count=9*len(groups),selected_group_count=len(chosen),
        ownstate_math_status='PASS_ALL_GROUPS',forward_status='PASS_EXACT_NUMERIC_CAMERA_FORWARD' if all(forward.values()) else 'FAIL_EXACT_NUMERIC_CAMERA_FORWARD',
        forward_field_equality=forward,first_q_difference_step=int(changed[0]) if len(changed) else None,
        maximum_q_difference_rad=float(np.abs(dense['q'].astype(float)-numeric['q']).max()),
        camera_pixel_inspection_performed=False,
        forward_scope='Exact dtype/shape/byte identity of declared saved arrays only; numerical full rigid/root state is not archived every frame.',
        parent_camera_job_retains_protocol_sha256='protocol_sha256' in record)


def run(ledger,plan,selection):
    execution=ledger.json(H/'visual_execution.json')
    require(execution['status']=='complete' and execution['plan_sha256']==ledger.hashes[str(H/'visual_plan.json')], 'camera execution not complete/bound')
    require(len(execution['jobs'])==2 and {j['mode'] for j in execution['jobs']}==set(selection['modes']),'camera job inventory')
    all_source_bindings(ledger,plan)
    results=[]
    for mode in selection['modes']:
        record=next(j for j in execution['jobs'] if j['mode']==mode)
        results.append(audit_mode(ledger,plan,selection,record,mode))
        write('astra_zero_camera_progress.json',dict(status='MATH_AUDIT_RUNNING',utc=now(),completed_modes=[r['mode'] for r in results],actual_images_viewed=0))
    changed=ledger.recheck()
    require(not changed,'consumed camera inputs changed: '+str(changed))
    selected=[g for r in results for g in r['groups'] if g['selected_for_actual_view_image']]
    queue=[dict(group_id=g['group_id'],mode=g['mode'],env=g['env'],step=g['step'],kind=g['kind'],view=name,
                image_path=g['views'][name]['path'],image_sha256=g['views'][name]['sha256'],
                state_path=g['state_path'],native_before_path=g['native_before_path'],native_after_path=g['native_after_path'])
           for g in selected for name in VIEWS]
    require(len(selected)==8 and len(queue)==72,'fixed eight-group/72-original selection incomplete')
    duplicates={}
    for r in results:
        for g in r['groups']:
            for name,v in g['views'].items():duplicates.setdefault(v['sha256'],[]).append(dict(group_id=g['group_id'],view=name,path=v['path']))
    duplicate_sets=[entries for entries in duplicates.values() if len(entries)>1]
    result=dict(status='PASS_ALL_GROUP_MATH_NATIVE_STATE_ONLY_PIXELS_PENDING',utc=now(),runs=results,
        all_group_count=sum(r['actual_group_count'] for r in results),all_image_hashes_and_headers_checked=sum(r['actual_image_count'] for r in results),
        preselected_groups=len(selected),preselected_images=len(queue),actual_images_viewed=0,pixel_inspection_performed=False,
        input_sha256=ledger.hashes,raw_hash_before_after_pass=True,
        exact_image_duplicate_sets=duplicate_sets,duplicate_hashes_are_candidates_for_review_not_automatic_staleness=True,
        scope='all-group metadata mathematics/native state; PNG headers and hashes only; no visual sufficiency verdict',
        no_physical_safety_claim=True,no_occlusion_or_mesh_silhouette_certificate=True)
    write('astra_zero_camera_metadata.json',result)
    write('astra_zero_camera_image_selection.json',dict(status='READY_FOR_ACTUAL_VIEW_IMAGE_REVIEW',utc=now(),
        selection_sha256=ledger.hashes[str(H/'astra_zero_camera_selection.json')],groups=selected,view_queue=queue,
        image_count=len(queue),actual_images_viewed=0,metadata_evidence='astra_zero_camera_metadata.json',
        required_tool='view_image on each original PNG; no edits/composites/generation'))
    print(json.dumps({k:result[k] for k in ('status','all_group_count','preselected_groups','preselected_images','raw_hash_before_after_pass')}),flush=True)


def main():
    # One-shot only; no reviewer watcher or simulation is started.
    parser=argparse.ArgumentParser();parser.add_argument('--audit-closed',action='store_true')
    args=parser.parse_args()
    ledger=Ledger();plan,selection=registered(ledger)
    if not args.audit_closed:
        print('PASS_CAMERA_REGISTRATION_ONLY_NO_OUTCOME_READS',flush=True);return
    state=closure();write('astra_zero_camera_readiness.json',dict(utc=now(),**state,pixel_inspection_performed=False))
    if not state['terminal']:
        print(json.dumps(state));return
    require(not (H/'astra_zero_camera_metadata.json').exists(),'preserve previous camera audit; no automatic overwrite/retry')
    try:run(ledger,plan,selection)
    except Exception as error:
        write('astra_zero_camera_metadata_error.json',dict(status='FAIL_OR_UNAVAILABLE_CAMERA_MATH_INPUT',utc=now(),
            error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),input_sha256=ledger.hashes,
            actual_images_viewed=0,pixel_inspection_performed=False))
        raise


if __name__=='__main__':main()
