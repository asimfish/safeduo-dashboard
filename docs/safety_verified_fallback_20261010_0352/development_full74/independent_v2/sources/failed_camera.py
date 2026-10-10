"""Failure-aware raw camera diagnostics; no producer group is promoted to acceptance."""
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
from evidence_io import ARMS,H,contained,require,parse_json,exact,write_new,sha
import camera_adapter as h
import camera_oracle as o
from camera_index import read_index,metadata_digest

SOURCES={'source_sha256':H/'astra/next_mechanism/qualified_views_v7.py',
 'inherited_hand_views_sha256':H/'astra/next_mechanism/hand_views_v3.py',
 'frozen_native_hold_source_sha256':H/'astra/adaptation32/hand_views_v2.py'}


def error_text(exc):return type(exc).__name__+': '+str(exc)


def pixels_without_hold(audit,rec):
    """Continue raw pixel diagnosis after a held-state failure, retaining that failure."""
    outputs=audit.npz(rec['original_outputs']);info=parse_json(audit.ref(rec['original_info']));rgb=outputs['rgb']
    require(rgb.dtype==np.uint8 and rgb.shape in ((720,1280,3),(720,1280,4)),'original RGB layout')
    require(hashlib.sha256(rgb.tobytes()).hexdigest()==rec['raw_rgb_sha256'],'raw RGB digest')
    with Image.open(io.BytesIO(audit.ref(rec['rgb']))) as im:
        require(im.format=='PNG' and im.size==(1280,720),'original PNG layout')
        exact(np.asarray(im),rgb,'original RGB/PNG')
    return outputs,info


def wide_metrics(records,inventory,slot):
    require(records,'wide evidence empty')
    rows=[p for p,m in inventory.items() if m['env_id']==slot]+[f'/World/envs/env_{slot}/{t}' for t in ('TableF','TableU')]
    peaks={p:max(r['_counts']['per_native_object_pixels'].get(p,0) for r in records) for p in rows}
    arms={a:max(sum(r['_counts']['per_native_object_pixels'].get(p,0) for p,m in inventory.items()
                     if m['env_id']==slot and m['arm']==a) for r in records) for a in ARMS}
    tables={t:peaks[f'/World/envs/env_{slot}/{t}'] for t in ('TableF','TableU')}
    critical={p:n for p,n in peaks.items() if p in inventory and inventory[p]['body_name'] in ('forearm_link','wrist_2_link')}
    require(len(critical)==4,'both U critical body denominator')
    ok=all(n>=1024 for n in [*arms.values(),*tables.values()]) and all(n>=128 for n in critical.values())
    ok=ok and all(r['_framing_pass'] and not r['_counts']['errors'] for r in records)
    return dict(qualified=bool(ok),per_native_object_peak_pixels=peaks,arm_peak_pixels=arms,table_peak_pixels=tables,
        critical_body_peak_pixels=critical,critical_body_visibility_gaps=[p for p,n in critical.items() if n<128])


def selected_metrics(declared,attempts,compute):
    """No replacement views selected. Insufficient declared trio is diagnostic data."""
    refs=declared.get('selected_group_images',declared.get('selected_images',[]))
    errors=[];metric=None
    if refs:
        try:metric=compute(o.selected_three(refs,attempts))
        except Exception as exc:errors.append(error_text(exc))
    group=declared.get('group_coverage')
    if metric is not None and group is not None:
        for k,v in metric.items():
            if k!='qualified' and k in group and group[k]!=v:errors.append('producer coverage disagreement: '+k)
        if group.get('status') not in ('qualified','failed'):errors.append('producer group status missing/invalid')
        elif (group['status']=='qualified')!=metric['qualified']:errors.append('producer coverage status disagreement')
    elif group is not None:errors.append('coverage metadata lacks verifiable selected trio')
    sufficient=bool(metric and metric['qualified'])
    # A producer failure is preserved even if its selected pixel predicate is sufficient.
    if declared.get('status')=='qualified':
        if not sufficient:errors.append('declared qualified but selected raw coverage insufficient')
        if declared.get('qualified_images')!=refs:errors.append('qualified/selected image references disagree')
    elif declared.get('status')=='failed':
        if declared.get('qualified_images',[])!=[]:errors.append('failed group carries qualified image references')
    else:errors.append('unknown producer status')
    return dict(producer_status=declared.get('status'),producer_failure_reason=declared.get('failure_reason'),
        selected_view_count=len(refs),selected_coverage=metric,selected_raw_predicate_sufficient=sufficient,
        integrity_errors=errors,acceptance=False)


def inspect_attempt(audit,embedded,baseline,points,inventory,context,palms,slot,kind):
    rec={k:v for k,v in embedded.items() if k!='attempt_record'};errors=[];outputs=info=None;held_verified=False
    def check(fn):
        try:return fn()
        except Exception as exc:errors.append(error_text(exc));return None
    saved=check(lambda:parse_json(audit.ref(embedded['attempt_record'])))
    if saved is not None and saved!=rec:errors.append('embedded raw attempt differs from saved bytes')
    try:
        outputs,info=o.native_and_pixels(audit,rec,baseline);held_verified=True
    except Exception as exc:
        errors.append(error_text(exc))
        if 'original_outputs' in rec:
            try:outputs,info=pixels_without_hold(audit,rec)
            except Exception as pixel_error:errors.append(error_text(pixel_error))
    rec['_framing_pass']=False
    if outputs is not None:
        try:
            if kind=='hand':require(np.array_equal(points,rec['hand_link_positions_world_m']),'hand framing not actual native link positions')
            framing=h.frustum(points,rec['camera']);rec['_framing_pass']=bool(framing['pass_gate']);rec['_independent_azimuth']=framing['azimuth_deg']
            require(abs((framing['azimuth_deg']-rec['azimuth_deg']+180)%360-180)<1e-4,'azimuth differs from actual camera matrix')
        except Exception as exc:errors.append(error_text(exc))
        try:rec['_counts']=o.mask_counts(outputs,info,inventory,context,palms,slot)
        except Exception as exc:errors.append(error_text(exc))
    elif 'rgb' in rec:errors.append('RGB reference without decoded original outputs')
    return rec,dict(attempt=rec.get('attempt'),arm=rec.get('arm'),kind=kind,producer_status=rec.get('status'),
        producer_reasons=rec.get('reasons'),attempt_record=embedded.get('attempt_record'),rgb=rec.get('rgb'),
        raw_pixels_decoded=outputs is not None,held27_verified=held_verified,
        framing_sufficient=rec['_framing_pass'],mask_counts=rec.get('_counts'),integrity_errors=errors,acceptance=False)


def inspect_group(audit,receipt,fields,asset,origins):
    slot=receipt['env_id'];state_ref={'path':receipt['state'],'sha256':receipt['sha256']}
    state=parse_json(audit.ref(state_ref));errors=[];disagreements=[]
    require(state['schema']=='safeduo.qualified_views.v7' and state['env_id']==slot and state['status'] in ('qualified','failed'),'initial group identity/status')
    require(state['environment_count']==32 and state['all_registered_lanes'] is True and state['all64'] is False and
            state['physics_advanced_by_collector'] is False,'initial camera native scope')
    require(receipt['_metadata_sha256']==metadata_digest(state),'camera receipt/state metadata digest disagreement')
    for key,path in SOURCES.items():require(state[key]==audit.evidence.hash(path),'camera source SHA '+key)
    baseline=audit.snapshot(state['native_before_all64']);h.exact_snapshot(baseline,audit.snapshot(state['native_final_all64']))
    if origins is not None:exact(origins,baseline['environment_origins'],'all32 common held camera origins')
    native=h.micro_native(fields,0,audit.layout,baseline['environment_origins']);native.pop('environment_origins')
    o.bind_native_event(state,baseline,native,('initial',slot,-1,None))
    inventory=audit.load_inventory(state['native_path_mapping']);setup=parse_json(audit.ref(state['semantic_setup']))
    context,palms=o.validate_palms_and_context(setup,inventory,asset)
    refs=set();pngrefs=set();hand=[];wide=[];attempt_reports=[]
    def visit(embedded,kind,points):
        ref=embedded['attempt_record']['path'];require(ref not in refs,'duplicate attempt reference');refs.add(ref)
        rec,r=inspect_attempt(audit,embedded,baseline,points,inventory,context,palms,slot,kind)
        if 'rgb' in rec:
            path=rec['rgb']['path'];require(path not in pngrefs,'duplicate original RGB reference');pngrefs.add(path)
        attempt_reports.append(r);errors.extend(f'{kind}/{rec.get("arm")}/{rec.get("attempt")}: {x}' for x in r['integrity_errors'])
        return rec
    for embedded in state['attempts']:
        try:
            arm=embedded['arm'];require(arm in ARMS,'unknown camera arm')
            rows=[m for m in inventory.values() if m['env_id']==slot and m['arm']==arm and m['hand']]
            points=baseline[arm+'_native_link_transforms_xyzw'][slot,[m['body_index'] for m in rows],:3].astype(float)
            hand.append(visit(embedded,'hand',points))
        except Exception as exc:errors.append(error_text(exc))
    arms={}
    for arm in ARMS:
        subset=[x for x in hand if x['arm']==arm];declared=state['arms'][arm]
        if [r['attempt'] for r in subset]!=list(range(1,len(subset)+1)) or len(subset)>24:errors.append('hand attempt sequence '+arm)
        if declared.get('attempts',[])!=[r['attempt_record'] for r in state['attempts'] if r['arm']==arm]:errors.append('hand attempt reference inventory '+arm)
        compute=lambda records,arm=arm:o.hand_group(records,inventory,slot,arm)
        result=selected_metrics(declared,subset,compute);errors.extend(arm+': '+x for x in result['integrity_errors'])
        valid=[r for r in subset if '_counts' in r]
        result['all_attempt_peak_coverage']=compute(valid) if valid else None
        result['all_attempt_peaks_are_not_selected_views']=True;arms[arm]=result
    wide_declared=state.get('wide_context')
    if wide_declared is not None:
        try:points=o.wide_points_from_bound_record(parse_json(audit.ref(wide_declared['authored_bounds'])),baseline,inventory,slot,state['actual_table_aabbs'])
        except Exception as exc:points=None;errors.append(error_text(exc))
        for embedded in wide_declared['attempts']:
            try:wide.append(visit(embedded,'wide',points))
            except Exception as exc:errors.append(error_text(exc))
        if [r['attempt'] for r in wide]!=list(range(1,len(wide)+1)) or len(wide)>6:errors.append('wide attempt sequence')
        wide_result=selected_metrics(wide_declared,wide,lambda records:wide_metrics(records,inventory,slot))
        errors.extend('wide: '+x for x in wide_result['integrity_errors'])
        valid=[r for r in wide if '_counts' in r]
        wide_result['all_attempt_peak_coverage']=wide_metrics(valid,inventory,slot) if valid else None
        wide_result['all_attempt_peaks_are_not_selected_views']=True
    else:wide_result=dict(producer_status='MISSING',selected_raw_predicate_sufficient=False,selected_coverage=None)
    directory=contained(audit.root/state_ref['path'],audit.root).parent
    if refs!={str(p.relative_to(audit.root)) for p in directory.glob('*.attempt.json')}:errors.append('orphan/omitted original attempt')
    if pngrefs!={str(p.relative_to(audit.root)) for p in directory.glob('*.png')}:errors.append('orphan/omitted original PNG')
    if state['image_count']!=sum('rgb' in r for r in hand):errors.append('hand RGB metadata count disagreement')
    sufficient=all(x['selected_raw_predicate_sufficient'] for x in arms.values()) and wide_result['selected_raw_predicate_sufficient']
    if state['status']=='qualified' and not sufficient:disagreements.append('qualified producer group lacks independently sufficient selected raw views')
    supplement=dict(observed_coverage_sufficient=False,reason='selected coverage missing')
    if all(x['selected_coverage'] for x in arms.values()) and wide_result['selected_coverage']:
        try:supplement=h.supplement(dict(arms={a:x['selected_coverage'] for a,x in arms.items()},
                    critical_body_pixels=wide_result['selected_coverage']['critical_body_peak_pixels']),inventory,slot)
        except Exception as exc:errors.append(error_text(exc))
    supplement['scope']='32 initial groups only; zero prefix steps and no final captures'
    supplement['acceptance']=False
    return dict(env_id=slot,producer_status=state['status'],producer_reasons=state['reasons'],state_reference=state_ref,
        raw_selected_predicate_sufficient=bool(sufficient),arms=arms,wide=wide_result,strong_camera_supplement_v3=supplement,
        attempts=attempt_reports,original_attempt_count=len(refs),original_png_count=len(pngrefs),raw_pixel_attempts_decoded=sum(x['raw_pixels_decoded'] for x in attempt_reports),
        held_native_fields=27,external_initial_fields=26,integrity_errors=errors,metadata_disagreements=disagreements,
        original_png_references=sorted(pngrefs),native_pass=False,stage_pass=False,safety_acceptance=False),baseline['environment_origins']


def audit_all(e,initial,output):
    batch=initial['batch'];packet=read_index(e,batch/'initial_camera_receipts.json',kind='worker');receipts=packet['receipts']
    require(len(receipts)==32 and [r['env_id'] for r in receipts]==list(range(32)),'all32 fixed ordered initial groups required')
    require(packet['qualified'] is False,'failed-initial diagnostic requires failed initial aggregate')
    audit=h.CameraAudit(batch,e,initial['body'],initial['identity'],initial['layout'])
    asset=e.js(H/'astra/next_mechanism/CAMERA_PALM_USD_IDENTITIES_V1.json');e.verify(asset['source_files_sha256'])
    reports=[];allpng=set();origins=None
    for receipt in receipts:
        slot=receipt['env_id']
        try:
            result,new_origins=inspect_group(audit,receipt,initial['fields'],asset,origins)
            if origins is None:origins=new_origins.copy()
            allpng.update(result['original_png_references'])
        except Exception as exc:
            result=dict(env_id=slot,producer_status=receipt.get('status'),integrity_errors=[error_text(exc)],
                original_png_count=0,raw_pixel_attempts_decoded=0,raw_selected_predicate_sufficient=False,
                native_pass=False,stage_pass=False,safety_acceptance=False)
        path=output/'groups'/f'env_{slot:03d}.json';write_new(path,result)
        reports.append(dict(env_id=slot,producer_status=result['producer_status'],
            raw_selected_predicate_sufficient=result['raw_selected_predicate_sufficient'],
            original_png_count=result['original_png_count'],raw_pixel_attempts_decoded=result['raw_pixel_attempts_decoded'],
            integrity_errors=result['integrity_errors'],report=str(path),sha256=sha(path)))
        print('INITIAL_CAMERA_DIAGNOSTIC env='+str(slot)+' errors='+str(len(result['integrity_errors'])),flush=True)
    observed={str(p.relative_to(batch)) for p in (batch/'qualified_views/initial').rglob('*.png')}
    errors=[]
    if observed!=allpng:errors.append('global initial PNG inventory differs from decoded attempt references')
    statepaths={str(p.relative_to(batch)) for p in (batch/'qualified_views/initial').rglob('state.json')}
    if statepaths!={r['state'] for r in receipts}:errors.append('global initial state inventory differs from all32 receipts')
    roots={Path(r['state']).parent.parent for r in receipts};require(len(roots)==1,'one shared initial capture event')
    shared=read_index(e,batch/next(iter(roots))/'receipts.json',kind='shared')
    require(shared['receipts']==receipts,'shared/worker initial receipts disagree')
    return dict(scope='ALL32_INITIAL_RECORDED_RAW_DIAGNOSTIC_ONLY',groups=reports,groups_visited=len(reports),
        bounded_camera_indices={'worker':packet['_stream'],'shared':shared['_stream']},
        producer_metadata_qualified=sum(r['status']=='qualified' for r in receipts),
        producer_metadata_failed_envs=[r['env_id'] for r in receipts if r['status']=='failed'],
        independently_sufficient_selected_raw_groups=sum(r['raw_selected_predicate_sufficient'] for r in reports),
        initial_png_inventory_count=len(observed),referenced_original_png_count=len(allpng),
        decoded_raw_pixel_attempts=sum(r['raw_pixel_attempts_decoded'] for r in reports),
        held_snapshot_checks=audit.snapshots,integrity_errors=errors,
        complete_raw_diagnostic=not errors and all(not r['integrity_errors'] for r in reports),
        no_view_reselection=True,zero_prefix_steps=True,no_final_captures=True,no_acceptance=True,
        limitations=['Pixel presence does not certify whole mesh, hidden surfaces, contact visibility or physical safety.',
          'Recorded authored bounds/table AABBs are inputs; native transforms and camera projection are rebuilt independently.',
          'All27 held fields are checked bitwise;26 are bound to native initial row; origins are held camera bytes.',
          'All-attempt peaks are diagnostic aggregates, not three-view qualifications.'],
        native_pass=False,stage_pass=False,safety_acceptance=False)
