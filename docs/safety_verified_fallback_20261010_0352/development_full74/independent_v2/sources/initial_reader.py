"""Read only initialized native row0 from a terminal failed first32 attempt."""
from pathlib import Path
import numpy as np
from evidence_io import ARMS,H,HERE,R,SCREEN,contained,require,exact
from binding_kernels import (EXECUTION_ENTRY,execution_sources,thread_settings,thread_cpu_binding,
    root_cpu_binding,diagnostics_cpu_binding)
from root_restore_reader import protocol_binding,audit_roots
from science import NAMESPACE,SEEDS,parameters_layout,scalar_contacts,flags,REASONS,STATES,counts


def source_inputs(e,w,root,anchors):
    e.verify(anchors['files']);b=e.js(SCREEN/'INPUT_BINDING_V1.json')
    require(len(b['files'])==509 and len(b['original500_sources'])==500 and
            all(b['files'][p]==s for p,s in b['original500_sources'].items()),'frozen509/500 denominator')
    e.verify(b['files']);schema=e.js(SCREEN/'NATIVE_SHORT_READER_SCHEMA_V3.json')
    req=e.js(w['request'],w['request_sha256'])
    require(req['schema']=='astra.full74.native_execution_request.v1' and req['parent_frozen'] is True and
        req['single_Kit_process_at_a_time'] is True and req['purpose']=='DEVELOPMENT_ONLY' and
        req['mode']=='camera-short' and req['maximum_batches']==1 and req['prefix_microsteps']==12 and
        req['cohort_subset']==schema['cohort_subset'] and req['out']==str(root/'native'),'fixed first32 request scope')
    require(req['input_binding_sha256']==e.hash(SCREEN/'INPUT_BINDING_V1.json') and
            req['runner_sources']==execution_sources(schema,anchors) and req['resource']==b['original_resource'],'exact request source/resource binding')
    e.verify(req['runner_sources']);protocol=protocol_binding(e,req,anchors,w['started_utc'])
    review=e.js(w['parent_review_path'],w['parent_review_sha256'])
    require(review['attempt_root']==str(root) and review['execute_authorized'] is True,'actual reviewed attempt')
    before=e.js(root/'SOURCE_BINDING_BEFORE.json');after=e.js(root/'SOURCE_BINDING_AFTER.json')
    require(w['source_binding']==before,'wait/source binding')
    original={k:v for k,v in req['runner_sources'].items() if k not in anchors['execution_extras']}
    for source in (before,after):
        require(source['original500_sources']==b['original500_sources'] and source['all_input_sources']==b['files'] and
                source['native_sources']==original and source['runtime_sources']==review['runtime_source_sha256'],
                'flattened original/runtime source binding')
        require(source['science_entry_sources_unchanged'] is True and source['process_groups_are_telemetry_not_ownership'] is True,
                'runtime ownership protocol')
        e.verify(source['runtime_sources'])
        require(source['runtime_CPU_report_sha256']==review['runtime_CPU_report_sha256'],'runtime CPU source proof')
    e.hash(review['runtime_CPU_report'],review['runtime_CPU_report_sha256'])
    life=e.js(review['lifecycle_CPU_report'],review['lifecycle_CPU_report_sha256'])
    require(life['status']=='PASS_CPU_SHUTDOWN_DURABILITY_ONLY','lifecycle CPU proof')
    e.verify(life['source_sha256']);e.verify(life['artifacts'])
    thread_cpu_binding(e,review,anchors);root_cpu_binding(e,review,anchors);diagnostics_cpu_binding(e,review,anchors)
    e.verify(w['prerequisite_sources']);require(req['parent_serialization_receipts']==w['prerequisite_sources'],'serialization receipt chain')
    argv=w['argv'];require(argv[1:3]==['-B',str(EXECUTION_ENTRY)] and '--execute-native-short' in argv,'V9 entry argv')
    for key,value in [('--request',w['request']),('--request-sha256',w['request_sha256']),('--proposal-start','0'),('--proposal-count','32'),('--prefix-microsteps','12')]:
        require(argv.count(key)==1 and argv[argv.index(key)+1]==value,'native argv '+key)
    gate=req['parent_prelaunch_gate'];require(gate['path']==w['parent_prelaunch_gate'] and gate['sha256']==w['parent_prelaunch_gate_sha256'],'prelaunch binding')
    manifest=contained(req['proposals_manifest'],R/'astra_full74_screen_development_v1')
    proposals=e.js(manifest,req['proposals_sha256']);reg=proposals['registration']
    require(reg==e.js(manifest.parent/'IDENTITY.json',proposals['identity_sha256']) and reg['namespace']==NAMESPACE and
            reg['purpose']=='DEVELOPMENT_ONLY' and reg['seeds']==SEEDS and reg['proposals']==8192 and reg['per_seed']==2048 and
            reg['lanes']==32 and reg['prefix_microsteps']==12,'development proposal registration')
    require(reg['selected']==reg['native_completed']==reg['formal_holdout']==0 and reg['policy_outcomes_used'] is False and
            reg['fresh_proposal_directory_opened'] is False and reg['parameter_sha256']==b['files'][b['parameter_path']],
            'unselected original proposal bank')
    e.verify(proposals['files'],manifest.parent);params=e.npz(b['parameter_path']);layout=parameters_layout(params)
    for key in ('hard','soft','bounds','vmax','controlled','names'):exact(e.npy(manifest.parent/(key+'.npy')),layout[key],'proposal '+key)
    q=e.npy(manifest.parent/'q74.npy',(8192,74),'float32');qd=e.npy(manifest.parent/'qd74.npy',(8192,74),'float32')
    require(np.isfinite(q).all() and np.isfinite(qd).all() and np.all(q>=layout['bounds'][:,0]) and
            np.all(q<=layout['bounds'][:,1]) and np.all(abs(qd)<=layout['vmax']),'entire8192 support')
    exact(e.npy(manifest.parent/'proposal_id.npy',(8192,),'int64'),np.tile(np.arange(2048,dtype=np.int64),4),'registered IDs')
    exact(e.npy(manifest.parent/'seed_index.npy',(8192,),'uint8'),np.repeat(np.arange(4,dtype=np.uint8),2048),'registered4 seeds')
    exact(e.npy(manifest.parent/'family.npy',(8192,),'uint8'),np.tile(np.arange(2048,dtype=np.uint8)%4,4),'registered families')
    return req,b,schema,params,layout,q[:32].copy(),qd[:32].copy(),protocol


def initial_only_inventory(e,root,batch):
    unexpected=[]
    for p in batch.glob('points_*'):
        if p.name!='points_000':unexpected.append(str(p))
    unexpected.extend(str(p) for p in batch.glob('applied_target74_micro_*.npy'))
    for p in (batch/'final_camera_receipts.json',batch/'BATCH_CLOSED.json',root/'PARENT_PRODUCT_CLOSURE.json'):
        if p.exists():unexpected.append(str(p))
    if (batch/'qualified_views/final').exists():unexpected.append(str(batch/'qualified_views/final'))
    rows=sorted(p.name for p in (batch/'native_fields').glob('row_*_complete.json'))
    require(rows==['row_000_complete.json'],'initial-only row completion inventory')
    require(not unexpected,'prefix/final/complete products present: '+str(unexpected))
    p=batch/'native_fields/CLOSED.json'
    if p.exists():require(e.js(p)['filled']==1,'stream contains more than initial row')
    return dict(completed_initial_rows=1,observed_prefix_steps=0,prefix12_completed=False,
        final_camera_present=False,producer_full_closure_present=False,uninitialized_native_stream_rows_read=False)


def contact_identity(e,batch,binding,body):
    identity=e.js(batch/'native_contact_identity.json');old=e.js(binding['contact_identity_path'])
    require(identity['environment_count']==32 and identity['physics_dt_s']==.008333 and identity['partners_env0']==old['partners_env0'] and
            identity['inventory']==old['inventory'] and len(identity['partners_env0'])==85,'initial native contact identity')
    require([x['arm'] for x in identity['views']]==list(ARMS),'contact arm order')
    owners=[[] for _ in range(32)]
    for view,ref in zip(identity['views'],old['views']):
        for key in ('arm','sensors','filters','env_ids','capacity'):require(view[key]==ref[key],'contact owner/filter '+key)
        require(view['capacity']==262144 and len(view['sensors'])==32*len(body[view['arm']]),'contact owner denominator')
        for path,lane,filters in zip(view['sensors'],view['env_ids'],view['filters']):
            require(filters==[p.replace('/env_0/',f'/env_{lane}/') for p in identity['partners_env0']],'native partner order')
            owners[lane].append(path)
    require(all(len(x)==len(set(x))==82 for x in owners),'82 unique native owners')
    return identity


def actual_entry_readbacks(e,out,entry,params):
    parameter_readbacks={}
    for label in ('BEFORE','AFTER'):
        p=out/('PARAMETER_READBACK_'+label+'_V1.npz')
        if p.exists():
            ref=(entry or {}).get('parameter_readbacks',{}).get(label.lower())
            require(ref is not None and ref['path']==str(p),'unbound actual parameter readback')
            e.hash(p,ref['sha256'])
            actual=e.npz(p);keys={k for k in params if not k.endswith('_full_initial_position_target') and k!='initial_strata'}
            require(set(actual)==keys,'parameter getter inventory')
            for k in keys:exact(actual[k],params[k],'native parameter '+label+' '+k)
            parameter_readbacks[label]=dict(status='RECORDED_BITWISE_MATCH_ONLY',path=str(p),sha256=e.hash(p))
        else:parameter_readbacks[label]=dict(status='UNKNOWN_MISSING_ACTUAL_ARTIFACT',path=str(p))
    producer={'actual_parameter_readbacks':parameter_readbacks}
    producer['full_native_parameters_certified']=False
    producer['frozen_reference_parameters_are_actual_readback']=False
    geometry=out/'NATIVE_GEOMETRY_IDENTITY_V1.json'
    producer['native_geometry_identity']=dict(status='UNKNOWN_MISSING_ACTUAL_ARTIFACT',path=str(geometry))
    if geometry.exists():
        ref=(entry or {}).get('geometry_identity');require(ref and ref['path']==str(geometry),'unbound native geometry identity')
        producer['native_geometry_identity']=dict(status='RECORDED_NOT_INDEPENDENTLY_CERTIFIED',path=str(geometry),sha256=e.hash(geometry,ref['sha256']))
    return producer


def read_initial(e,w,receipts,root,anchors):
    req,b,schema,params,layout,q,qd,protocol=source_inputs(e,w,root,anchors)
    out=root/'native';batch=out/'batch_00000';inventory=initial_only_inventory(e,root,batch)
    native=receipts['native_receipt'];entry=receipts['entry_receipt']
    producer=dict(native_receipt_present=native is not None,entry_receipt_present=entry is not None,
                  actual_thread_settings='UNKNOWN_NO_ENTRY_RECEIPT',app_close_error=None)
    if native is not None:
        require(native['schema']=='astra.full74.native_screen_receipt.v1' and native['backend']=='isaac_physx_native' and
                native['status']=='failed' and native['mode']=='camera-short' and native['completed_batches']==0 and
                native['actual_prefix_microsteps']==0 and native['camera_all32_initial_final_qualified'] is False,'failed initial-only producer receipt')
        require(native['runner_sources']==req['runner_sources'] and native['input_binding_sha256']==req['input_binding_sha256'] and
                native['proposals_sha256']==req['proposals_sha256'],'failed receipt request/source')
        e.verify(native['artifacts'],out)
        producer.update(native_status=native['status'],native_error=native.get('error'),camera_cleanup=native.get('camera_cleanup'))
    if entry is not None:
        require(entry['schema']=='astra.full74.native_short_entry.v1' and entry['status']=='failed' and
                entry['entry_sha256']==anchors['files'][str(EXECUTION_ENTRY)] and entry['request']==w['request'] and
                entry['request_sha256']==w['request_sha256'],'failed entry binding')
        producer.update(entry_status=entry['status'],entry_error=entry.get('error'),entry_cleanup_errors=entry.get('cleanup_errors'),
            actual_thread_settings=thread_settings(entry))
    p=out/'APP_CLOSE_ERROR_V3.json'
    if p.exists():producer['app_close_error']=e.js(p)
    producer.update(actual_entry_readbacks(e,out,entry,params))
    exact(e.npy(batch/'proposal_ids.npy',(32,),'int64'),np.arange(32,dtype=np.int64),'fixed attempted IDs0..31')
    exact(e.npy(batch/'requested_q74.npy',(32,74),'float32'),q,'requested initial q74')
    exact(e.npy(batch/'requested_qd74.npy',(32,74),'float32'),qd,'requested initial qd74')
    exact(e.npy(batch/'fixed_prefix_targets_6x32x74.npy',(6,32,74),'float32'),np.repeat(q[None],6,axis=0),'six scheduled targets, not executed controls')
    points=batch/'points_000';closed=e.js(points/'CLOSED.json')
    require(closed['micro']==0 and closed['fresh'] is False,'initial point freshness must remain UNKNOWN')
    e.verify(closed['files'],points)
    require(set(closed['files'])=={str(p) for p in points.glob('*.npy')},'unbound initial point file')
    require(e.js(batch/'native_fields/row_000_complete.json')=={'row':0},'initial row marker')
    row={k:e.npy_first_row(batch/'native_fields'/(k+'.npy'),v['shape'],v['dtype']) for k,v in schema['native_fields'].items()}
    statekeys=set(row)-{'scalar82','violations'}
    for key in statekeys:exact(e.npy(points/(key+'.npy')),row[key],'point/native initial '+key)
    exact(row['q74'],q,'native full74 initial q');exact(row['qd74'],qd,'native full74 initial qd');exact(row['target74'],q,'native full74 initial target')
    require(int(row['micro'])==0 and int(row['macro_step'])==int(row['substep'])==-1 and not bool(row['point_fresh']) and
            bool(row['initial_point_freshness_unknown']),'actual initial row indices/freshness')
    initial=e.npz(b['initial_path']);fields={k:v[None] for k,v in row.items()}
    restored=audit_roots(fields,initial,protocol,e.js(batch/'ROOT_RESTORE_EQUIVALENCE_V1.json'))
    body=e.npz(Path(b['parameter_path']).parent/'body_identity.npz');identity=contact_identity(e,batch,b,body)
    keys={'counts','starts','point_indices','sensor_indices','partner_indices','normal_forces','points','normals','separations','partner_abs_normal_sum_N'}
    require({Path(p).stem for p in closed['files']}==statekeys|{a+'_'+k for a in ARMS for k in keys},'initial point inventory')
    by_lane=[[] for _ in range(32)];point_count=0
    for view in identity['views']:
        arm=view['arm'];arrays={k:e.npy(points/(arm+'_'+k+'.npy')) for k in keys}
        require(arrays['counts'].shape==(len(view['sensors']),85),'native contact matrix')
        sums=scalar_contacts(arrays['counts'],arrays['starts'],arrays,arrays['normal_forces']);n=len(arrays['point_indices']);point_count+=n
        for k,width in [('points',3),('normals',3),('separations',1)]:
            require(arrays[k].shape==(n,width) and arrays[k].dtype in (np.float32,np.float64) and np.isfinite(arrays[k]).all(),'initial owned point field '+k)
        exact(sums,arrays['partner_abs_normal_sum_N'],'initial recorded scalar algebra; freshness UNKNOWN')
        for i,lane in enumerate(view['env_ids']):by_lane[lane].append(sums[i].max())
    exact(np.asarray(by_lane,np.float64),row['scalar82'],'initial82 scalar getter bytes')
    violations=flags(row,layout,q,False);exact(violations,row['violations'],'initial numeric reason mask; contact not fresh')
    ledger=e.npy(out/'all_proposal_status.npy',(8192,),'uint8');exact(ledger[32:],np.zeros(8160,np.uint8),'retained8160 unused')
    require(np.all(ledger<len(STATES)),'ledger state range')
    if native is not None:require(native['proposal_status_counts']==counts(ledger),'producer/independent8192 ledger counts')
    return dict(req=req,binding=b,layout=layout,body=body,identity=identity,fields=fields,batch=batch,
        report=dict(scope='ACTUAL_INITIAL_ONLY_NO_PREFIX_OR_FINAL',inventory=inventory,producer=producer,
            all32_initial_q74_qd74_target74_bitwise=True,all6_targets_scheduled_only=True,executed_prefix_controls=0,
            root_restore=restored,point000_raw_getter_points=point_count,initial_contact_freshness='UNKNOWN',
            recorded_raw9021_shape=list(row['raw9021'].shape),geometry_reconstruction='UNKNOWN_WITHOUT_ACTUAL_NATIVE_GEOMETRY_IDENTITY',
            initial_numeric_reason_counts={r:int(violations[:,i].sum()) for i,r in enumerate(REASONS)},
            initial_numeric_reason_mask_32x7=violations.tolist(),ledger_8192=ledger.tolist(),ledger_counts=counts(ledger),
            unused_count=8160,selected_count=0,formal_holdout_count=0,native_pass=False,stage_pass=False,safety_acceptance=False))
