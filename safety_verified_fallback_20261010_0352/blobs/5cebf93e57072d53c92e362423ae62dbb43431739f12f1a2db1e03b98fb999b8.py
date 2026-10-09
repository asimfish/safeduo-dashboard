"""First CLOSED candidate480x32 CPU calibration; NO GPU or runtime mutation."""
import csv
import json
import mmap
from pathlib import Path
import resource
import sys
import tempfile

import numpy as np

from native_teacherforced_cpu_v1 import ARMS,WIDTHS,Macro74,UNKNOWN,VARIANTS,require,same_bytes,teacher_forced_two_micro
from native_calibration_strata_v2 import contact_strata,STRATA_DEFINITION,error_statistics,original_delay6
from run_native_calibration_v1 import D,H,A,Inputs,sha,write_json,frozen_hashes,finish_frozen,block_diag

ROOT=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/shard_verified_v1/batch0_shard0_verified')
TMP=ROOT.parent.parent/'native_calibration_cpu_tmp'
OUT=D/'first_candidate480x32_v2'
API=Path('/home/liyufeng/miniforge3/envs/safeduo/lib/python3.11/site-packages/isaacsim/extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physics/tensors/impl/api.py')
ACTUATOR=Path('/home/liyufeng/safeduo_isaaclab/source/isaaclab/isaaclab/actuators/actuator_pd.py')


def supplemental_witness_strata(inputs):
    rows=[]; findings=[]
    inputs.add(D/'two_witness_v1/known_witness_report.json','minimal known-witness baseline report')
    known=json.loads((D/'two_witness_v1/known_witness_report.json').read_text())
    arrays_path=inputs.add(D/'two_witness_v1/two_witness_arrays.npz','minimal teacher-forced errors')
    with np.load(arrays_path,allow_pickle=False) as arrays:
        for lane in (18,19):
            p=inputs.add(H/f'astra/adaptation32/strategy_original_witness_lane{lane}_v4.npz','immutable integration witness for scalar strata')
            with np.load(p,allow_pickle=False) as z:scalar=z['actual_scalar_peak_N'][:,None]
            masks=contact_strata(scalar)
            for v in VARIANTS:
                name=v.name
                for stratum,mask in masks.items():
                    qe=arrays[f'lane{lane}_{name}_q_error'][mask[:,0]]
                    ve=arrays[f'lane{lane}_{name}_qd_error'][mask[:,0]]
                    rows.append(dict(lane=lane,variant=name,stratum=stratum,unique_native_micros=int(mask.sum()),
                                     q=error_statistics(qe),qd=error_statistics(ve)))
            w=next(x for x in known['cases'] if x['lane']==lane)
            if lane==18:
                failures=[x for x in w['variants'] if x['name'].startswith('implicit_')]
                require(all(x['known_hard_joint']['model_hard_violation']==0 for x in failures),'implicit false-positive summary changed')
                findings.append(dict(lane=18,micro=12,event_scalar_N=0.,recorded_contact_free_proxy=bool(masks['recorded_contact_free_proxy'][12,0]),
                    no_positive_scalar_observed_so_far=True,all7_implicit_variants_still_miss_native_lower_hard_violation=True,
                    model_q_range_rad=[min(x['known_hard_joint']['predicted_q'] for x in failures),max(x['known_hard_joint']['predicted_q'] for x in failures)],
                    actual_q_rad=-.001607122365385294,baseline_q_error_rad=.009403144913260346,
                    explicit_diagnostic='Predicts +1.851258rad above upper hard limit; huge erroneous trajectory, not a repaired native model.'))
            else:
                findings.append(dict(lane=19,micro=13,event_scalar_N=float(scalar[13,0]),contact_linked_macro=True,
                    CONTACT='UNKNOWN',variants_do_not_predict_or_bound_contact=True))
    write_json(OUT/'known_witness_stratification_v2.json',dict(scope='SAME_ORIGINAL2_WITNESSES_NOT_NEW_RUN',
        rows=rows,findings=findings,strata=STRATA_DEFINITION,
        target_alignment='Each macro uses actual full74 target; original36 repeated plan diverges atmicro14 for both lanes and is disabled.',**UNKNOWN))


def main():
    OUT.mkdir(exist_ok=False)
    inputs=Inputs(); frozen=frozen_hashes(inputs)
    for p in sorted(D.glob('*.py')):inputs.add(p,'calibration implementation source')
    for p in (H/'TASK_ACCEPTANCE_PROTOCOL_V1.json',H/'prospective_task_gate_v2.py',H/'native_verified_v4.py',API,ACTUATOR):
        inputs.add(p,'read-only source / newly frozen protocol provenance')
    task=json.loads((H/'TASK_ACCEPTANCE_PROTOCOL_V1.json').read_text())
    for path,expected in task['source_sha256'].items():inputs.add(path,'task protocol frozen source',expected)
    summary_path=inputs.add(A/'FIRST_FULL_CANDIDATE_INDEPENDENT_SUMMARY_V1.json','independent closed first candidate locator')
    summary=json.loads(summary_path.read_text())
    model_ref=summary['reports']['model']; model=json.loads(inputs.add(model_ref['path'],'closed original field SHA authority',model_ref['sha256']).read_text())
    closure=model['closure'];require(closure['actual_exit']==0 and closure['actual_wait_receipt_present'],'selected native subjob not closed0')
    require(closure['producer_root']==str(ROOT) and model['controls']==480 and model['lanes']==32,'exact first completed candidate scope')
    for pk,hk in [('binding','binding_sha256'),('execution_snapshot','execution_snapshot_sha256')]:
        inputs.add(closure[pk],'original closed native subjob provenance',closure[hk])
    tracking_ref=summary['reports']['tracking']; tracking=json.loads(inputs.add(tracking_ref['path'],'immutable original input hashes',tracking_ref['sha256']).read_text())
    ledger=tracking['input_sha256']
    def expected(path):
        e=ledger.get(str(path));return e.get('sha256') if isinstance(e,dict) else e
    params_path=inputs.add(ROOT/'resolved_native_parameters.npz','recorded native getter parameters',expected(ROOT/'resolved_native_parameters.npz'))
    with np.load(params_path,allow_pickle=False) as z:params={k:z[k] for k in z.files}
    fields={}
    for group,names in [('dynamics_fields',['step']+[a+'_'+k for a in ARMS for k in ('q','qd','mass_matrix','actual_velocity_target','actuation_force','projected_joint_force','gravity','coriolis','actual_position_target')]),
                        ('response_fields',['pre_sim_time_s','pre_sim_step_index','applied_target','scalar_normal_max_N']+[p+a+'_'+k for a in ARMS for k in ('q','qd') for p in ('pre_','post_')])]:
        authority=next(x for x in model['input_ledgers'] if x['directory']==group)
        cp=inputs.add(ROOT/group/'closed.json','closed typed-field schema',authority['closed_sha256'])
        closed=json.loads(cp.read_text());require(closed['filled']==closed['registered']==480 and closed['status']=='complete','field stream not480 closed')
        for name in names:
            path=inputs.add(ROOT/group/(name+'.npy'),'original recorded numerical field',authority['original_sha256'][name])
            fields[group+'/'+name]=np.load(path,allow_pickle=False,mmap_mode='r')
    def ds(name):return fields['dynamics_fields/'+name]
    def rs(name):return fields['response_fields/'+name]
    require(np.array_equal(ds('step'),np.arange(480)),'480 native macro indices')
    ids=[a+'/'+str(j) for a in ARMS for j in params[a+'_native_joint_names']]
    offsets=np.cumsum([0,*WIDTHS]); controlled=np.concatenate([params[a+'_controlled_joint_indices']+offsets[i] for i,a in enumerate(ARMS)])
    require(len(ids)==len(set(ids))==74 and len(set(controlled))==26,'canonical74/26')
    preq=np.concatenate([ds(a+'_q') for a in ARMS],axis=-1)
    prev=np.concatenate([ds(a+'_qd') for a in ARMS],axis=-1)
    for a in ARMS:
        require(same_bytes(ds(a+'_q'),rs('pre_'+a+'_q')) and same_bytes(ds(a+'_qd'),rs('pre_'+a+'_qd')),'pre-state field disagreement')
    q=np.empty((960,32,74),np.float32); v=np.empty_like(q); target=np.empty_like(q)
    scalar=np.empty((960,32),float); times=np.empty(960); steps=np.empty(960,np.int64)
    receipts_path=inputs.add(ROOT/'point_contact_receipts.json','closed native micro chunks',expected(ROOT/'point_contact_receipts.json'))
    receipts=json.loads(receipts_path.read_text());require(receipts['status']=='complete' and receipts['physics_events']==960,'960micro contact receipt')
    identity=json.loads(inputs.add(ROOT/'native_contact_identity.json','all82 owner lane identity only',receipts['identity_sha256']).read_text())
    owner_map={spec['arm']:[np.flatnonzero(np.asarray(spec['env_ids'])==lane) for lane in range(32)] for spec in identity['views']}
    require(all(sum(len(owner_map[a][lane]) for a in ARMS)==82 for lane in range(32)),'all82 owners per lane')
    cursor=0
    for chunk in receipts['chunks']:
        lo,hi=chunk['start'],chunk['stop'];require(lo==cursor and lo%2==hi%2==0,'contiguous macro-aligned chunks')
        p=inputs.add(ROOT/chunk['path'],'actual native micro q/qd/full74 target/contact scalar',chunk['sha256'])
        with np.load(p,allow_pickle=False) as z:
            require(np.array_equal(z['frame'],np.arange(lo,hi)//2) and np.array_equal(z['substep'],np.arange(lo,hi)%2),'native micro clocks')
            times[lo:hi]=z['simulation_time_s'];steps[lo:hi]=z['simulation_time_step_index']
            scalar[lo:hi]=0.
            for i,a in enumerate(ARMS):
                sl=slice(offsets[i],offsets[i+1])
                q[lo:hi,:,sl]=z[a+'_native_q'];v[lo:hi,:,sl]=z[a+'_native_qd'];target[lo:hi,:,sl]=z[a+'_native_position_targets']
                allowner=z[a+'_partner_scalar_abs_N'].max(-1)
                for lane in range(32):scalar[lo:hi,lane]=np.maximum(scalar[lo:hi,lane],allowner[:,owner_map[a][lane]].max(-1))
                del allowner
        cursor=hi
    require(cursor==960 and np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(target).all(),'complete native response')
    require(same_bytes(target[0::2],target[1::2]),'within-macro actual74 targets changed')
    require(same_bytes(target[0::2,:,controlled],rs('applied_target')),'actual controlled26 targets differ')
    require(same_bytes(preq[1:],q[1:-1:2]) and same_bytes(prev[1:],v[1:-1:2]),'native state chain mismatch')
    for i,a in enumerate(ARMS):
        sl=slice(offsets[i],offsets[i+1])
        require(same_bytes(q[1::2,:,sl],rs('post_'+a+'_q')) and same_bytes(v[1::2,:,sl],rs('post_'+a+'_qd')),'native macro post-state mismatch')
        require(same_bytes(target[0::2,:,sl],ds(a+'_actual_position_target')),'actual native full target disagrees with dynamics field')
    require(np.array_equal(scalar.reshape(480,2,32).max(1),rs('scalar_normal_max_N')),'all-owner scalar vs response mismatch')
    ip=inputs.add(ROOT/'native_initial_all_lanes.npz','original initial state / task reference',expected(ROOT/'native_initial_all_lanes.npz'))
    with np.load(ip,allow_pickle=False) as z:initial=np.concatenate([z[a+'_q'] for a in ARMS],axis=-1);initial_v=np.concatenate([z[a+'_qd'] for a in ARMS],axis=-1)
    require(same_bytes(initial,preq[0]) and same_bytes(initial_v,prev[0]),'native initial state mismatch')
    tape_path=inputs.add(ROOT/'actual_random_issue_tape.npz','original requested26; distinct from applied target',expected(ROOT/'actual_random_issue_tape.npz'))
    with np.load(tape_path,allow_pickle=False) as z:requests=np.concatenate([z[a] for a in ARMS],axis=-1)
    task_reference=original_delay6(initial[:,controlled],requests)
    has_pause=same_bytes(requests[240:272],np.repeat(requests[239:240],32,axis=0))
    masks=contact_strata(scalar)
    # Time labels use prospective protocol's actual6-delay windows, not fallback actions.
    c=np.arange(960)//2
    phase_masks={name:np.broadcast_to(mask[:,None],(960,32)) for name,mask in {
        'initial_fifo_controls0_6':c<6,'active_pre_pause_controls6_246':(c>=6)&(c<246),
        'applied_pause_window_controls246_278':(c>=246)&(c<278),
        'resume_window_controls278_310':(c>=278)&(c<310),
        'later_active_controls310_480':c>=310}.items()}
    for name,mask in phase_masks.items():masks[name]=mask
    source_stats={a:dict(armature_values=np.unique(params[a+'_armature']),gravity_disabled_values=np.unique(params[a+'_disable_gravity_cfg']),
        explicit_actuation_max_abs=float(np.abs(ds(a+'_actuation_force')).max()),
        projected_force_max_abs=float(np.abs(ds(a+'_projected_joint_force')).max())) for a in ARMS}
    source_stats['armature_note']='get_dof_armatures recorded native float32 values, not inferred gains. Generalized M getter wrapper does not document inclusion of armature; both interpretations retained.'
    with (OUT/'actual_native_shared_arrays.npz').open('xb') as f:
        np.savez_compressed(f,q=q,qd=v,actual_targets=target,pre_q=preq,pre_qd=prev,scalar_peak_N=scalar,
            native_steps=steps,native_times=times,original_request26=requests,delayed_original_request26=task_reference,controlled_columns=controlled)
    supplemental_witness_strata(inputs)
    print('INPUT_CLOSED: first candidate480x32,30720 micro-lane observations; same-target/clock/state chain validated',flush=True)
    def cat_param(key,lane):return np.concatenate([params[a+'_'+key][lane] for a in ARMS])
    def cat_ds(key,control,lane):return np.concatenate([ds(a+'_'+key)[control,lane] for a in ARMS])
    params_by_lane=[];initial_mass=[];initial_bias=[]
    for lane in range(32):
        flags=np.concatenate([np.full(w,params[a+'_disable_gravity_cfg'][lane],bool) for a,w in zip(ARMS,WIDTHS)])
        params_by_lane.append(dict(armature=cat_param('armature',lane),kp=cat_param('stiffness',lane),kd=cat_param('damping',lane),
            effort_limits=cat_param('max_force',lane),gravity_disabled=flags))
        initial_mass.append(block_diag([ds(a+'_mass_matrix')[0,lane] for a in ARMS]))
        initial_bias.append(cat_ds('coriolis',0,lane).astype(float)+cat_ds('gravity',0,lane).astype(float)*~flags)
    summaries=[]; tables=[]; lane_summaries=[]; full_error_files=[]
    with tempfile.TemporaryDirectory(prefix='first480_v2_',dir=TMP) as tmp:
        for variant in VARIANTS:
            qe=np.lib.format.open_memmap(Path(tmp)/'q_error.npy',mode='w+',dtype=np.float64,shape=(960,32,74))
            ve=np.lib.format.open_memmap(Path(tmp)/'qd_error.npy',mode='w+',dtype=np.float64,shape=(960,32,74))
            residual=0.;saturation=0;worst=[]
            for control in range(480):
                pair=slice(2*control,2*control+2)
                for lane in range(32):
                    sample=Macro74(q=preq[control,lane],qd=prev[control,lane],actual_targets=target[pair,lane],
                        native_q=q[pair,lane],native_qd=v[pair,lane],
                        mass=block_diag([ds(a+'_mass_matrix')[control,lane] for a in ARMS]),
                        gravity=cat_ds('gravity',control,lane),coriolis=cat_ds('coriolis',control,lane),
                        velocity_target=cat_ds('actual_velocity_target',control,lane),explicit_actuation=cat_ds('actuation_force',control,lane),
                        dt_model=.008333,pre_native_step=int(rs('pre_sim_step_index')[control]),native_steps=steps[pair],
                        pre_native_time=float(rs('pre_sim_time_s')[control]),native_times=times[pair],native_tick=float(np.float32(1/120)),
                        **params_by_lane[lane])
                    r=teacher_forced_two_micro(sample,variant,initial_mass=initial_mass[lane],initial_bias=initial_bias[lane])
                    qe[pair,lane]=r['q_error'];ve[pair,lane]=r['qd_error']
                    residual=max(residual,max(x['residual_max'] for x in r['numerical']))
                    saturation+=int(sum(x['saturation_mask'].sum() for x in r['numerical']))
                if control%120==119:print(f'{variant.name}: {control+1}/480 controls all32 lanes',flush=True)
            qe.flush();ve.flush()
            strata=[]
            for name,mask in masks.items():
                e=qe[mask];ev=ve[mask]
                strata.append(dict(name=name,native_micro_lane_samples=int(mask.sum()),q=error_statistics(e),qd=error_statistics(ev)))
                for sub in ('both',0,1):
                    selected=mask.copy()
                    if sub!='both':selected[np.arange(960)%2!=sub]=False
                    for j,joint in enumerate(ids):
                        row=dict(variant=variant.name,stratum=name,substep=sub,column=j,joint=joint,
                            group='controlled26' if j in controlled else 'hand48',native_micro_lane_samples=int(selected.sum()))
                        row.update({'q_'+k:value for k,value in error_statistics(qe[:,:,j][selected]).items()})
                        row.update({'qd_'+k:value for k,value in error_statistics(ve[:,:,j][selected]).items()})
                        tables.append(row)
                del e,ev
            for lane in range(32):
                lane_summaries.append(dict(variant=variant.name,lane=lane,q=error_statistics(qe[:,lane]),qd=error_statistics(ve[:,lane]),
                                          positive_scalar_micro_count=int((scalar[:,lane]>0).sum())))
            for key,error in [('q',qe),('qd',ve)]:
                micro,lane,j=np.unravel_index(np.abs(error).argmax(),error.shape)
                worst.append(dict(metric=key,micro=int(micro),control=int(micro//2),sub=int(micro%2),lane=int(lane),column=int(j),joint=ids[j],
                    signed_error=float(error[micro,lane,j]),actual_target=float(target[micro,lane,j]),
                    native_value=float((q if key=='q' else v)[micro,lane,j]),scalar_peak_N=float(scalar[micro,lane]),
                    recorded_contact_free_proxy=bool(masks['recorded_contact_free_proxy'][micro,lane])))
            p=OUT/(variant.name+'_errors74.npz')
            with p.open('xb') as f:np.savez_compressed(f,q_error=qe,qd_error=ve)
            full_error_files.append(dict(path=str(p),sha256=sha(p)))
            summaries.append(dict(variant=variant.name,all_solves_converged=True,max_equation_residual=residual,
                saturated_model_coordinate_micro_count=saturation,strata=strata,worst=worst))
            print(f'VARIANT_COMPLETE {variant.name}: qmax={worst[0]["signed_error"]:.8g}, qdmax={worst[1]["signed_error"]:.8g}',flush=True)
            qe._mmap.close();ve._mmap.close();del qe,ve
    with (OUT/'per74_stratified_error_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    write_json(OUT/'per_lane_variant_errors.json',lane_summaries)
    report=dict(schema='safeduo.astra.native_calibration.full480.v2',scope='DEVELOPMENT_FIRST_CLOSED_CANDIDATE480x32',
        original_native_root=str(ROOT),native_closure=closure,controls_per_lane=480,lanes=32,microsteps_per_lane=960,
        unique_macro_lane_samples=15360,unique_micro_lane_samples=30720,all_cases_retained=True,variants_reuse_same_samples=True,
        target_alignment=dict(all15360_macro_full74_targets_bitexact_across_two_micro=True,controlled26_crosschecked=True,
            dynamics_full74_target_crosschecked=True,native_pre_post_state_chain_bitexact=True,
            teacher_forcing='Reset every control to actual native pre-q/qd; propagate2micro under same ACTUAL full74 target.',
            future36_direct_error_enabled=False,future36_reason='No planned36 trajectory used; only actual-target macro2micro calibration.'),
        strata_definitions=STRATA_DEFINITION,strata_counts={name:int(mask.sum()) for name,mask in masks.items()},
        initial_precontrol_contact='UNKNOWN_UNMEASURED',summaries=summaries,error_array_files=full_error_files,
        parameter_getter_provenance=source_stats,
        torque_provenance=dict(recorded_getter='ArticulationView.get_dof_actuation_forces()',
            recorded_stage='native_verified_v4.py pre-control dynamics capture',
            projected_getter='get_dof_projected_joint_forces(): motion-axis projection of incoming link forces',
            implicit_drive_actual_torque_getter='NOT_PRESENT_IN_RETAINED_EVIDENCE',
            isaaclab_implicit_actuator='actuator_pd.py lines38..47 and122..141 identify approximate computed/applied effort; no actual-drive-torque measurement.',
            native_M_armature_inclusion='UNKNOWN: native API wrapper does not specify armature inclusion; no independent native counterfactual run.'),
        task_protocol=dict(path=str(H/'TASK_ACCEPTANCE_PROTOCOL_V1.json'),sha256=sha(H/'TASK_ACCEPTANCE_PROTOCOL_V1.json'),
            gate_path=str(H/'prospective_task_gate_v2.py'),gate_sha256=sha(H/'prospective_task_gate_v2.py'),
            reference='Original requested26 delayed exactly6controls; calibration target is actual applied74 and is never task ground truth.',
            original_request_pause240_272_present=has_pause,
            task_acceptance_evaluated=False,
            phase_labels='Prospective actual windows used only for error time stratification; historical data must not be relabeled as having a pause when original request tape lacks it.'),
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        independent_holdout_count=0,parameters_fitted=False,certified_bound=None,
        all_empirical_errors_are_observed_sample_statistics=True,
        contact_strata_are_associations_not_causal_identification=True,gpu_started=False,**UNKNOWN)
    write_json(OUT/'FULL480_NATIVE_CALIBRATION_REPORT_V2.json',report)
    finish_frozen(frozen);write_json(OUT/'frozen493_before_after.json',frozen)
    write_json(OUT/'input_source_sha256.json',dict(entries=inputs.entries,verification=inputs.recheck(),
        numpy_version=np.__version__,python_version=sys.version,interpreter_sha256=sha(sys.executable)))
    write_json(OUT/'output_sha256.json',{str(p):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    require(not any(k=='torch' or k.startswith(('isaac','omni.')) for k in sys.modules),'forbidden runtime imported')
    print(json.dumps(dict(status='COMPLETE_DEVELOPMENT_ONLY',macro_lane_samples=15360,micro_lane_samples=30720,
        variants=len(VARIANTS),original_request_pause_present=has_pause,all493_unchanged=True,
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
