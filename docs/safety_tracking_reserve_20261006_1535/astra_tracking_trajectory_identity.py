"""Post-observation independent saved-trajectory comparison, no new windows.

Do not import parent saturation/zero-intent analysis or modify frozen scorer.
This companion deliberately does not re-read all forecast chunks. Gap formula
and stored minima are checked by the existing independent raw scorer; saturation
itself remains a separate parent diagnostic until its evidence is reviewed.
"""
import json
import numpy as np
from astra_tracking_raw_score import H,Ledger,canonical_closed,now,registered_inputs,sha,write_owned
from astra_tracking_score_core import require,fingerprint

FIELDS={
    'q':((960,64,26),1), 'pre_qd_compact':((960,64,26),1),
    'q_initial':((64,26),0), 'cmd':((960,64,26),1),
    'external_unscaled_cmd':((960,64,26),1), 'exec':((960,64,26),1),
    'controller_target':((960,64,26),1), 'actuator_target':((960,64,26),1),
    'effective_target_delta':((960,64,26),1),
    'pre_pending_actuator_targets':((960,6,64,26),2),
    'pre_pending_project_history':((960,6,64,26),2),
    'pre_target_debt':((960,64,26),1),
    'official_margins':((960,64,4),1), 'official_deep':((960,64,4),1)}

def compare(a,b,env_axis,allow_positive_inf=False):
    require(a.shape==b.shape and a.dtype==b.dtype,'trajectory dtype/shape mismatch')
    require(a.dtype in (np.dtype('float32'),np.dtype('bool')),'unexpected native dtype')
    if a.dtype.kind=='f':
        for x in (a,b):
            valid=~np.isnan(x)&~np.isneginf(x) if allow_positive_inf else np.isfinite(x)
            require(valid.all(),'nonfinite trajectory outside declared absent class')
    aa=np.ascontiguousarray(np.moveaxis(a,env_axis,0))
    bb=np.ascontiguousarray(np.moveaxis(b,env_axis,0))
    bits_equal=(aa.view(np.uint8).reshape(len(aa),-1)==bb.view(np.uint8).reshape(len(bb),-1)).all(-1)
    return dict(shape=list(a.shape),dtype=a.dtype.str,all_bits_equal=bool(bits_equal.all()),
        env_bits_equal=bits_equal.tolist(),tight_sha256=fingerprint(a),adaptive_sha256=fingerprint(b),
        differing_env_ids=np.flatnonzero(~bits_equal).tolist())

def main():
    ledger=Ledger();own=ledger.json(H/'astra_tracking_trajectory_plan.json')
    require(own['status']=='REGISTERED_POST_OBSERVATION_READONLY_TRAJECTORY_COMPARISON','not registered')
    for n,d in own['source_sha256'].items():ledger.bind(H/n,d)
    require(not (H/'ASTRA_TRACKING_FINAL_TRAJECTORY_IDENTITY.json').exists(),'preserve prior comparison')
    design,plans=registered_inputs(ledger);blocks=[];cases=[]
    for block,plan in enumerate(plans):
        records=canonical_closed(ledger,plan)
        require(records is not None,'canonical campaign still open')
        jobs={j['env']['SAFEDUO_JOINT_MODE']:j for j in plan['jobs']}
        arrays={};paths={}
        for mode in ('tight_reference','delay_reserve'):
            job=jobs[mode];record=records[job['id']]
            from pathlib import Path
            root=Path(plan['output_root'])/job['id'];file=root/'cell_001.npz'
            require(record['exit_code']==0 and record['status']=='complete','canonical child incomplete')
            require(record['argv']==job['argv']+['--out',str(root)],'actual argv not frozen command')
            protocol=ledger.json(root/'protocol.json',record['protocol_sha256'])
            require(protocol['status']=='complete' and protocol['steps']==960 and protocol['completed_cells']==1,'protocol incomplete')
            arrays[mode]=ledger.npz(file,list(FIELDS));paths[mode]=str(file)
        fields={}
        for name,(shape,axis) in FIELDS.items():
            a=arrays['tight_reference'][name];b=arrays['delay_reserve'][name]
            require(a.shape==shape and b.shape==shape,'registered field shape: '+name)
            require(a.dtype==(np.bool_ if name=='official_deep' else np.float32),'registered dtype: '+name)
            fields[name]=compare(a,b,axis,allow_positive_inf=name=='official_margins')
        equal=np.array([f['env_bits_equal'] for f in fields.values()]).all(0)
        row=design['rows'][block]
        cases.extend(dict(case_id=f'b{block}:i{row["initial_seed"]}:c{row["command_seed"]}:e{e:02d}',
            block=block,env=e,declared_saved_arrays_bitwise_equal=bool(equal[e]),
            differing_fields=[name for name,f in fields.items() if not f['env_bits_equal'][e]]) for e in range(64))
        blocks.append(dict(block=block,initial_seed=row['initial_seed'],command_seed=row['command_seed'],
            raw_cell_paths=paths,raw_cell_sha256={m:ledger.hashes[p] for m,p in paths.items()},
            compared_case_pairs=64,all_declared_saved_arrays_bitwise_equal=bool(equal.all()),fields=fields))
        print(json.dumps(dict(event='INDEPENDENT_TRAJECTORY_COMPARISON',block=block,equal_case_pairs=int(equal.sum()))),flush=True)
        del arrays,a,b
    changed=ledger.recheck();require(not changed,'consumed input changed: '+repr(changed))
    equal=sum(c['declared_saved_arrays_bitwise_equal'] for c in cases)
    result=dict(status='PASS_CLOSED_INDEPENDENT_RECORDED_TRAJECTORY_COMPARISON',utc=now(),
        analysis_timing='post-observation requested independent confirmation, not preregistered outcome selection',
        compared_methods=['tight_reference','delay_reserve'],compared_case_pairs=192,equal_case_pairs=equal,
        all_declared_saved_arrays_bitwise_equal=equal==192,blocks=blocks,cases=cases,
        input_sha256=ledger.hashes,raw_hash_before_after_pass=True,parent_comparison_counts_used=False,
        declared_array_scope=list(FIELDS),full_unarchived_rigid_state_equality_claim=False,
        equality_implies_safety=False,new_random_windows=0,registered_primary_windows_remain=576,
        candidate_physical_status='REJECTED_BY_PARENT; this artifact verifies saved-array identity only',
        reserve_floor_saturation_independently_recomputed_here=False,
        frozen_primary_scorer_modified=False,policy_parameters_changed=False)
    write_owned('ASTRA_TRACKING_FINAL_TRAJECTORY_IDENTITY.json',result)
    print(result['status'],equal,flush=True)

if __name__=='__main__':main()
