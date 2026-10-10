"""Independent initial rigid-pose equivalence; no producer evaluator imports.

Quaternion distance uses normalized chord lengths, not the producer's relative
quaternion cross-product formula. Per-lane results are retained for all32.
"""
import datetime
import math
import numpy as np
from evidence_io import ARMS,require,equal_bits,exact


def validate_protocol(p):
    require(p['schema']=='astra.full74.root_restore_equivalence_protocol.v1','root protocol schema')
    require(p['root_position_max_error_m']==1e-7 and p['quaternion_orientation_max_error_rad']==1e-6 and
            p['quaternion_raw_norm_max_error']==8*float(np.finfo(np.float32).eps),'frozen root thresholds')
    require(p['root_velocity_requirement']==p['full74_q_qd_target_requirement']=='float32 bit exact' and
            p['all_camera_held_native27_bitwise_unchanged'] is True and
            p['native_physical_parameters_timestep_geometry_and_contact_oracles_unchanged'] is True and
            p['proposal_bytes_unchanged'] is True and p['native_prefix_steps_at_freeze']==0 and
            p['selected_states']==p['formal_holdout_states']==0 and p['physical_safety_certified'] is False and
            p['all_requirements_completed'] is False,'prospective root scope')
    return p


def protocol_binding(e,req,anchors,started_utc):
    ref=anchors['root_protocol']
    require(req['root_restore_equivalence_protocol']==ref,'request frozen root protocol binding')
    p=validate_protocol(e.js(ref['path'],ref['sha256']))
    frozen=datetime.datetime.fromisoformat(p['created_utc']);started=datetime.datetime.fromisoformat(started_utc)
    require(frozen.tzinfo is not None and started.tzinfo is not None and frozen<started,'root protocol must predate native launch')
    return p


def evaluate(actual,requested,actual_velocity,requested_velocity,protocol):
    validate_protocol(protocol)
    for a,b,shape,label in ((actual,requested,(32,7),'pose'),(actual_velocity,requested_velocity,(32,6),'velocity')):
        require(isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape==shape and
                a.dtype==b.dtype==np.dtype('float32'),'root '+label+' shape/type')
        require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite root '+label)
    exact(actual_velocity,requested_velocity,'root velocity remains bitwise')
    lanes=[]
    for i in range(32):
        delta=[float(actual[i,j])-float(requested[i,j]) for j in range(3)]
        position=max(map(abs,delta));require(position<=protocol['root_position_max_error_m'],'root position error')
        aq=list(map(float,actual[i,3:]));bq=list(map(float,requested[i,3:]))
        an=math.sqrt(math.fsum(x*x for x in aq));bn=math.sqrt(math.fsum(x*x for x in bq))
        require(an>0 and bn>0,'zero root quaternion')
        norm=max(abs(an-1),abs(bn-1));require(norm<=protocol['quaternion_raw_norm_max_error'],'root quaternion norm error')
        a=[x/an for x in aq];b=[x/bn for x in bq]
        # q and -q denote the same rotation; choose the shorter normalized chord.
        minus=math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))
        plus=math.sqrt(math.fsum((x+y)**2 for x,y in zip(a,b)))
        angle=4*math.atan2(min(minus,plus),max(minus,plus))
        require(angle<=protocol['quaternion_orientation_max_error_rad'],'root orientation error')
        lanes.append(dict(proposal_id=i,position_delta_m=delta,position_max_abs_error_m=position,
            actual_quaternion_xyzw=aq,requested_quaternion_xyzw=bq,actual_quaternion_norm=an,requested_quaternion_norm=bn,
            quaternion_norm_max_error=norm,orientation_error_rad=angle,
            position_bitwise_equal=equal_bits(actual[i,:3],requested[i,:3]),
            full_pose_bitwise_equal=equal_bits(actual[i],requested[i]),velocity_bitwise_equal=True))
    metrics=dict(status='PASS_RECORDED_ROOT_POSE_EQUIVALENCE_ONLY',position_bitwise_equal=equal_bits(actual[:,:3],requested[:,:3]),
        position_max_abs_error_m=max(x['position_max_abs_error_m'] for x in lanes),position_limit_m=protocol['root_position_max_error_m'],
        velocity_bitwise_equal=True,full_pose_bitwise_equal=equal_bits(actual,requested),
        # This producer diagnostic specifically uses float32 component subtraction.
        quaternion_component_max_abs_error=float(np.max(np.abs(actual[:,3:]-requested[:,3:]))),
        quaternion_norm_max_error=max(x['quaternion_norm_max_error'] for x in lanes),
        orientation_max_error_rad=max(x['orientation_error_rad'] for x in lanes),
        orientation_limit_rad=protocol['quaternion_orientation_max_error_rad'],quaternion_norm_error_limit=protocol['quaternion_raw_norm_max_error'],
        camera_held_native27_bitwise_requirement_unchanged=True,hidden_solver_state_restored=False,physical_safety_certified=False)
    return dict(summary=metrics,lanes=lanes)


def verify_report(computed,reported):
    expected=computed['summary'];require(isinstance(reported,dict) and set(reported)==set(expected),'root report inventory')
    for key,value in expected.items():
        other=reported[key]
        if type(value) is float:
            require(type(other) in (int,float) and math.isfinite(other),'root report finite scalar '+key)
            # Arithmetic reconciliation only. Pose acceptance above uses exact
            # frozen bounds without this tolerance or any global-error claim.
            tol=64*float(np.finfo(np.float64).eps)*max(1.,abs(value)) if key in ('quaternion_norm_max_error','orientation_max_error_rad') else 0.
            require(abs(other-value)<=tol,'root report metric mismatch '+key)
        else:require(type(other) is type(value) and other==value,'root report claim mismatch '+key)


def audit_roots(fields,initial,protocol,reported):
    require(isinstance(reported,dict) and set(reported)==set(ARMS),'all4 root reports required')
    results={}
    for arm in ARMS:
        result=evaluate(fields[arm+'_root_xyzw'][0],initial[arm+'_root'],fields[arm+'_root_velocity'][0],
                        initial[arm+'_root_velocity'],protocol)
        verify_report(result,reported[arm]);results[arm]=result
    return dict(status='INDEPENDENT_INITIAL_ROOT_EQUIVALENCE_ONLY',arms=results,checked_lanes_per_arm=32,
        quaternion_method='normalized antipodal-invariant chord geodesic; float64 math.fsum',
        report_arithmetic_tolerance=64*float(np.finfo(np.float64).eps),
        root_acceptance_thresholds_unrelaxed=True,prior_V7_failed_not_reclassified=True,
        q74_qd74_targets_velocity_bitwise_unchanged=True,camera_held27_bitwise_unchanged=True,
        hidden_solver_state_restored=False,physical_safety_certified=False,global_error_bound='UNKNOWN')
