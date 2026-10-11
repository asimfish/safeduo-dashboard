"""Validate the prospective single numerical factor, not safety acceptance."""
from pathlib import Path
import hashlib
import json
import numpy as np

H=Path(__file__).resolve().parent
P=H/'fourhand_native_probe_v57'
B=H/'fourhand_native_probe_v55'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    target=P/'INDEPENDENT_CONTACT_DISTURBANCE_COUNTERFACTUAL_V2.json'
    assert not target.exists()
    plan=read(P/'COUNTERFACTUAL_AUDIT_PLAN_V1.json')
    assert plan['registration_sha256']==sha(P/'REGISTRATION_V1.json')
    actual=read(P/'ACTUAL_COMPLETION_EXIT_V57.json');assert actual['actual_native_exit']==actual['actual_finite_exit']==actual['actual_launcher_subprocess_exit']==0
    assert read(P/'INDEPENDENT_CAPTURE_AUDIT_V1.json')['record_integrity_verified']
    reg,old=read(P/'REGISTRATION_V1.json'),read(B/'REGISTRATION_V1.json')
    assert reg['numerical_baseline_version']==55
    assert reg['hand_target_update_period']=='physics_microstep' and reg['hand_target_update_dt_s']==reg['physics_dt_s']
    assert read(P/'INDEPENDENT_COUPLED_HAND_TARGETS_V3.json')['status']=='PASS_INDEPENDENT_ALL_MICROSTEP_COUPLED_NATIVE_TARGET_BINDING'
    assert read(P/'INDEPENDENT_FULL_MOTOR_EFFORT_V3.json')['status']=='PASS_INDEPENDENT_FULL50_CONTACT_DISTURBANCE_PREVIOUS_ACTUAL_TORQUE_GRAVITY_CAPS_SLAVE_ZEROS'
    assert reg['native_solver_profile']==dict(position_iterations=32,velocity_iterations=128)
    assert old['native_solver_profile']==dict(position_iterations=32,velocity_iterations=128)
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    fields=['maximum_controls','control_dt_s','physics_dt_s','physics_substeps_per_control',
        'position_target_FIFO_controls','lift_gate','release_gate','scene_geometry_gate_m',
        'all74_hard_violation_rad_le','all74_velocity_exceedance_rad_s_le','retreat_endpoint_gate',
        'coupling_relation_tolerance_rad','coupling_velocity_relation_tolerance_rad_s',
        'nominal_size_m','object_mass_kg','support_posts','feedback_grip_settings',
        'interior_hand_target_settings','case','pose_variants','camera_views','coupled_assets',
        'coupled_reference_endpoints','path','pose_solutions','calibrated_hand_poses','fixture',
        'object_usd','prelift_load_reserve_fraction','video_capture_every_controls','video_capture_Hz',
        'camera_framing_gate','grasp_settle_gate','solve_articulation_contact_last',
        'enable_external_forces_every_iteration','preshape_hand_error_max_rad','full_motor_effort_controller','hand_target_update_period','hand_target_update_dt_s','full_motor_native_PD_exactly_zero','original_native_armature_used_in_controller','full_motor_mass_relative_asymmetry_tolerance']
    for field in fields:assert reg[field]==old[field],field
    raw=Path(read(P/'RESOURCE_EXECUTION_V1.json')['out']);before=Path(read(B/'RESOURCE_EXECUTION_V1.json')['out'])
    with np.load(raw/'native_parameters.npz') as x,np.load(before/'native_parameters.npz') as y:
        assert set(x.files)==set(y.files) and len(x.files)==77
        for key in x.files:np.testing.assert_array_equal(x[key],y[key],err_msg=key)
        keys=x.files
    for filename in ['native_scene_settings.json','native_mimic_identity.json','native_contact_identity.json']:
        assert read(raw/filename)==read(before/filename),filename
    roots={r['path']:r for r in read(raw/'native_solver_roots.json')}
    previous={r['path']:r for r in read(before/'native_solver_roots.json')}
    assert set(roots)==set(previous) and len(roots)==4
    for path in roots:
        assert previous[path]['position_iterations']==roots[path]['position_iterations']==32
        assert previous[path]['velocity_iterations']==roots[path]['velocity_iterations']==128
    caps=read(raw/'native_scene_iteration_caps_v1.json')['scenes']
    assert caps==read(before/'native_scene_iteration_caps_v1.json')['scenes']
    assert all(c['min_position']<=32<=c['max_position'] and c['min_velocity']<=128<=c['max_velocity'] for c in caps)
    sources=[Path(__file__),P/'COUNTERFACTUAL_AUDIT_PLAN_V1.json',P/'REGISTRATION_V1.json',B/'REGISTRATION_V1.json',
        raw/'native_parameters.npz',before/'native_parameters.npz',raw/'native_solver_roots.json',before/'native_solver_roots.json',
        raw/'native_scene_iteration_caps_v1.json',before/'native_scene_iteration_caps_v1.json']
    target.write_text(json.dumps(dict(status='PASS_SINGLE_HAND_DISTURBANCE_PREDICTION_FACTOR_ALL77_ARRAYS_AND_ORIGINAL_GATES_FIXED',
        actual_outer_exit=actual['actual_outer_exit'],posthoc_transport_bounded_audit=True,original_predeclared_v1_transport_guard_not_passed=True,baseline=55,candidate=57,hand_target_cadence_before='physics_microstep',hand_target_cadence_after='physics_microstep',position_iterations_fixed=32,velocity_iterations_fixed=128,
        identical_native_parameter_keys=keys,identical_original_gate_fields=fields,
        requested_and_USD_readback_profile_verified=True,exact_GPU_internal_execution_count_not_certified=True,
        numerical_parity_does_not_qualify_loaded_model_or_task=True,candidate_adopted=False,formal_holdout_count=0,fullSystem0_accepted=False,
        source_sha256={str(f):sha(f) for f in sources}),indent=2)+'\n')
    print('NATIVE57_SCIENTIFIC56_SINGLE_HAND_DISTURBANCE_FACTOR_ALL77_PARAMETERS_AND_ALL_ORIGINAL_GATES_PASS',flush=True)


if __name__=='__main__':main()
