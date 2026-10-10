"""Bind precisely 24 motor damping changes; retain every original gate."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from microstep_contract_v1 import microsteps
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(version):
    p=H/f'fourhand_native_probe_v{version}';reg=read(p/'REGISTRATION_V1.json')
    a=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json');assert a['record_integrity_verified']
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    baseline=H/f"fourhand_native_probe_v{reg['drive_policy_baseline_version']}";br=read(baseline/'REGISTRATION_V1.json')
    assert microsteps(reg)==microsteps(br)
    fields=['maximum_controls','control_dt_s','physics_dt_s','position_target_FIFO_controls','lift_gate',
        'release_gate','scene_geometry_gate_m','all74_hard_violation_rad_le','all74_velocity_exceedance_rad_s_le',
        'retreat_endpoint_gate','grasp_settle_gate','coupling_relation_tolerance_rad',
        'coupling_velocity_relation_tolerance_rad_s','nominal_size_m','object_mass_kg','support_posts',
        'feedback_grip_settings','interior_hand_target_settings','case','pose_variants','camera_views',
        'coupled_assets','coupled_reference_endpoints','path','pose_solutions','calibrated_hand_poses',
        'fixture','object_usd','prelift_load_reserve_fraction','video_capture_every_controls','video_capture_Hz',
        'camera_framing_gate','solve_articulation_contact_last','enable_external_forces_every_iteration','native_solver_profile']
    for key in fields:assert reg[key]==br[key],key
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out']);old=Path(read(baseline/'RESOURCE_EXECUTION_V1.json')['out'])
    assert read(raw/'native_scene_settings.json')==read(old/'native_scene_settings.json')
    assert read(raw/'native_solver_roots.json')==read(old/'native_solver_roots.json')
    assets=read(reg['coupled_assets']);diagnosis=read(reg['motor_damping_diagnosis'])
    by_arm={r['arm']:r for r in diagnosis['rows']};keys=[];changes=[]
    with np.load(raw/'native_parameters.npz') as x,np.load(old/'native_parameters.npz') as y:
        assert set(x.files)==set(y.files)
        for key in x.files:
            if key.endswith('_damping'):
                arm=key[:-len('_damping')];expected=y[key].copy();names=x[arm+'_joint_names'].tolist()
                specified=reg['hand_motor_damping_by_joint'][arm]
                assert set(specified)==set(assets['assets'][arm]['independent_motor_joint_names'])
                for name,value in specified.items():
                    i=names.index(name);d=by_arm[arm];j=d['motors'].index(name)
                    assert value==d['estimated_critical_D_max'][j] and 0<value<float(y[key][0,i])
                    expected[0,i]=value
                    changes.append(dict(arm=arm,motor=name,before=float(y[key][0,i]),after=float(expected[0,i])))
                np.testing.assert_array_equal(x[key],expected,err_msg=key)
            else:np.testing.assert_array_equal(x[key],y[key],err_msg=key);keys.append(key)
    assert len(changes)==24
    assert read(raw/'native_mimic_identity.json')==read(old/'native_mimic_identity.json')
    x=read(raw/'native_contact_identity.json');y=read(old/'native_contact_identity.json')
    assert x.pop('source_sha256')==sha(p/'native_all_body_contacts.py')
    assert y.pop('source_sha256')==sha(baseline/'native_all_body_contacts.py')
    assert x==y
    paths=[Path(__file__),p/'REGISTRATION_V1.json',p/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
        baseline/'REGISTRATION_V1.json',raw/'native_scene_settings.json',old/'native_scene_settings.json',
        raw/'native_parameters.npz',old/'native_parameters.npz',raw/'native_solver_roots.json',
        old/'native_solver_roots.json',raw/'native_contact_identity.json',old/'native_contact_identity.json',
        Path(reg['motor_damping_diagnosis'])]
    report=dict(status='PASS_NATIVE_ONLY_24_MOTOR_CONTROLLER_DAMPING_CHANGES_ORIGINAL_GATES_FIXED',
        baseline_version=reg['drive_policy_baseline_version'],changes=changes,
        unchanged_native_arrays=keys,all_original_gate_and_scene_fields_identical=fields,
        all_other_motor_arm_passive_drives_and_parameters_unchanged=True,
        all_original_acceptance_thresholds_unchanged=True,hardware_compliance_identified=False,
        controller_identity_is_not_model_or_task_qualification=True,fullSystem0_accepted=False,
        source_SHA256={str(f):sha(f) for f in paths})
    target=p/'INDEPENDENT_MOTOR_DAMPING_COUNTERFACTUAL_V1.json';assert not target.exists()
    target.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],len(changes),'changes',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('version',type=int);main(ap.parse_args().version)
