"""All native parameters and original gates fixed except registered resolution."""
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
    baseline=H/f"fourhand_native_probe_v{reg['numerical_baseline_version']}";br=read(baseline/'REGISTRATION_V1.json')
    n=microsteps(reg);old_n=microsteps(br);assert n>old_n
    assert reg['physics_dt_s']*n==br['physics_dt_s']*old_n==reg['control_dt_s']==br['control_dt_s']
    fields=['maximum_controls','control_dt_s','position_target_FIFO_controls','lift_gate',
       'release_gate','scene_geometry_gate_m','all74_hard_violation_rad_le',
       'all74_velocity_exceedance_rad_s_le','retreat_endpoint_gate',
       'coupling_relation_tolerance_rad','coupling_velocity_relation_tolerance_rad_s',
       'nominal_size_m','object_mass_kg','support_posts','feedback_grip_settings',
       'interior_hand_target_settings','case','pose_variants','camera_views',
       'coupled_assets','coupled_reference_endpoints','path','pose_solutions',
       'calibrated_hand_poses','fixture','object_usd','prelift_load_reserve_fraction',
       'video_capture_every_controls','video_capture_Hz','camera_framing_gate',
       'solve_articulation_contact_last','enable_external_forces_every_iteration','native_solver_profile']
    for key in fields:assert reg[key]==br[key],key
    current_gate=dict(reg['grasp_settle_gate']);old_gate=dict(br['grasp_settle_gate'])
    assert current_gate.pop('physics_dt_s')==reg['physics_dt_s'] and old_gate.pop('physics_dt_s')==br['physics_dt_s']
    assert current_gate==old_gate
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out']);old=Path(read(baseline/'RESOURCE_EXECUTION_V1.json')['out'])
    assert read(raw/'native_scene_settings.json')==read(old/'native_scene_settings.json')
    assert read(raw/'native_solver_roots.json')==read(old/'native_solver_roots.json')
    keys=[]
    with np.load(raw/'native_parameters.npz') as x,np.load(old/'native_parameters.npz') as y:
        assert set(x.files)==set(y.files)
        for key in x.files:
            if key=='actual_physics_dt_s':
                assert float(x[key])==reg['physics_dt_s'] and float(y[key])==br['physics_dt_s']
            else:np.testing.assert_array_equal(x[key],y[key],err_msg=key);keys.append(key)
    assert read(raw/'native_mimic_identity.json')==read(old/'native_mimic_identity.json')
    x=read(raw/'native_contact_identity.json');y=read(old/'native_contact_identity.json')
    assert x.pop('physics_dt_s')==reg['physics_dt_s'] and y.pop('physics_dt_s')==br['physics_dt_s']
    assert x.pop('source_sha256')==sha(p/'native_all_body_contacts.py')
    assert y.pop('source_sha256')==sha(baseline/'native_all_body_contacts.py')
    assert x==y
    receipt=read(raw/'point_contact_receipts.json')
    assert receipt['substeps_per_control']==n and receipt['physics_events']==n*a['completed_controls']
    assert a['physics_substeps_per_control']==n and a['position_target_FIFO_controls']==br['position_target_FIFO_controls']==6
    paths=[Path(__file__),p/'REGISTRATION_V1.json',p/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
       baseline/'REGISTRATION_V1.json',raw/'native_scene_settings.json',old/'native_scene_settings.json',
       raw/'native_parameters.npz',old/'native_parameters.npz',raw/'native_solver_roots.json',
       old/'native_solver_roots.json',raw/'native_contact_identity.json',old/'native_contact_identity.json',
       p/'native_all_body_contacts.py',baseline/'native_all_body_contacts.py',raw/'point_contact_receipts.json']
    report=dict(status='PASS_NATIVE_ONLY_TIME_RESOLUTION_CHANGE_WITH_ORIGINAL_GATES_FIXED',version=version,
       baseline_version=reg['numerical_baseline_version'],physics_substeps_before=old_n,physics_substeps_after=n,
       physics_dt_before_s=br['physics_dt_s'],physics_dt_after_s=reg['physics_dt_s'],
       control_dt_fixed_s=reg['control_dt_s'],FIFO_delay_fixed_controls=6,
       identical_native_parameter_arrays=keys,only_native_parameter_difference='actual_physics_dt_s',
       all_original_acceptance_thresholds_unchanged=True,all_additional_microsteps_independently_audited=True,
       identical_registered_gate_and_scene_fields=fields,numerical_identity_is_not_model_qualification=True,
       fullSystem0_accepted=False,source_SHA256={str(f):sha(f) for f in paths})
    f=p/'INDEPENDENT_TIME_REFINEMENT_COUNTERFACTUAL_V1.json';assert not f.exists();f.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(keys),'fixed native arrays',n,'microsteps/control',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('version',type=int);main(ap.parse_args().version)
