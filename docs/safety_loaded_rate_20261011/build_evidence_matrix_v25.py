"""Reconcile closed60, recovered execution receipts and software-only object tests."""
from pathlib import Path
import copy
import datetime
import hashlib
import json

H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=H/'full_requirements_v1/EVIDENCE_MATRIX_V25.json';assert not out.exists()
    prior=H/'full_requirements_v1/EVIDENCE_MATRIX_V24.json';old=read(prior);d=copy.deepcopy(old)
    p=H/'fourhand_native_probe_v60';actual=read(p/'ACTUAL_COMPLETION_EXIT_V60.json')
    for key in ['actual_outer_exit','actual_native_exit','actual_finite_exit']:
        assert type(actual[key]) is int and actual[key]==0
    capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    assert capture['record_integrity_verified'] and not model['model_window_pass']
    prerequisite=read(p/'LOAD_ONLY_QUALIFICATION_PREREQUISITES_V1.json')
    replay=read(H/'object_input_fault_replay_v2/RESULT_V1.json')
    replay_exit=read(H/'object_input_fault_replay_v2/REPLAY_ACTUAL_EXIT_V1.json')
    assert replay_exit['actual_outer_exit']==0
    current=dict(native_version=60,actual_outer_exit=0,outer_tool_receipt_unknown=False,
        controls=capture['completed_controls'],physics_events=capture['physics_events'],
        original_position_error_rad=model['maximum_q_residual_rad'],original_velocity_error_rad_s=model['maximum_qd_residual_rad_s'],
        original_model_window_pass=model['model_window_pass'],physical_gates_pass=capture['physical_gates_pass'],
        maximum_hand_normal_N=capture['maximum_hand_payload_normal_N'],four_hand_contact_s=capture['measured_stage_windows_s']['four_hand_close'],
        stable_lift_s=capture['measured_stage_windows_s']['stable_lift'],fulltask_pass=False,
        loaded_only_diagnostic_scope=True,canonical_qualified=False,formal_holdout_count=0,learned_system0_tested=False,
        matched_clock_zero_force_is_uncontacted_window=True)
    for row in d['rows']:
        if row['id'] in ['grasp','cooperative_task','native_oracle','visual_delivery']:
            row['measured']=dict(prior_revision24=row['measured'],revision25=current)
            row['status']='FAIL_NATIVE60_ORIGINAL_VELOCITY_MODEL_GATE_FULLTASK_AND_FORMAL_PENDING'
        if row['id']=='objects':
            row['measured']=dict(prior_revision24=row['measured'],revision25_software_fault_replay=replay,
                runtime_adopted=False,learned_actor_object_input_ABI_changed=False,physical_stop_verified=False)
            row['status']='PARTIAL_SOFTWARE_OBJECT_INPUT_CONTRACT_NATIVE_AND_ACTOR_INTEGRATION_PENDING'
        row['joint_requirement_accepted']=False
    for before,after in zip(old['rows'],d['rows']):
        for field in ['id','requirement','gate']:assert before[field]==after[field]
    assert len(d['rows'])==9
    recovered57=H/'fourhand_native_probe_v57/ORIGINAL_OUTER_ACTUAL_EXIT_RECOVERY_V1.json'
    recovered55=H/'fourhand_native_probe_v55/AUXILIARY_ACTUAL_EXITS_RECOVERY_V1.json'
    d.update(revision=25,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_native_excluded=[61,62],
        fullSystem0_accepted=False,every_required_row_jointly_accepted=False,formal_holdout_acceptance_count=0,
        latest_development_frontier=current,execution_receipt_supplements={str(p):read(p) for p in [recovered57,recovered55]},
        no_independent_native_random_input_added=True)
    files=[Path(__file__),prior,p/'ACTUAL_COMPLETION_EXIT_V60.json',p/'INDEPENDENT_CAPTURE_AUDIT_V1.json',
           p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json',p/'LOAD_ONLY_QUALIFICATION_PREREQUISITES_V1.json',
           H/'object_input_fault_replay_v2/RESULT_V1.json',H/'object_input_fault_replay_v2/REPLAY_ACTUAL_EXIT_V1.json',recovered57,recovered55]
    d['source_sha256'].update({str(f):sha(f) for f in files})
    out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');print('MATRIX25_ORIGINAL9_ALL_JOINT_FALSE_CLOSED60',flush=True)


if __name__=='__main__':main()
