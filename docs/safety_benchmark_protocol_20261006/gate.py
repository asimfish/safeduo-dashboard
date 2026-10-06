"""Readiness declaration gate; does not certify evidence contents or robot safety."""
COMMON=('physical_input_validity','data_provenance','negative_endpoint_controls','independent_hazard_oracle',
        'repeatability','reachable_reference_atlas','development_test_separation','fair_baselines','frozen_candidate','statistical_power_plan')
TASK=('actual_object_input_binding','reachable_layouts','real_grasp_release','task_success_negative_controls','synchronized_camera_binding')
ROBUST=('frozen_perturbation_domain','fault_injection_controls')
def registration_gate(category,evidence):
    if category not in ('random_motion','real_task','robustness'):raise ValueError('unknown category')
    required=COMMON+(TASK if category!='random_motion' else ())+(ROBUST if category=='robustness' else ())
    missing=[]
    for key in required:
        item=evidence.get(key)
        if not isinstance(item,dict) or item.get('status')!='PASS' or not item.get('evidence_path') or not item.get('sha256'):
            missing.append(key)
    return dict(category=category,status='READY_TO_REGISTER' if not missing else 'NOT_READY_TO_REGISTER',missing=missing,
                physical_safety_certified=False,scope='checks declared evidence completeness only; actual receipts need source/oracle readback')
