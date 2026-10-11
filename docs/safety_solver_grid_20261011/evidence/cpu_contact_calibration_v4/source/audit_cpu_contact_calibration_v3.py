"""Qualify continuous awake-fixture reporting while retaining sleeping-state dropout."""
import hashlib
import json
from pathlib import Path

import numpy as np

H=Path(__file__).resolve().parent
P=H/'cpu_contact_calibration_v4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=P/'INDEPENDENT_CALIBRATION_V1.json';assert not target.exists()
    assert read(P/'EXECUTION_ACTUAL_EXIT_V1.json')['actual_outer_exit']==0
    execution=read(H/'cpu-contact-calibration-v6_execution.json');assert execution['status']=='complete'
    job=execution['jobs'][0];assert job['actual_exit']==job['actual_native_exit']==0 and job['native_owned_process_reaped_and_absent']
    raw=Path(job['out']);protocol=read(raw/'protocol.json');reg=read(P/'REGISTRATION_V1.json')
    assert protocol['physics_device']=='cpu' and protocol['physics_steps']==reg['physics_steps']==480
    assert len(protocol['sensors'])==3 and protocol['missing_api_fixture_unbound']
    with np.load(raw/'native_fixture_stream.npz',allow_pickle=False) as d:
        np.testing.assert_array_equal(d['contact_processing_disabled'].reshape(-1),np.zeros(480,dtype=bool))
        forces=np.linalg.norm(d['matrix'].astype(np.float64),axis=-1).sum(axis=-1)
        idx=protocol['sensors'].index('/World/ZeroThresholdAwakeReportAPI')
        sample=forces[300:,idx]
        mean=float(sample.mean());fraction=float(np.mean(sample>.1));error=abs(mean-reg['expected_weight_N'])/reg['expected_weight_N']
        passed=bool(fraction>=reg['positive_force_fraction_min'] and error<=reg['settled_mean_weight_relative_error_max'])
        metrics=dict(enabled_settled_mean_force_N=mean,expected_weight_N=reg['expected_weight_N'],
            enabled_positive_force_fraction=fraction,enabled_relative_weight_error=error,
            enabled_settled_valid_point_count=int(d['counts'][300:,idx].sum()))
        profiles=[]
        for i,name in enumerate(protocol['sensors']):
            profiles.append(dict(sensor=name,positive_report_steps=int(np.sum(forces[:,i]>.1)),
                last180_mean_force_N=float(forces[300:,i].mean()),last180_valid_point_count=int(d['counts'][300:,i].sum())))
    result=dict(status='PASS_CONTINUOUS_AWAKE_CPU_CONTACT_REPORT_CALIBRATION' if passed else 'FAIL_AWAKE_CPU_CONTACT_REPORT_CALIBRATION',
        calibration_pass=passed,metrics=metrics,profiles=profiles,native_fixture_sha256=sha(raw/'native_fixture_stream.npz'),
        actual_stage_sha256=sha(raw/'actual_stage.usda'),actual_native_wait0=True,all_owned_children_reaped=True,
        initialized_reporting_before_first_reset=True,zero_reporting_threshold_explicit=True,
        awake_fixture_sleep_threshold=0.,actual_four_arm_asset_sleep_parameters_changed=False,
        sleeping_fixture_dropout_retained=True,sleeping_state_contact_observation_qualified=False,
        all82_robot_contact_classes_qualified=False,loaded_model_or_full_task_qualified=False,
        formal_holdout_count=0,fullSystem0_accepted=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(result['status'],metrics,flush=True)

if __name__=='__main__':main()
