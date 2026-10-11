"""Independent physics weight balance and reporting-toggle audit, not task acceptance."""
import hashlib
import json
from pathlib import Path

import numpy as np

H=Path(__file__).resolve().parent
P=H/'cpu_contact_calibration_v3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=P/'INDEPENDENT_CALIBRATION_V1.json';assert not target.exists()
    assert read(P/'EXECUTION_ACTUAL_EXIT_V1.json')['actual_outer_exit']==0
    r=read(H/'cpu-contact-calibration-v5_execution.json')
    assert r['status']=='complete' and len(r['jobs'])==1
    j=r['jobs'][0];assert j['actual_exit']==j['actual_native_exit']==0 and j['native_owned_process_reaped_and_absent']
    reg=read(P/'REGISTRATION_V1.json');raw=Path(j['out']);p=read(raw/'protocol.json')
    assert p['physics_device']=='cpu' and p['physics_steps']==reg['physics_steps']==480
    assert p['missing_api_fixture_unbound'] and len(p['sensors'])==2
    assert p['before_view_reporting_metadata'][0]['report_api'] is False
    assert p['before_view_reporting_metadata'][2]['threshold']==0.
    with np.load(raw/'native_fixture_stream.npz',allow_pickle=False) as z:
        disabled=z['contact_processing_disabled'].reshape(-1);np.testing.assert_array_equal(disabled,np.zeros(480,dtype=bool))
        matrix=z['matrix'].astype(np.float64);counts=z['counts'];vel=z['velocities'];pose=z['transforms']
        assert matrix.shape[1]==2 and pose.shape[1]==3
        n=np.linalg.norm(matrix,axis=-1).sum(axis=-1)
        # Both valid views are retained. The explicit zero-threshold view is preregistered as the weight oracle.
        index=p['sensors'].index('/World/ZeroThresholdReportAPI')
        control=H/'cpu_contact_calibration_v2/native_fixture_v2/native_fixture_stream.npz'
        assert read(H/'cpu_contact_calibration_v2/EXECUTION_ACTUAL_EXIT_V2.json')['actual_outer_exit']==0
        with np.load(control,allow_pickle=False) as previous:
            off=np.linalg.norm(previous['matrix'][:240,index].astype(np.float64),axis=-1).sum(axis=-1)
            zero_count=int(np.sum(previous['counts'][:240]))
            np.testing.assert_array_equal(previous['transforms'],pose)
            np.testing.assert_array_equal(previous['velocities'],vel)
        on=n[300:,index]
        mean=float(on.mean());positive=float(np.mean(on>.1))
        error=abs(mean-reg['expected_weight_N'])/reg['expected_weight_N']
        passed=bool(np.max(off)==0 and zero_count==0 and positive>=reg['positive_force_fraction_min'] and error<=reg['settled_mean_weight_relative_error_max'])
        metrics=dict(disabled_max_force_N=float(off.max()),disabled_valid_point_count=zero_count,
            enabled_settled_mean_force_N=mean,expected_weight_N=reg['expected_weight_N'],
            enabled_positive_force_fraction=positive,enabled_relative_weight_error=error,
            enabled_valid_point_count=int(np.sum(counts[300:,index])),all_fixture_z_range_m=[float(pose[:,:,2].min()),float(pose[:,:,2].max())])
    result=dict(status='PASS_POSITIVE_CPU_CONTACT_REPORT_CALIBRATION' if passed else 'FAIL_CPU_CONTACT_REPORT_CALIBRATION',
        calibration_pass=passed,metrics=metrics,native_fixture_sha256=sha(raw/'native_fixture_stream.npz'),
        actual_stage_sha256=sha(raw/'actual_stage.usda'),actual_native_wait0=True,all_owned_children_reaped=True,
        fresh_scene_reporting_enabled_before_reset=True,all480_actual_native_poses_and_velocities_equal_prior_control=True,invalid_no_api_sensor_not_given_fake_zero=True,
        root_cause_scope='Fresh-scene CPU contact processing enabled before reset yields positive weight-calibrated reports; the previous runtime toggle yielded all zeros. All480 native body poses and velocities are bit-identical to that control. This qualifies this reporting initialization fixture only, not all robot contact classes.',
        loaded_model_or_full_task_qualified=False,all82_robot_contact_classes_qualified=False,
        formal_holdout_count=0,fullSystem0_accepted=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(result['status'],metrics,flush=True)

if __name__=='__main__':main()
