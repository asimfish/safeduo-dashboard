"""Native singlefactor joint-stop diagnostic; no configuration adoption."""
from pathlib import Path
import json
import numpy as np
from analyse_prefix512_v2 import oracle, sha

H=Path(__file__).resolve().parent
R=Path(json.loads((H/'REGISTRATION_HAND_VALID_V2.json').read_text())['raw'])
ARMS=('F_L','F_R','U_L','U_R')


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def main():
    execution=json.loads((H/'solver_prefix64_v2_execution.json').read_text())
    assert execution['status']=='complete' and len(execution['jobs'])==4
    assert all(j['status']=='complete' and j['actual_exit']==0 for j in execution['jobs'])
    reference=load(R/'prefix_batch_0_v5/prefix_inputs.npz')
    reference_output=load(R/'prefix_batch_0_v5/prefix_result.npz')
    rows=[];first_inputs=None
    for count in [8,16,32,64]:
        root=R/f'solver_prefix_iters{count}_v2'
        params=load(root/'prefix_inputs.npz')
        output=load(root/'prefix_result.npz')
        protocol=json.loads((root/'prefix_protocol.json').read_text())
        roots=json.loads((root/'actual_solver_roots.json').read_text())
        assert protocol['status']=='complete' and protocol['position_iterations']==count
        assert len(roots)==4 and all(r['position_iterations']==count and r['velocity_iterations']==0 for r in roots)
        for arm in ARMS:
            for field in ['initial_q','initial_qd','full_hold_target','hard_limits','vmax']:
                assert np.array_equal(params[arm+'_'+field],reference[arm+'_'+field]),'input binding '+arm+' '+field
            if first_inputs is not None:
                for field in ['kp','kd','effort','armature']:
                    assert np.array_equal(params[arm+'_'+field],first_inputs[arm+'_'+field])
        if first_inputs is None:
            first_inputs=params
        force_oracle=oracle(root,output,params)
        geometry=output['all_raw_geometry']
        assert geometry.shape==(12,64,9021)
        force=output['scalar_normal_max_N']
        hard=output['supplementary_hard_limit_violation_rad']
        vel=output['supplementary_velocity_limit_violation_rad_s']
        admitted=output['admitted']
        hard_bad=hard.max(0)>1e-5
        full_prefix=admitted & ~hard_bad & (vel.max(0)<=1e-5)
        rows.append(dict(position_iterations=count,native_exit=0,all12event_raw_point_oracle=force_oracle,
            all64hard_breaches=int(hard_bad.sum()),max_all74hard_overshoot_rad=float(hard.max()),
            geometry_qualified=int((geometry.min((0,2))>=0).sum()),point_force_qualified=int((force.max(0)<=.1).sum()),
            primary_qualified=int(admitted.sum()),all74_hard_and_velocity_and_primary_qualified=int(full_prefix.sum()),
            all64force_peak_N=float(force.max()),all64minimum_raw_gap_m=float(geometry.min()),
            perlane_hard_max_rad=hard.max(0).tolist(),perlane_primary_qualified=admitted.tolist(),
            perlane_full_prefix_qualified=full_prefix.tolist(),input_sha256=sha(root/'prefix_inputs.npz'),
            output_sha256=sha(root/'prefix_result.npz')))
        if count==8:
            reproduction=dict(raw_geometry_max_error_m=float(abs(geometry-reference_output['all_raw_geometry']).max()),
                point_force_max_error_N=float(abs(force-reference_output['scalar_normal_max_N']).max()),
                hard_violation_max_error_rad=float(abs(hard-reference_output['supplementary_hard_limit_violation_rad']).max()),
                primary_admission_exact=bool(np.array_equal(admitted,reference_output['admitted'])))
    result=dict(status='CLOSED_NATIVE_POSITION_ITERATION64STATE_PREFIX_DIAGNOSTIC',source_states=64,controls_each=6,
        single_varied_factor='all4native articulation positioniterations',all_inputs74q_qd_handtargets_limits_gains_exact=True,
        actual4root_solver_readbacks_verified=True,reference8_reproduction=reproduction,settings=rows,
        native_jobs_closed_actual_exit0=True,all4_independent_rawpoint_oracles_pass=True,
        safety_configuration_adopted=False,full_system0_accepted=False,
        scope='One sourcebatch diagnostic with all64originalstates, including primaryprefix rejects; not independentholdout or full512coverage')
    (H/'SOLVER_PREFIX_DIAGNOSTIC_RESULT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],[{k:r[k] for k in ['position_iterations','all64hard_breaches','max_all74hard_overshoot_rad','primary_qualified','all74_hard_and_velocity_and_primary_qualified']} for r in rows])


if __name__=='__main__':
    main()
