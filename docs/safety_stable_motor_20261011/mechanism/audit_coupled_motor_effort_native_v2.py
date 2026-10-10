"""Independently audit backward-Euler motor effort, gravity and passive zeros."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from audit_coupled_hand_targets_native_v1 import matrix

H = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(version):
    p=H/f'fourhand_native_probe_v{version}'
    reg=read(p/'REGISTRATION_V1.json')
    audit=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    assert audit['record_integrity_verified']
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out'])
    params=np.load(raw/'native_parameters.npz',allow_pickle=False)
    meta=read(reg['coupled_assets'])
    sources={}
    rows=[]
    dt=reg['physics_dt_s']
    for arm,asset in meta['assets'].items():
        names=params[arm+'_joint_names'].tolist()
        hand=asset['hand_joint_names']
        handids=[names.index(name) for name in hand]
        roots,A,b=matrix(hand,asset['original_URDF'])
        rootids=[names.index(name) for name in roots]
        followers=[names.index(row['slave']) for row in asset['relations']]
        projection=np.eye(len(names))
        projection[handids]=0
        projection[np.ix_(rootids,handids)]=A.T
        np.testing.assert_allclose(projection,params[arm+'_motor_effort_projection'],atol=1e-12,rtol=0)
        cfg=reg['hand_motor_effort_controller'][arm]
        mass=np.array([cfg[name]['inertia_estimate_kg_m2'] for name in roots])
        stiffness=np.array([cfg[name]['nominal_K_Nm_rad'] for name in roots])
        damping=np.array([cfg[name]['nominal_D_Nm_s_rad'] for name in roots])
        assert (mass>0).all() and (stiffness==5).all() and (damping==.5).all()
        assert np.all(params[arm+'_stiffness'][0,handids]==0) and np.all(params[arm+'_damping'][0,handids]==0)
        prior_q=prior_v=None
        events=0
        maximum_error=0.
        for chunk in read(raw/'physics_chunks.json'):
            file=raw/chunk['path']
            assert sha(file)==chunk['sha256']
            sources[str(file)]=sha(file)
            with np.load(file,allow_pickle=False) as data:
                q=data[arm+'_motor_control_q_before']
                v=data[arm+'_motor_control_qd_before']
                after_q=data[arm+'_native_q']
                after_v=data[arm+'_native_qd']
                assert np.isfinite(q).all() and np.isfinite(v).all()
                if prior_q is not None:
                    np.testing.assert_array_equal(q[0],prior_q)
                    np.testing.assert_array_equal(v[0],prior_v)
                np.testing.assert_array_equal(q[1:],after_q[:-1])
                np.testing.assert_array_equal(v[1:],after_v[:-1])
                prior_q,prior_v=after_q[-1],after_v[-1]
                targets=data[arm+'_actual_full_position_target'][...,rootids].astype(np.float64)
                before_q=q[...,rootids].astype(np.float64)
                before_v=v[...,rootids].astype(np.float64)
                next_v=(mass*before_v+dt*stiffness*(targets-before_q))/(mass+dt*damping+dt*dt*stiffness)
                # Use the force-form equation independent of the producer's
                # inertia*(v_next-v)/dt expression.
                torque=stiffness*(targets-before_q-dt*next_v)-damping*next_v
                pd=np.zeros_like(data[arm+'_gravity_compensation_raw'])
                pd[...,rootids]=torque.astype(pd.dtype)
                np.testing.assert_allclose(pd,data[arm+'_software_hand_motor_PD_preclip'],atol=1e-7,rtol=0)
                gravity=data[arm+'_gravity_compensation_raw']@projection.T
                if not bool(params[arm+'_gravity_compensation_enabled']):
                    gravity=np.zeros_like(gravity)
                expected=np.clip(gravity.astype(pd.dtype)+pd,-params[arm+'_max_force'],params[arm+'_max_force'])
                submitted=data[arm+'_extra_effort']
                native=data[arm+'_native_external_actuation_effort']
                np.testing.assert_allclose(expected,submitted,atol=1e-6,rtol=0)
                np.testing.assert_array_equal(submitted,native)
                assert np.max(abs(native[...,followers]))==0
                maximum_error=max(maximum_error,float(abs(expected-native).max()))
                events+=len(q)
        assert events==audit['physics_events']
        rows.append(dict(arm=arm,physics_events=events,independent_hand_motors=6,
                         passive_followers=6,maximum_binding_error_Nm=maximum_error,
                         pre_state_matches_actual_previous_physics_event=True,
                         passive_user_effort_exactly_zero=True))
        sources[asset['original_URDF']]=sha(asset['original_URDF'])
    for file in [Path(__file__),p/'REGISTRATION_V1.json',raw/'native_parameters.npz']:
        sources[str(file)]=sha(file)
    report=dict(status='PASS_INDEPENDENT_STABLE_MOTOR_EFFORT_INPUT_GRAVITY_CAPS_AND_NATIVE_BINDING',
                version=version,rows=rows,source_SHA256=sources,physics_events=audit['physics_events'],
                force_form_oracle_independent_of_producer_implementation=True,
                physical_or_model_or_task_qualification_not_claimed=True,fullSystem0_accepted=False)
    target=p/'INDEPENDENT_COUPLED_MOTOR_EFFORT_V2.json'
    assert not target.exists()
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],audit['physics_events'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('version',type=int)
    main(parser.parse_args().version)
