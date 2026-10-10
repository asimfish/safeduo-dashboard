"""Bind software motor control to unchanged physical parameters and gates."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(version):
    p=H/f'fourhand_native_probe_v{version}'
    reg=read(p/'REGISTRATION_V1.json')
    for path,digest in reg['source_sha256'].items():
        assert sha(path)==digest,path
    baseline=H/f"fourhand_native_probe_v{reg['stable_effort_baseline_version']}"
    oldreg=read(baseline/'REGISTRATION_V1.json')
    changed={'attempt_version','scientific_candidate_version','registered_utc',
             'change_scope','source_sha256','independent_motor_effort_auditor',
             'independent_velocity_iteration_auditor'}
    for key,value in oldreg.items():
        if key not in changed:
            assert reg[key]==value,key
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out'])
    old=Path(read(baseline/'RESOURCE_EXECUTION_V1.json')['out'])
    for name in ['native_scene_settings.json','native_mimic_identity.json',
                 'native_contact_identity.json','native_solver_roots.json']:
        assert read(raw/name)==read(old/name),name
    meta=read(reg['coupled_assets'])
    identical=[]
    altered=[]
    with np.load(raw/'native_parameters.npz',allow_pickle=False) as x,np.load(old/'native_parameters.npz',allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        gainkeys={arm+suffix for arm in meta['assets'] for suffix in ['_stiffness','_damping']}
        for key in x.files:
            if key in gainkeys:
                arm=key.split('_stiffness')[0].split('_damping')[0]
                names=y[arm+'_joint_names'].tolist()
                ids=[names.index(n) for n in meta['assets'][arm]['independent_motor_joint_names']]
                wanted=y[key].copy()
                expected=5. if key.endswith('_stiffness') else .5
                assert np.all(wanted[0,ids]==expected),key
                wanted[0,ids]=0.
                np.testing.assert_array_equal(x[key],wanted,err_msg=key)
                altered.append(key)
            else:
                np.testing.assert_array_equal(x[key],y[key],err_msg=key)
                identical.append(key)
    report=dict(status='PASS_ONLY24_NATIVE_MOTOR_DRIVES_REPLACED_WITH_REGISTERED_SOFTWARE_EFFORT',
                version=version,baseline_version=reg['stable_effort_baseline_version'],
                identical_native_parameter_arrays=identical,changed_native_gain_arrays=altered,
                original_model_physical_and_fulltask_gates_unchanged=True,
                original_mass_inertia_geometry_materials_effortcaps_unchanged=True,
                identity_is_not_model_or_task_qualification=True,fullSystem0_accepted=False,
                source_SHA256={str(file):sha(file) for file in [Path(__file__),p/'REGISTRATION_V1.json',
                    baseline/'REGISTRATION_V1.json',raw/'native_parameters.npz',old/'native_parameters.npz']})
    target=p/'INDEPENDENT_STABLE_MOTOR_COUNTERFACTUAL_V1.json'
    assert not target.exists()
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(identical),'unchanged arrays',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('version',type=int)
    main(parser.parse_args().version)
