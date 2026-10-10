"""Independent original-XML effort reduction and native user-effort binding."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from audit_coupled_hand_targets_native_v1 import matrix
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(version):
    P=H/f'fourhand_native_probe_v{version}';reg=json.loads((P/'REGISTRATION_V1.json').read_text());audit=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out']);params=np.load(raw/'native_parameters.npz');meta=json.loads(Path(reg['coupled_assets']).read_text());records=[];sources={}
    files=json.loads((raw/'physics_chunks.json').read_text())
    for arm,a in meta['assets'].items():
        names=params[arm+'_joint_names'].tolist();hn=a['hand_joint_names'];ids=[names.index(n) for n in hn];roots,A,b=matrix(hn,a['original_URDF']);rootids=[names.index(n) for n in roots];slaveids=[names.index(r['slave']) for r in a['relations']]
        M=np.eye(len(names));M[ids]=0
        M[np.ix_(rootids,ids)]=A.T;np.testing.assert_allclose(M,params[arm+'_motor_effort_projection'],atol=1e-12,rtol=0)
        max_binding=0.;max_passive=0.;events=0
        for chunk in files:
            path=raw/chunk['path'];assert sha(path)==chunk['sha256'];sources[str(path)]=sha(path)
            with np.load(path) as z:
                g=z[arm+'_gravity_compensation_raw'];submitted=z[arm+'_extra_effort'];native=z[arm+'_native_external_actuation_effort']
                expected=np.clip(g@M.T,-params[arm+'_max_force'],params[arm+'_max_force'])
                if not bool(params[arm+'_gravity_compensation_enabled']):expected=np.zeros_like(g)
                np.testing.assert_allclose(expected,submitted,atol=1e-6,rtol=0);np.testing.assert_allclose(submitted,native,atol=1e-6,rtol=0)
                passive=float(abs(native[...,slaveids]).max());assert passive==0.
                max_passive=max(max_passive,passive);max_binding=max(max_binding,float(abs(native-expected).max()));events+=len(g)
        assert events==audit['physics_events'];records.append(dict(arm=arm,physics_events=events,independent_motors=len(roots),passive_followers=len(slaveids),maximum_passive_user_effort_Nm=max_passive,maximum_expected_native_binding_error_Nm=max_binding))
        sources[a['original_URDF']]=sha(a['original_URDF'])
    for path in [Path(__file__),H/'audit_coupled_hand_targets_native_v1.py',P/'REGISTRATION_V1.json',raw/'native_parameters.npz']:sources[str(path)]=sha(path)
    report=dict(status='PASS_INDEPENDENT_COUPLED_MOTOR_VIRTUAL_WORK_AND_NATIVE_EFFORT_BINDING',records=records,source_sha256=sources,physics_events=audit['physics_events'],implicit_drive_or_total_measured_torque_claimed=False,physical_gates_pass=audit['physical_gates_pass'],fullSystem0_accepted=False)
    path=P/'INDEPENDENT_COUPLED_MOTOR_EFFORT_V1.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['physics_events'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',type=int);main(p.parse_args().version)
