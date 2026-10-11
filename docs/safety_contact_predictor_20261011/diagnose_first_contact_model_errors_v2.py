"""Locate first original model/force violations in a closed record, no retuning."""
from pathlib import Path
import argparse,datetime,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(version):
    p=H/f'fourhand_native_probe_v{version}';out=p/'FIRST_CONTACT_MODEL_ERROR_DIAGNOSIS_V2.json';assert not out.exists()
    actual=read(p/f'ACTUAL_COMPLETION_EXIT_V{version}.json');assert actual['actual_native_exit']==actual['actual_finite_exit']==actual['actual_launcher_subprocess_exit']==0
    assert read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json')['record_integrity_verified']
    reg=read(p/'REGISTRATION_V1.json');raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out']);assets=read(reg['coupled_assets'])['assets'];params=np.load(raw/'native_parameters.npz',allow_pickle=False)
    links={};first={};phase_peaks={};sources=[Path(__file__),p/'REGISTRATION_V1.json',raw/'native_parameters.npz']
    for arm,asset in assets.items():
        names=params[arm+'_joint_names'].tolist();relations=[]
        for joint in ET.parse(asset['original_URDF']).findall('joint'):
            mimic=joint.find('mimic')
            if mimic is not None and joint.get('name') in asset['hand_joint_names']:
                relations.append((names.index(joint.get('name')),names.index(mimic.get('joint')),float(mimic.get('multiplier','1')),float(mimic.get('offset','0'))))
        links[arm]=(names,relations);sources.append(Path(asset['original_URDF']))
    for chunk in read(raw/'physics_chunks.json'):
        f=raw/chunk['path'];assert sha(f)==chunk['sha256'];sources.append(f)
        with np.load(f,allow_pickle=False) as z:
            def describe(i,extra):
                return dict(event=int(z['physics_event'][i]),time_s=float(z['state_time_s'][i]),phase=str(z['phase'][i]),four_hand_normal_N=z['four_hand_normal_N'][i].tolist(),support_N=float(z['support_normal_N'][i]),minimum_geometry_m=float(z['all9021_raw_geometry'][i].min()),**extra)
            for j,arm in enumerate(['F_L','F_R','U_L','U_R']):
                normals=z['four_hand_normal_N'][:,j]
                for bound in [.1,6.,30.]:
                    key=f'{arm}/first_normal_gt_{bound:g}N';ii=np.flatnonzero(normals>bound)
                    if key not in first and len(ii):first[key]=describe(ii[0],dict(bound_N=bound))
                names,relations=links[arm];q=z[arm+'_native_q'][:,0];v=z[arm+'_native_qd'][:,0]
                for slave,master,scale,bias in relations:
                    qe=abs(q[:,slave]-scale*q[:,master]-bias);ve=abs(v[:,slave]-scale*v[:,master])
                    for field,values in [('q',qe),('qd',ve)]:
                        ii=np.flatnonzero(values>1e-4);key=f'{arm}/first_original_{field}_gt_1e-4'
                        if len(ii):
                            i=int(ii[0]);r=describe(i,dict(slave=names[slave],master=names[master],q_residual_rad=float(qe[i]),qd_residual_rad_s=float(ve[i])))
                            if key not in first or r['event']<first[key]['event']:first[key]=r
                    for phase in np.unique(z['phase']):
                        active=z['phase']==phase;key=f'{arm}/{phase}';r=phase_peaks.setdefault(key,dict(max_q_rad=0.,max_qd_rad_s=0.,max_normal_N=0.));r['max_q_rad']=max(r['max_q_rad'],float(qe[active].max()));r['max_qd_rad_s']=max(r['max_qd_rad_s'],float(ve[active].max()));r['max_normal_N']=max(r['max_normal_N'],float(normals[active].max()))
    out.write_text(json.dumps(dict(status='CLOSED_NATIVE_CONTACT_AND_ORIGINAL_MODEL_EVENT_LOCALIZATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),version=version,first_events=first,phase_peaks=phase_peaks,
        actual_outer_exit=actual['actual_outer_exit'],posthoc_transport_bounded_audit=True,no_new_native_samples=True,not_causal_certificate=True,original_gates_unchanged=True,fullSystem0_accepted=False,source_sha256={str(f):sha(f) for f in sources}),indent=2)+'\n');print('FIRST_CONTACT_MODEL_LOCALIZATION',first,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);main(parser.parse_args().version)
