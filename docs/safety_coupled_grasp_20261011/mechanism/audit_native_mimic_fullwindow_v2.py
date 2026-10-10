"""Independent original-URDF relation audit of the complete captured native development window."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(version):
    P=H/f'fourhand_native_probe_v{version}';audit=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    reg=json.loads((P/'REGISTRATION_V1.json').read_text());assert not reg['diagnostic_only'] and reg['maximum_controls']==3600
    raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out']);params=np.load(raw/'native_parameters.npz');meta=json.loads(Path(reg['coupled_assets']).read_text())
    pieces={a:[] for a in ARMS};sources={}
    for c in json.loads((raw/'physics_chunks.json').read_text()):
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f,allow_pickle=False) as z:
            for a in ARMS:pieces[a].append({k:z[a+k].copy() for k in ['_native_q','_native_qd','_mimic_position_residual_rad','_mimic_velocity_residual_rad_s']})
    rows=[]
    for a in ARMS:
        asset=meta['assets'][a];urdf=Path(asset['original_URDF']);tree=ET.parse(urdf);native=params[a+'_joint_names'].tolist();relations=[j for j in tree.findall('joint') if j.find('mimic') is not None and j.get('name') in asset['hand_joint_names']];assert len(relations)==6
        data={k:np.concatenate([p[k] for p in pieces[a]]) for k in pieces[a][0]};q=data['_native_q'][:,0];v=data['_native_qd'][:,0]
        positions=[];velocities=[]
        for j in relations:
            m=j.find('mimic');slave=j.get('name');master=m.get('joint');mult=float(m.get('multiplier','1'));offset=float(m.get('offset','0'));si=native.index(slave);mi=native.index(master)
            delta=q[:,si]-mult*q[:,mi]-offset;vd=v[:,si]-mult*v[:,mi]
            assert float(params[a+'_stiffness'][0,si])==float(params[a+'_damping'][0,si])==0.,'Follower must not have its own PD drive'
            rows.append(dict(arm=a,slave=slave,master=master,source_multiplier=mult,source_offset_rad=offset,max_native_q_residual_rad=float(abs(delta).max()),max_native_qd_residual_rad_s=float(abs(vd).max()),zero_slave_PD=True))
            positions.append(delta);velocities.append(vd)
        # Candidate telemetry order comes from the original diagnosed relation
        # list; the independent XML traversal may use a different ordering.
        by_slave={r['slave']:r for r in rows if r['arm']==a}
        for k,r in enumerate(asset['relations']):
            si=native.index(r['slave']);mi=native.index(r['master']);delta=q[:,si]-r['urdf_multiplier']*q[:,mi]-r['urdf_offset_rad'];vd=v[:,si]-r['urdf_multiplier']*v[:,mi]
            np.testing.assert_allclose(delta,data['_mimic_position_residual_rad'][:,k],atol=1e-7,rtol=0)
            np.testing.assert_allclose(vd,data['_mimic_velocity_residual_rad_s'][:,k],atol=1e-6,rtol=0)
        sources[str(urdf)]=sha(urdf)
    assert audit['physics_events']==2*audit['completed_controls']>0,'Incomplete capture cannot validate coupling'
    maximum=max(r['max_native_q_residual_rad'] for r in rows);max_v=max(r['max_native_qd_residual_rad_s'] for r in rows)
    model_pass=maximum<=1e-4 and max_v<=1e-4 and audit['physical_gates_pass']
    for f in [Path(__file__),P/'REGISTRATION_V1.json',P/'INDEPENDENT_CAPTURE_AUDIT_V1.json',Path(reg['coupled_assets']),raw/'native_parameters.npz',raw/'native_mimic_identity.json']:sources[str(f)]=sha(f)
    report=dict(status='CLOSED_INDEPENDENT_NATIVE_URDF_COUPLING_FULLWINDOW_AUDIT',version=version,controls=audit['completed_controls'],physics_events=audit['physics_events'],model_window_pass=model_pass,
        angular_URDF_offsets_reconstructed_from_native_positions=True,maximum_q_residual_rad=maximum,maximum_qd_residual_rad_s=max_v,predeclared_relation_tolerance=1e-4,
        original_physical_gates_pass=audit['physical_gates_pass'],relations=rows,source_SHA256=sources,
        loaded_friction_task_validated=model_pass and audit['four_hand_task_development_pass'],full_task_pass=audit['four_hand_task_development_pass'],fullSystem0_accepted=False,formal_holdout_count=0)
    target=P/'INDEPENDENT_NATIVE_MIMIC_FULLWINDOW_V2.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],'window_pass',model_pass,'qerr',maximum,'verr',max_v,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',type=int);main(p.parse_args().version)
