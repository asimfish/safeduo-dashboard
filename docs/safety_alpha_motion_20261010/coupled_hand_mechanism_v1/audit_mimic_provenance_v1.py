"""Compare original URDF hand coupling with delivered USD and native14 records.

Read-only diagnosis. Missing asset coupling does not by itself establish the
cause of contact instability or prove a replacement physical model.
"""
from pathlib import Path
import csv,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,'/home/liyufeng/b_usdtools/lib/python3.11/site-packages')
from pxr import Usd,UsdPhysics
H=Path(__file__).resolve().parent
P=H/'mimic_provenance_v1'
A=Path('/home/liyufeng/safeduo/assets_real')
ARMS=('F_L','F_R','U_L','U_R')
URDFS={
 'F_L':A/'inspire_RH56F2/inspire_RH56F2/urdf/RH56F2_ws/RH56F2_L/urdf/RH56F2_L.urdf',
 'F_R':A/'inspire_RH56F2/inspire_RH56F2/urdf/RH56F2_ws/RH56F2_R/urdf/RH56F2_R.urdf',
 'U_L':A/'ur5_dfx/urdf/ur5_dfx_left_v7.urdf',
 'U_R':A/'ur5_dfx/urdf/ur5_dfx_right_v7.urdf'}
USDS={'F_L':A/'usd/combined/fr3_f2_left_v5.usd','F_R':H/'fourhand_native_probe_v14/fr3_f2_right_explicit_axes_v1.usda',
      'U_L':A/'usd/ur5_dfx_left_v7.usd','U_R':A/'usd/ur5_dfx_right_v7.usd'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not P.exists();P.mkdir()
    capture=H/'fourhand_native_probe_v14'
    audit=json.loads((capture/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    raw=Path(json.loads((capture/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
    params=np.load(raw/'native_parameters.npz',allow_pickle=False)
    records=json.loads((raw/'physics_chunks.json').read_text())
    fields=['physics_event','state_time_s','phase']+[a+s for a in ARMS for s in ['_native_q','_native_qd','_actual_full_position_target']]
    pieces={k:[] for k in fields};sources={}
    for r in records:
        f=raw/r['path'];actual=sha(f);assert actual==r['sha256'];sources[str(f)]=actual
        with np.load(f,allow_pickle=False) as z:
            for k in fields:pieces[k].append(z[k].copy())
    data={k:np.concatenate(v) for k,v in pieces.items()}
    assert np.array_equal(data['physics_event'],np.arange(audit['physics_events']))
    coupling=[];arrays={'physics_event':data['physics_event'],'time_s':data['state_time_s'],'phase':data['phase']}
    asset_records={}
    for a in ARMS:
        tree=ET.parse(URDFS[a]);joints=tree.findall('joint');native_names=params[a+'_joint_names'].tolist()
        stage=Usd.Stage.Open(str(USDS[a]));native_joints={}
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() in native_names:
                assert prim.GetName() not in native_joints
                native_joints[prim.GetName()]=prim
        hand_names=[n for n in native_names if any(t in n for t in ['thumb','index','middle','ring','pinky','little'])]
        assert len(hand_names)==12 and set(hand_names)<=set(native_joints)
        mimics=[j for j in joints if j.find('mimic') is not None and j.get('name') in hand_names]
        assert len(mimics)==6,(a,len(mimics))
        rows=[]
        for j in mimics:
            slave=j.get('name');m=j.find('mimic');master=m.get('joint');mult=float(m.get('multiplier','1'));offset=float(m.get('offset','0'))
            assert master in hand_names;si=native_names.index(slave);mi=native_names.index(master)
            actual_q=data[a+'_native_q'][:,0,si]-mult*data[a+'_native_q'][:,0,mi]-offset
            actual_v=data[a+'_native_qd'][:,0,si]-mult*data[a+'_native_qd'][:,0,mi]
            target=data[a+'_actual_full_position_target'][:,0,si]-mult*data[a+'_actual_full_position_target'][:,0,mi]-offset
            prim=native_joints[slave];schemas=list(prim.GetAppliedSchemas());apis=[s for s in schemas if s.startswith('PhysxMimicJointAPI:')]
            row=dict(arm=a,slave=slave,master=master,urdf_multiplier=mult,urdf_offset_rad=offset,
              source_USD_joint_path=str(prim.GetPath()),source_USD_mimic_schemas=apis,
              native_slave_stiffness=float(params[a+'_stiffness'][0,si]),native_slave_damping=float(params[a+'_damping'][0,si]),
              initial_actual_position_residual_rad=float(actual_q[0]),max_actual_position_residual_rad=float(abs(actual_q).max()),
              max_actual_velocity_residual_rad_s=float(abs(actual_v).max()),max_target_residual_rad=float(abs(target).max()),
              max_contact_phase_actual_residual_rad=float(abs(actual_q[np.isin(data['phase'],['CLOSE','LIFT'])]).max()))
            for key,value in [('q',actual_q),('v',actual_v),('target',target)]:arrays[a+'__'+slave+'__'+key+'_residual']=value
            rows.append(row);coupling.append(row)
        asset_records[a]=dict(hand_joint_coordinates=len(hand_names),original_URDF_mimic_relations=6,source_USD_mimic_relations=sum(bool(r['source_USD_mimic_schemas']) for r in rows),
            independently_driven_native_slave_joints=sum(r['native_slave_stiffness']>0 for r in rows),
            native_joint_coordinates=len(native_names),original_URDF_independent_hand_coordinates=6)
    paths=[Path(__file__),capture/'INDEPENDENT_CAPTURE_AUDIT_V1.json',raw/'native_parameters.npz',raw/'physics_chunks.json',*URDFS.values(),*USDS.values()]
    for f in paths:sources[str(f)]=sha(f)
    report=dict(status='CLOSED_SOURCE_COUPLING_VS_NATIVE_COORDINATE_DIAGNOSIS',physics_events=audit['physics_events'],all24_original_relations_missing_from_source_USD=all(not r['source_USD_mimic_schemas'] for r in coupling),
      assets=asset_records,relations=coupling,native_observed_state_coordinates=74,original_URDF_independent_hand_coordinates=24,original_URDF_independent_arm_coordinates=26,
      independent_URDF_motor_space_coordinates=50,source_sha256=sources,fullSystem0_accepted=False,
      official_schema_source='https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_mimic_joint_a_p_i.html',
      PhysX_relation='q_slave + gearing*q_reference + offset_degrees =0; original URDF linear rule uses gearing=-multiplier and offset_degrees=-rad2deg(URDF offset)',
      nonclaims=['Missing source mimic is an asset-fidelity discrepancy, not a proven sole cause of grip failure','Original URDF relations require independent hardware/manufacturer validation before hardware acceptance','No physical properties or native candidate changed by this audit','All74 joint positions/velocities remain mandatory observations;50 source-independent motor coordinates are not74 independently controllable motors','Existing unrelated geometry and physical criteria are not relaxed'])
    (P/'MIMIC_PROVENANCE_V1.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(P/'NATIVE_RELATION_RESIDUALS_V1.npz',**arrays)
    with (P/'relations.csv').open('w',newline='') as f:
        fields=[k for k in coupling[0] if k!='source_USD_mimic_schemas'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(coupling)
    print(report['status'],'missing',sum(not r['source_USD_mimic_schemas'] for r in coupling),'of',len(coupling),'maxQresid',max(r['max_actual_position_residual_rad'] for r in coupling),flush=True)
if __name__=='__main__':main()
