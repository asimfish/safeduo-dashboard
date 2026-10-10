"""Source/native inertia estimate for controller conditioning, not a load certificate.

Native COM translations were not recorded in prior probes. This diagnostic uses
authored COM after checking authored/native masses and inertia tensors agree.
The hand mass block assumes the arm fixed; it is not an observed full generalized
mass matrix or identification of hardware transmission compliance.
"""
from pathlib import Path
import hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/liyufeng/b_usdtools/lib/python3.11/site-packages')
from pxr import Usd,UsdPhysics
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quaternion(q):return np.array([*q.GetImaginary(),q.GetReal()],float)
def main():
    dest=H/'coupled_motor_damping_diagnosis_v1';assert not dest.exists();dest.mkdir()
    p=H/'fourhand_native_probe_v26';reg=json.loads((p/'REGISTRATION_V1.json').read_text())
    raw=Path(json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
    audit=json.loads((p/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    assets=json.loads(Path(reg['coupled_assets']).read_text());params=np.load(raw/'native_parameters.npz')
    pieces=[];sources={}
    for c in json.loads((raw/'physics_chunks.json').read_text()):
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:pieces.append({k:z[k].copy() for k in z.files if k in ['physics_event','state_time_s','phase'] or '_body_position_local' in k or '_body_quaternion_wxyz' in k})
    data={k:np.concatenate([z[k] for z in pieces]) for k in pieces[0]}
    np.testing.assert_array_equal(data['physics_event'],np.arange(audit['physics_events']))
    rows=[];matrices={k:data[k] for k in ['physics_event','state_time_s','phase']}
    for a,asset in assets['assets'].items():
        names=params[a+'_body_names'].tolist();joint_names=params[a+'_joint_names'].tolist()
        hn=asset['hand_joint_names'];motors=asset['independent_motor_joint_names']
        xml={j.get('name'):j for j in ET.parse(asset['original_URDF']).findall('joint')}
        def derivative(name,seen=()):
            assert name not in seen
            m=xml[name].find('mimic')
            if m is None:
                x=np.zeros(len(motors));x[motors.index(name)]=1.;return x
            return float(m.get('multiplier','1'))*derivative(m.get('joint'),(*seen,name))
        A=np.stack([derivative(n) for n in hn]);assert A.shape==(12,6) and np.linalg.matrix_rank(A)==6
        stage=Usd.Stage.Open(asset['layer']);assert stage
        prims={}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in hn:
                assert prim.GetName() not in prims;prims[prim.GetName()]=prim
        assert set(prims)==set(hn)
        positions=data[a+'_body_position_local'][:,0];R=Rotation.from_quat(data[a+'_body_quaternion_wxyz'][:,0][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(positions),len(names),3,3)
        joints=[];children={}
        for n in hn:
            j=UsdPhysics.RevoluteJoint(prims[n]);body0=j.GetBody0Rel().GetTargets();body1=j.GetBody1Rel().GetTargets();assert len(body0)==len(body1)==1
            parent=body0[0].name;child=body1[0].name;assert parent in names and child in names
            children.setdefault(parent,[]).append(child)
            pi=names.index(parent);axis=j.GetAxisAttr().Get();unit=np.eye(3)['XYZ'.index(axis)]
            localR=Rotation.from_quat(quaternion(j.GetLocalRot0Attr().Get())).as_matrix()
            world_axis=np.einsum('nij,j->ni',R[:,pi],localR@unit)
            pivot=positions[:,pi]+np.einsum('nij,j->ni',R[:,pi],np.asarray(j.GetLocalPos0Attr().Get(),float))
            joints.append(dict(name=n,parent=parent,child=child,axis=world_axis,pivot=pivot,body_path=str(body1[0])))
        def descend(name,seen=()):
            assert name not in seen
            return {name}|{x for c in children.get(name,[]) for x in descend(c,(*seen,name))}
        descendants=[descend(j['child']) for j in joints]
        handbodies=sorted(set.union(*descendants));mass=np.asarray(params[a+'_link_masses'][0],float)
        inertias=np.asarray(params[a+'_link_inertias'][0],float).reshape(len(names),3,3).transpose(0,2,1)
        block=np.zeros((len(positions),12,12));body_rows=[]
        for name in handbodies:
            bi=names.index(name);source_path=next(j['body_path'] for j in joints if j['child']==name)
            body=stage.GetPrimAtPath(source_path);api=UsdPhysics.MassAPI(body);assert api
            source_mass=float(api.GetMassAttr().Get());com=np.array(api.GetCenterOfMassAttr().Get(),float)
            principal=Rotation.from_quat(quaternion(api.GetPrincipalAxesAttr().Get())).as_matrix()
            source_I=principal@np.diag(np.asarray(api.GetDiagonalInertiaAttr().Get(),float))@principal.T
            np.testing.assert_allclose(source_mass,mass[bi],atol=1e-8,rtol=1e-6)
            np.testing.assert_allclose(source_I,inertias[bi],atol=2e-10,rtol=1e-5)
            assert np.isfinite(com).all() and np.linalg.eigvalsh(inertias[bi]).min()>0
            C=positions[:,bi]+np.einsum('nij,j->ni',R[:,bi],com)
            Jw=np.zeros((len(positions),12,3));Jv=np.zeros_like(Jw)
            for ji,j in enumerate(joints):
                if name in descendants[ji]:Jw[:,ji]=j['axis'];Jv[:,ji]=np.cross(j['axis'],C-j['pivot'])
            Iworld=np.einsum('nij,jk,nlk->nil',R[:,bi],inertias[bi],R[:,bi])
            block+=mass[bi]*np.einsum('nik,njk->nij',Jv,Jv)+np.einsum('nik,nkl,njl->nij',Jw,Iworld,Jw)
            body_rows.append(dict(name=name,mass_kg=float(mass[bi]),native_mass_and_inertia_match_authored_USD=True,COM_translation_source='authored_USD_not_independently_native_recorded'))
        reduced=np.einsum('ia,nij,jb->nab',A,block,A)
        assert np.max(abs(reduced-reduced.transpose(0,2,1)))<1e-12
        eigen=np.linalg.eigvalsh(reduced);assert eigen.min()>0
        mi=[joint_names.index(n) for n in motors];K=params[a+'_stiffness'][0,mi];D=params[a+'_damping'][0,mi]
        critical=2*np.sqrt(np.diagonal(reduced,axis1=1,axis2=2)*K)
        ratio=D[None]/critical;assert np.isfinite(critical).all() and (critical>0).all()
        # Independently compare reduced kinetic energy with full-coordinate
        # directional velocity; this validates algebra, not hardware physics.
        velocity=np.random.default_rng(2630).normal(size=(len(positions),6));full=velocity@A.T
        np.testing.assert_allclose(np.einsum('ni,nij,nj->n',velocity,reduced,velocity),np.einsum('ni,nij,nj->n',full,block,full),atol=1e-12,rtol=1e-12)
        matrices[a+'_full_hand_inertia']=block;matrices[a+'_motor_inertia']=reduced;matrices[a+'_critical_diagonal_damping_estimate']=critical
        rows.append(dict(arm=a,motors=motors,legacy_K=K.tolist(),legacy_D=D.tolist(),
            estimated_critical_D_min=critical.min(0).tolist(),estimated_critical_D_max=critical.max(0).tolist(),
            legacy_D_over_estimated_critical_min=ratio.min(0).tolist(),legacy_D_over_estimated_critical_max=ratio.max(0).tolist(),
            reduced_inertia_min_eigenvalue=float(eigen.min()),reduced_inertia_max_eigenvalue=float(eigen.max()),
            body_rows=body_rows))
        for f in [Path(asset['layer']),Path(asset['source_USD']),Path(asset['original_URDF'])]:sources[str(f)]=sha(f)
    f=dest/'SOURCE_NATIVE_HAND_INERTIA_DIAGNOSTIC_V1.npz';np.savez_compressed(f,**matrices)
    for f0 in [Path(__file__),p/'REGISTRATION_V1.json',p/'INDEPENDENT_CAPTURE_AUDIT_V1.json',Path(reg['coupled_assets']),raw/'native_parameters.npz',f]:sources[str(f0)]=sha(f0)
    report=dict(status='CLOSED_SOURCE_NATIVE_DRIVE_CONDITIONING_DIAGNOSTIC_NOT_CONTROLLER_ACCEPTANCE',
        native_events=audit['physics_events'],rows=rows,source_SHA256=sources,
        explicit_limitations=['COM is authored USD; prior native COM was not recorded','Arm held fixed in estimated hand mass block','Diagonal critical-damping estimate ignores full matrix modal coupling','No measured hardware transmission compliance','Correlation or estimated overdamping is not causal proof of current mimic failures'],
        no_physics_or_USD_mutations=True,new_controller_adopted=False,fullSystem0_accepted=False)
    (dest/'DRIVE_CONDITIONING_DIAGNOSIS_V1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],flush=True)
    for r in rows:print(r['arm'],'legacy/critical estimate range',min(r['legacy_D_over_estimated_critical_min']),max(r['legacy_D_over_estimated_critical_max']),flush=True)
if __name__=='__main__':main()
