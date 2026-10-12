"""Original URDF finger FK from measured coordinates, anchored at the actual wrist.

This checks hand kinematics only. A native wrist anchor is not an arm-motion
predictor, a tracking bound, a clearance certificate or a new safety experiment.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
D=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpy_matrix(rpy):
    r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def quaternion_matrix(xyzw):
    q=np.asarray(xyzw,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)


class HandFK:
    def __init__(self,urdf,anchor):
        self.anchor=anchor;children={}
        for j in ET.parse(urdf).findall('joint'):
            origin=j.find('origin');axis=j.find('axis')
            xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            direction=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
            kind=j.get('type');assert kind in ('fixed','revolute','continuous','prismatic'),kind
            if kind!='fixed':assert np.linalg.norm(direction)>0
            children.setdefault(j.find('parent').get('link'),[]).append((j.find('child').get('link'),j.get('name'),kind,xyz,rpy_matrix(rpy),direction))
        self.joints=[];queue=[anchor];seen={anchor}
        for parent in queue:
            for child,name,kind,xyz,rotation,axis in children.get(parent,[]):
                assert child not in seen;seen.add(child);queue.append(child)
                self.joints.append((parent,child,name,kind,xyz,rotation,axis))
        self.links=seen;assert len(self.joints)>0

    def transforms(self,q,anchor_transform):
        root=np.asarray(anchor_transform,float);assert root.ndim==3 and root.shape[1:]==(4,4)
        n=len(root);out={self.anchor:root.copy()}
        for parent,child,name,kind,xyz,rotation,axis in self.joints:
            origin=np.broadcast_to(np.eye(4),(n,4,4)).copy();origin[:,:3,:3]=rotation;origin[:,:3,3]=xyz
            move=np.broadcast_to(np.eye(4),(n,4,4)).copy()
            if kind!='fixed':
                angle=np.asarray(q[name],float);assert angle.shape==(n,) and np.isfinite(angle).all()
                axis=axis/np.linalg.norm(axis)
                if kind=='prismatic':move[:,:3,3]=angle[:,None]*axis
                else:
                    x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    move[:,:3,:3]=np.eye(3)+np.sin(angle)[:,None,None]*skew+(1-np.cos(angle))[:,None,None]*(skew@skew)
            out[child]=out[parent]@origin@move
        return out


def main():
    planfile=D/'HAND_FK_NATIVE_REPLAY_PROSPECTIVE_V1.json';assert not planfile.exists()
    plan={'scope':'CPU replay of all actual hand coordinates using original source URDF and an actual native wrist/hand-base anchor; not future arm trajectory prediction',
          'versions':[109,110,112,113],'translation_tolerance_m':1e-4,'rotation_tolerance_rad':1e-3,
          'state_projection_or_passive_follower_reconstruction_used':False,'new_native_launches':0,
          'native_body_or_joint_state_changes':0,'formal_holdouts':0,'full50_predictor_qualified':False,
          'source_sha256':sha(__file__)}
    planfile.write_text(json.dumps(plan,indent=2)+'\n')
    rows=[];sources={str(Path(__file__)):sha(__file__)}
    for v in plan['versions']:
        p=H/f'fourhand_native_probe_v{v}';reg=read(p/'REGISTRATION_V1.json');meta=read(reg['coupled_assets'])
        raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out']);params=np.load(raw/'native_parameters.npz')
        for a in ARMS:
            anchor=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
            urdf=Path(meta['assets'][a]['original_URDF']);fk=HandFK(urdf,anchor);sources[str(urdf)]=sha(urdf)
            names=params[a+'_joint_names'].tolist();bodies=params[a+'_body_names'].tolist();ai=bodies.index(anchor)
            matched=[(i,b) for i,b in enumerate(bodies) if b in fk.links and b!=anchor]
            assert len(matched)>=12
            position_error=0.;angle_error=0.;worst=None;events=0
            for c in read(raw/'physics_chunks.json'):
                f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
                with np.load(f) as z:
                    q=z[a+'_native_q'][:,0];native=z[a+'_native_body_transforms_after_xyzw'][:,0]
                    n=len(q);root=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                    root[:,:3,:3]=quaternion_matrix(native[:,ai,3:]);root[:,:3,3]=native[:,ai,:3]
                    predicted=fk.transforms({name:q[:,i] for i,name in enumerate(names)},root)
                    for i,b in matched:
                        pe=np.linalg.norm(predicted[b][:,:3,3]-native[:,i,:3],axis=1)
                        relative=predicted[b][:,:3,:3].transpose(0,2,1)@quaternion_matrix(native[:,i,3:])
                        ae=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
                        if pe.max()>position_error:position_error=float(pe.max());worst=b
                        angle_error=max(angle_error,float(ae.max()))
                    events+=n
            rows.append({'version':v,'arm':a,'native_events':events,'matched_hand_bodies':len(matched),
                'maximum_native_hand_position_error_m':position_error,'maximum_native_hand_rotation_error_rad':angle_error,
                'worst_position_body':worst,'replay_parity_pass':position_error<=plan['translation_tolerance_m'] and angle_error<=plan['rotation_tolerance_rad']})
        params.close()
    result={'status':'CLOSED_CPU_NATIVE_HAND_KINEMATIC_REPLAY','plan':plan,'rows':rows,
            'all4_hands_each_case_replay_parity_pass':all(r['replay_parity_pass'] for r in rows),
            'current_native_wrist_anchors_used':True,'predicted_actual_arm_motion_or_tracking_bound':False,
            'full50_guard_native_executed':0,'new_native_jobs':0,'new_training_jobs':0,'fullSystem0_accepted':False,
            'source_sha256':sources}
    (D/'HAND_FK_NATIVE_REPLAY_RESULT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HAND_FK_REPLAY',result['all4_hands_each_case_replay_parity_pass'],json.dumps(rows),flush=True)


if __name__=='__main__':main()
