"""Synthetic raw-camera integration specimen. NEVER native evidence."""
import hashlib
import itertools
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from evidence_io import ARMS,H,SCREEN,Evidence,write_new,sha
from camera_adapter import hand_path,micro_native,CameraAudit
from camera_oracle import token,FINGERS,family
from failed_camera import SOURCES
from test_science_baseline import layout_fixture,row_fixture


def fixture(root,weak_palm=False):
    e=Evidence();binding=e.js(SCREEN/'INPUT_BINDING_V1.json')
    body=e.npz(Path(binding['parameter_path']).parent/'body_identity.npz')
    contacts=e.js(binding['contact_identity_path'])
    native_paths={v['arm']:{(int(p.split('/env_')[1].split('/')[0]),p.rsplit('/',1)[-1]):p for p in v['sensors']} for v in contacts['views']}
    asset=e.js(H/'astra/next_mechanism/CAMERA_PALM_USD_IDENTITIES_V1.json')
    inventory={};context={};palms={};handtokens={}
    for slot in range(32):
        for arm in ARMS:
            for i,name in enumerate(body[arm]):
                path=native_paths[arm][slot,str(name)]
                inventory[path]=dict(env_id=slot,arm=arm,body_name=str(name),body_index=i,hand=hand_path(path))
                context[token('qv6_context_',path)]=path
                if hand_path(path):handtokens[token('qv4_hand_',path)]=path
                for part in asset['arms'][arm]['parts']:
                    if part['native_body_name']!=name:continue
                    visual=path+'/'+part['relative_visual_path'];key=token('qv6_palm_',visual)
                    palms[key]=dict(rigid_path=path,visual_path=visual,actual_mesh_paths=[path+'/'+s for s in part['relative_mesh_paths']])
        for table in ('TableF','TableU'):
            path=f'/World/envs/env_{slot}/{table}';context[token('qv6_context_',path)]=path
    setup=dict(environment_count=32,context_registry=context,semantic_token_to_native_path=handtokens,palm_visual_registry=palms)
    layout=layout_fixture();row=row_fixture()
    for arm in ARMS:
        row[arm+'_root_xyzw'][:,6]=1;row[arm+'_link_xyzw'][:,:,6]=1
    fields={k:v[None] for k,v in row.items()};fields.update(simulation_time_s=np.array([0.],np.float64),simulation_step=np.array([0],np.int64))
    native=micro_native(fields,0,layout,np.zeros((32,3),np.float32))
    def js(name,value):
        p=root/name;write_new(p,value);return dict(path=name,sha256=sha(p))
    def nz(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,**value)
        return dict(path=name,sha256=sha(p))
    native_ref=nz('native.npz',native);invref=js('inventory.json',inventory);setupref=js('setup.json',setup)
    paths=[p for p,m in inventory.items() if m['env_id']==0]+['/World/envs/env_0/TableF','/World/envs/env_0/TableU']
    ids=np.zeros((720,1280,1),np.uint32);labels_i={'0':'BACKGROUND'};labels_s={'0':{'class':'BACKGROUND'}};palm_pixels={a:0 for a in ARMS}
    for i,path in enumerate(paths,1):
        ids.flat[(i-1)*4096:i*4096]=i;labels_i[str(i)]=path+'/mesh'
        tokens=[token('qv6_context_',path)]
        if path in inventory and inventory[path]['hand']:tokens.append(token('qv4_hand_',path))
        parts=[k for k,v in palms.items() if v['rigid_path']==path]
        if parts and not (weak_palm and inventory[path]['arm']=='F_L'):
            tokens.append(parts[0]);palm_pixels[inventory[path]['arm']]+=4096
        labels_s[str(i)]={'class':','.join(tokens)}
    rgb=np.zeros((720,1280,3),np.uint8);rgb[...,0]=(ids[...,0]%251).astype(np.uint8)
    output=nz('outputs.npz',dict(rgb=rgb,instance_segmentation_fast=ids,semantic_segmentation=ids.copy()))
    info=js('info.json',{'instance_segmentation_fast':{'idToLabels':labels_i},'semantic_segmentation':{'idToLabels':labels_s}})
    tables=[dict(lower_world_m=[-.01]*3,upper_world_m=[.01]*3) for _ in range(2)]
    bounds={p:dict(actual_native_pose_xyzw=native[m['arm']+'_native_link_transforms_xyzw'][0,m['body_index']].astype(float).tolist(),local_lower=[-.01]*3,local_upper=[.01]*3) for p,m in inventory.items() if m['env_id']==0}
    corners=list(itertools.product(*zip([-.01]*3,[.01]*3)));wide_points=corners*(len(bounds)+2)
    boundref=js('bounds.json',dict(bounds=bounds,frustum_points_world_m=wide_points))
    directory='env_000';(root/directory).mkdir()
    def attempt(kind,arm,number):
        angle=(number-1)*60;rad=math.radians(angle);z=np.array([math.cos(rad),math.sin(rad),0.]);x=np.cross([0.,0.,1.],z);y=np.cross(z,x)
        world=np.eye(4);world[:3,:3]=np.array([x,y,z]);world[3,:3]=3*z;k=[[640.,0,640.],[0,640.,360.],[0,0,1.]]
        camera=dict(actual_camera_to_world_row_matrix=world.tolist(),actual_intrinsic_matrix=k,sdk_intrinsic_matrix=k,
            actual_clipping_range_m=[.01,10.],image_size=[1280,720],actual_usd_optics=dict(focal_length=10.,horizontal_aperture=20.,vertical_aperture=11.25,horizontal_aperture_offset=0,vertical_aperture_offset=0))
        name=f'{directory}/{kind}_{arm}_{number}';Image.fromarray(rgb).save(root/(name+'.png'))
        rec=dict(attempt=number,status='qualified',rendered=True,reasons=[],camera=camera,azimuth_deg=angle,
            native_before_all64=native_ref,native_after_all64=native_ref,
            held_renders=[dict(index=i,native_before_all64=native_ref,native_after_all64=native_ref) for i in range(3)],
            render_calls=3,sensor_frame_before=[0],sensor_frame_after=[1],sensor_update_dt_s=0.,original_outputs=output,original_info=info,
            rgb=dict(path=name+'.png',sha256=sha(root/(name+'.png'))),raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        if kind=='hand':
            rec['arm']=arm;rec['hand_link_positions_world_m']=[[0.,0.,0.] for p,m in inventory.items() if m['env_id']==0 and m['arm']==arm and m['hand']]
        return dict(rec,attempt_record=js(name+'.attempt.json',rec))
    attempts=[];arms={}
    for arm in ARMS:
        rows=[attempt('hand',arm,i) for i in range(1,4)];attempts.extend(rows)
        handpaths=[p for p,m in inventory.items() if m['env_id']==0 and m['arm']==arm and m['hand']]
        fingers={f:sum(4096 for p in handpaths if family(inventory[p]['body_name'])==f) for f in FINGERS}
        good=palm_pixels[arm]>=2048 and all(v>=512 for v in fingers.values());refs=[r['rgb'] for r in rows]
        coverage=dict(status='qualified' if good else 'failed',palm_visual_peak_pixels=palm_pixels[arm],finger_family_peak_pixels=fingers,
            per_native_body_peak_pixels={p:4096 for p in handpaths},native_body_visibility_gaps=[],all_native_hand_bodies_observed=True,full_mesh_coverage_certified=False)
        arms[arm]=dict(status='qualified' if good else 'failed',selected_group_images=refs,qualified_images=refs if good else [],qualified_count=3 if good else 0,
            attempts=[r['attempt_record'] for r in rows],representative_image=refs[0],group_coverage=coverage)
    wide=[attempt('wide','all',i) for i in range(1,4)];refs=[r['rgb'] for r in wide]
    wc=dict(status='qualified',per_native_object_peak_pixels={p:4096 for p in paths},
        arm_peak_pixels={a:sum(4096 for p,m in inventory.items() if m['env_id']==0 and m['arm']==a) for a in ARMS},
        table_peak_pixels={t:4096 for t in ('TableF','TableU')},
        critical_body_peak_pixels={p:4096 for p,m in inventory.items() if m['env_id']==0 and m['body_name'] in ('forearm_link','wrist_2_link')},critical_body_visibility_gaps=[])
    state=dict(schema='safeduo.qualified_views.v7',env_id=0,capture_kind='initial',step=-1,substep=None,status='failed' if weak_palm else 'qualified',reasons=['insufficient_palm'] if weak_palm else [],
        environment_count=32,all_registered_lanes=True,all64=False,physics_advanced_by_collector=False,
        simulation_time_s=0.,simulation_time_step_index=0,native_before_all64=native_ref,native_final_all64=native_ref,
        native_path_mapping=invref,semantic_setup=setupref,actual_table_aabbs=tables,arms=arms,attempts=attempts,
        wide_context=dict(status='qualified',group_coverage=wc,attempts=wide,selected_images=refs,qualified_images=refs,authored_bounds=boundref),
        image_count=12,qualified_image_count=sum(x['qualified_count'] for x in arms.values()))
    state.update({key:sha(path) for key,path in SOURCES.items()});ref=js(directory+'/state.json',state)
    receipt={k:state[k] for k in ('env_id','status','reasons','arms','wide_context','image_count','qualified_image_count')};receipt.update(state=ref['path'],sha256=ref['sha256'])
    from camera_index import metadata_digest
    receipt['_metadata_sha256']=metadata_digest(receipt)
    # Newly written CIFS files can expose cached sub-100ns timestamps until
    # the attribute cache refreshes. Settle the CPU specimen before its first
    # evidence ledger read; production evidence rules remain unchanged.
    time.sleep(2.2)
    identity={'views':[dict(sensors=list(inventory))]};audit=CameraAudit(root,Evidence(),body,identity,layout)
    return audit,receipt,fields,asset
