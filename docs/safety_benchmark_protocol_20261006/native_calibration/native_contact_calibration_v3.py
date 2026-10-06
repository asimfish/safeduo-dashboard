"""Registered primitive contact controls; independent of SafeDuo controller code.

Three known-mass cubes fall freely onto their own supports. Every cube is
filtered against all three supports plus a distant wrong partner. This is a
small native measurement calibration, not four-arm/grasp/safety validation.
"""
import argparse,json,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--registration',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
registration=json.loads(args.registration.read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==registration['source_sha256']
args.out.mkdir(parents=True,exist_ok=False)
launcher=AppLauncher(args);app=launcher.app
import torch,numpy as np
import os,importlib.metadata
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
import omni.physics.tensors as tensors
def main():
    dt=registration['dt_s'];steps=registration['steps'];masses=registration['masses_kg']
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=args.device,gravity=(0,0,-9.81)))
    cubes=[];supports=[]
    for i,mass in enumerate(masses):
        support=f'/World/Support{i}';supports.append(support)
        cfg=sim_utils.CuboidCfg(size=(.8,.8,.1),rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),activate_contact_sensors=True)
        cfg.func(support,cfg,translation=(2.*i,0,0))
        body=RigidObject(RigidObjectCfg(prim_path=f'/World/Cube{i}',
            spawn=sim_utils.CuboidCfg(size=(.1,.1,.1),rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.*i,0,.5))))
        cubes.append(body)
    wrong='/World/WrongPartner';cfg=sim_utils.CuboidCfg(size=(.8,.8,.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(mass=1),activate_contact_sensors=True)
    cfg.func(wrong,cfg,translation=(20,0,0));filters=supports+[wrong]
    sim.reset()
    view=tensors.create_simulation_view('torch');view.set_subspace_roots('/')
    contacts=[];identities=[];force_devices=set()
    for i in range(3):
        cv=view.create_rigid_contact_view(f'/World/Cube{i}',filter_patterns=filters,max_contact_data_count=128)
        assert cv.check() and cv.sensor_count==1 and cv.filter_count==4
        sensor=list(cv.sensor_paths);paths=list(cv.filter_paths)
        # Some versions expose each sensor's filter paths as one nested list.
        if len(paths)==1 and isinstance(paths[0],(list,tuple)):paths=list(paths[0])
        assert sensor==[f'/World/Cube{i}'] and paths==filters,(sensor,paths)
        contacts.append(cv);identities.append(dict(sensor_paths=sensor,filter_paths=paths,sensor_count=cv.sensor_count,filter_count=cv.filter_count))
    native_mass=[float(c.root_physx_view.get_masses().reshape(-1)[0].cpu()) for c in cubes]
    assert np.allclose(native_mass,masses,rtol=1e-6,atol=1e-8)
    matrix=[];positions=[];velocities=[];details=[];counts_saved=[];starts_saved=[];separations_saved=[]
    for t in range(steps):
        for cube in cubes:cube.write_data_to_sim()
        sim.step(render=False)
        for cube in cubes:cube.update(dt)
        pos=np.stack([c.root_physx_view.get_transforms().detach().cpu().numpy()[0,:3] for c in cubes]);positions.append(pos)
        velocities.append(np.stack([c.root_physx_view.get_velocities().detach().cpu().numpy()[0] for c in cubes]))
        matrices=[];normal_vectors=[];counts_step=[];starts_step=[];separation_step=[]
        for cv in contacts:
            native_data=cv.get_contact_data(dt)
            force_devices.add(str(native_data[0].device))
            raw=[v.detach().clone().cpu().numpy() for v in native_data]
            force,points,normals,separations,counts,starts=raw
            assert all(np.isfinite(x).all() for x in raw)
            counts=counts.reshape(4);starts=starts.reshape(4);assert (counts>=0).all() and counts.sum()<128
            vectors=[];seps=[]
            for n,s in zip(counts,starts):
                n=int(n);s=int(s);assert n==0 or (0<=s and s+n<=128)
                vectors.append((force[s:s+n].reshape(-1,1)*normals[s:s+n]).sum(0) if n else np.zeros(3))
                seps.append(float(separations[s:s+n].min()) if n else 0.)
            matrices.append(cv.get_contact_force_matrix(dt).detach().clone().cpu().numpy().reshape(4,3))
            normal_vectors.append(np.stack(vectors));counts_step.append(counts);starts_step.append(starts);separation_step.append(seps)
        matrix.append(np.stack(matrices));details.append(np.stack(normal_vectors));counts_saved.append(counts_step);starts_saved.append(starts_step);separations_saved.append(separation_step)
    matrix=np.asarray(matrix);positions=np.asarray(positions);velocities=np.asarray(velocities)
    detail=np.asarray(details);counts=np.asarray(counts_saved)
    assert all(np.isfinite(x).all() for x in (matrix,positions,velocities,detail,counts))
    assert force_devices=={args.device},force_devices
    own=np.stack([matrix[:,i,i] for i in range(3)],1)
    mask=np.ones((3,4),bool)
    for i in range(3):mask[i,i]=False
    negative_max=float(np.abs(matrix[:,mask]).max())
    separated=positions[:,:,2]>.12
    separated_max=float(np.abs(own[separated]).max())
    settle=registration['settle_start'];expected=np.asarray(masses)*9.81
    median=np.median(np.abs(own[settle:,:,2]),0);relative=np.abs(median-expected)/expected
    native_z=np.median(positions[settle:,:,2],0);zerr=np.abs(native_z-.1)
    detail_error=float(np.abs(np.abs(detail)-np.abs(matrix)).max())
    result=dict(status='PASS' if negative_max<=1e-6 and separated_max<=1e-6 and (relative<=.05).all() and (zerr<=.003).all() and detail_error<=1e-4 else 'FAIL',
        scope='three primitive native contact controls only; no robot/grasp/full-scene hazard or reliability acceptance',
        physics_steps=steps,cubes=3,dt_s=dt,registered_masses_kg=masses,native_mass_kg=native_mass,
        requested_device=args.device,force_tensor_devices=sorted(force_devices),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        isaacsim_version=importlib.metadata.version('isaacsim'),gpu_name=torch.cuda.get_device_name(torch.device(args.device)),
        expected_weight_n=expected.tolist(),settled_median_abs_vertical_n=median.tolist(),relative_weight_error=relative.tolist(),
        settled_median_native_z_m=native_z.tolist(),settled_height_error_m=zerr.tolist(),
        separated_own_force_max_n=separated_max,wrong_or_other_support_force_max_n=negative_max,
        perpoint_vs_matrix_abs_component_max_error_n=detail_error,identities=identities,
        global_contact_capacity_saturation=False,registration_sha256=hashlib.sha256(args.registration.read_bytes()).hexdigest())
    path=args.out/'native_trace.npz';np.savez_compressed(path,positions=positions,velocities=velocities,force_matrix=matrix,
        reconstructed_normal=detail,contact_counts=counts,start_indices=np.asarray(starts_saved),minimum_separations=np.asarray(separations_saved))
    result['trace_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('CALIBRATION',json.dumps(result),flush=True)
    assert result['status']=='PASS','native calibration gate failed; retain complete trace'
try:main()
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close()
