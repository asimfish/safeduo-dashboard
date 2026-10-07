"""Render-only reconstruction. Never generates new dynamics/force evidence."""
import argparse,json,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
pa=argparse.ArgumentParser();pa.add_argument('--registration',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);AppLauncher.add_app_launcher_args(pa);a=pa.parse_args();r=json.loads(a.registration.read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for f,h in r['source_sha256'].items():assert sha(f)==h,f
a.out.mkdir(exist_ok=False);a.enable_cameras=True;app=AppLauncher(a).app
import numpy as np,torch
from PIL import Image
import isaaclab.sim as sim_utils
from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
from safeduo.safety.types import ARM_KEYS
npv=lambda v:v.detach().clone().cpu().numpy()
def save(fn,obj):(a.out/fn).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(1);cfg=make_duo_env_cfg(num_envs=2,device=a.device,yaml_name=r['env_yaml'],coordinator=True);cfg.seed=r['seed'];cfg.enable_viz_camera=True;cfg.scene.env_spacing=5.;cfg.scene_dressing['table_visible']=True;cfg.scene_dressing['ground_visible']=True;cfg.scene_dressing['static_props']=[v for v in cfg.scene_dressing.get('static_props',[]) if v.get('collision',True)]
 for arm,ac in cfg.robot_cfgs.items():
  if arm.startswith('U'):ac.init_state.joint_pos={**ac.init_state.joint_pos,'.*thumb_1_joint':.35}
 env=DuoEnv(cfg);cam=env._viz_cam;env.scene.sensors.pop('viz_cam');cam.reset();env.reset();source=Path(r['source_raw']);meta=json.loads((source/'native_metadata.json').read_text());receipt=json.loads((source/'recording_receipt.json').read_text());ident=json.loads((source/'contact_identities.json').read_text());srcorig=np.asarray(meta['origins']);orig=npv(env.scene.env_origins);images=[];checks=[];cached={};static_time=float(env.sim.current_time)
 for step in r['steps']:
  ch=next(x for x in receipt['chunks'] if x['first_step']<=step<=x['last_step']);fn=ch['file']
  if fn not in cached:
   assert sha(source/fn)==ch['sha256'];z=np.load(source/fn);cached={fn:{k:z[k].copy() for k in z.files}};z.close()
  z=cached[fn];ix=step-ch['first_step'];expected={}
  for arm in ARM_KEYS:
   art=env._arms[arm];assert art.joint_names==meta['arms'][arm]['joint_names'] and art.body_names==meta['arms'][arm]['body_names'];q=z[arm+':q'][ix,r['source_envs']];qd=z[arm+':qd'][ix,r['source_envs']];links=z[arm+':link_pose'][ix,r['source_envs']].copy();links[:,:,:3]=links[:,:,:3]-srcorig[r['source_envs'],None,:]+orig[:,None,:];expected[arm]=links
   root=links[:,0].copy();root[:,3:]=root[:,[6,3,4,5]];art.write_root_pose_to_sim(torch.tensor(root,device=env.device));art.write_joint_state_to_sim(torch.tensor(q,device=env.device),torch.tensor(qd,device=env.device));art.set_joint_position_target(torch.tensor(z[arm+':target'][ix,r['source_envs']],device=env.device))
  for obj in ['beam700','beam300']:
   pose=np.stack([z[f'e{e}:{obj}:pose'][ix] for e in r['source_envs']]).copy();pose[:,:3]=pose[:,:3]-srcorig[r['source_envs']]+orig;pose[:,3:]=pose[:,[6,3,4,5]];env._objects[obj].write_root_pose_to_sim(torch.tensor(pose,device=env.device));env._objects[obj].write_root_velocity_to_sim(torch.tensor(np.stack([z[f'e{e}:{obj}:velocity'][ix] for e in r['source_envs']]),device=env.device))
  env.sim.forward();poserr=0.;quaterr=0.
  for arm in ARM_KEYS:
   actual=npv(env._arms[arm].root_physx_view.get_link_transforms());want=expected[arm];poserr=max(poserr,float(abs(actual[:,:,:3]-want[:,:,:3]).max()));quaterr=max(quaterr,float(np.minimum(abs(actual[:,:,3:]-want[:,:,3:]).max(axis=2),abs(actual[:,:,3:]+want[:,:,3:]).max(axis=2)).max()))
  assert poserr<=r['max_link_position_error_m'] and quaterr<=r['max_link_quaternion_component_error'],('reconstruction',step,poserr,quaterr)
  before={arm:npv(env._arms[arm].root_physx_view.get_dof_positions()) for arm in ARM_KEYS};before.update({o:npv(env._objects[o].root_physx_view.get_transforms()) for o in ['beam700','beam300']})
  for e,srcenv in enumerate(r['source_envs']):
   paths=next(x['sensors'] for x in ident['hands'] if x['env']==srcenv and x['arm']=='U_L');bodyids=[env._arms['U_L'].body_names.index(x.rsplit('/',1)[-1]) for x in paths];center=env._arms['U_L'].root_physx_view.get_link_transforms()[e,bodyids,:3].mean(0)
   cam.set_world_poses_from_view((center+torch.tensor(r['eye_offset_m'],device=env.device))[None],center[None]);env.sim.render();cam.update(env.cfg.sim.dt,force_recompute=True);rgb=cam.data.output['rgb'][0].cpu().numpy()[:,:,:3]
   if rgb.dtype!=np.uint8:rgb=(rgb*255).clip(0,255).astype(np.uint8)
   rel=f'images/e{srcenv}_s{step:04d}_clear_detail.png';dest=a.out/rel;dest.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(dest);images.append(dict(file=rel,sha256=sha(dest),source_env=srcenv,source_step=step,source_chunk=fn,source_chunk_sha256=ch['sha256'],source_native_state_time_s=(step+1)*r['physics_dt_s'],camera_eye_m=npv(center+torch.tensor(r['eye_offset_m'],device=env.device)).tolist(),camera_target_m=npv(center).tolist(),visualization='archived_state_render_only'))
  after={arm:npv(env._arms[arm].root_physx_view.get_dof_positions()) for arm in ARM_KEYS};after.update({o:npv(env._objects[o].root_physx_view.get_transforms()) for o in ['beam700','beam300']});assert all(np.array_equal(before[k],after[k]) for k in before);assert float(env.sim.current_time)==static_time;checks.append(dict(source_step=step,link_position_max_error_m=poserr,link_quaternion_component_max_error=quaterr,physics_time=static_time,render_did_not_change_q_or_objects=True));save('progress.json',dict(frames=len(images),source_step=step));print('CAMERA_REPLAY',step,len(images),flush=True)
 save('recording_receipt.json',dict(status='PASS_RENDER_ONLY_ARCHIVED_STATE_RECONSTRUCTION',registration_sha256=sha(a.registration),source_receipt_sha256=sha(source/'recording_receipt.json'),new_physics_steps=0,new_safety_trials=0,images=images,checks=checks));env.close()
try:main()
except BaseException:
 import traceback
 (a.out/'failure.txt').write_text(traceback.format_exc());raise
finally:sim_utils.SimulationContext.clear_instance();app.close()
