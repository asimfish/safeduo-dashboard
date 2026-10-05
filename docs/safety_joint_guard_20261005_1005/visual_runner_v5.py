"""Nine actual views across all six pair strata and a global slot; before/after native receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
import risk_runner as risk
import guard_runner_v4 as mechanism
from safeduo.safety.types import ARM_KEYS

SLOTS = [0, 8, 16, 24, 32, 40, 56]
CAPTURE_STEPS = [71, 75, 959]


class VisualTrace(mechanism.GuardTrace):
    def start(self, env):
        super().start(env)
        from pxr import UsdGeom
        import omni.usd
        self.visuals = [UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath(f'/World/envs/env_{e}'))
                        for e in range(env.num_envs)]
        self.receipts = []
        assert env._viz_cam is not None

    def capture(self, env, t):
        from pxr import UsdGeom, Usd
        import omni.usd
        before = {a: (env._arms[a].data.joint_pos.clone(), env._arms[a].data.joint_vel.clone(),
                      env._arms[a].data.root_state_w.clone()) for a in ARM_KEYS}
        native_before = {a: {key: getter().clone() for key,getter in dict(
            q=env._arms[a].root_physx_view.get_dof_positions,
            qd=env._arms[a].root_physx_view.get_dof_velocities,
            root=env._arms[a].root_physx_view.get_root_transforms,
            root_vel=env._arms[a].root_physx_view.get_root_velocities).items()} for a in ARM_KEYS}
        np.savez_compressed(self.out / f'native_render_step_{t:04d}_before.npz',
                            **{a+'_'+k:v.cpu().numpy() for a,values in native_before.items() for k,v in values.items()})
        centers = env._sph.last_centers.clone()
        full = env._last_out
        for slot in SLOTS:
            for e, v in enumerate(self.visuals):
                v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if e == slot else UsdGeom.Tokens.invisible)
            origin = env.scene.env_origins[slot]
            points = centers[slot] - origin
            center = (points.min(0).values + points.max(0).values) * .5
            f = points[env._sph.arm_id < 2]
            u = points[env._sph.arm_id >= 2]
            fc, uc = (f.min(0).values + f.max(0).values) * .5, (u.min(0).values + u.max(0).values) * .5
            # Wide overview encloses the full four-arm sphere envelope. The
            # three U angles retain the opposing side and lower shoulder view.
            def pose(target, offset):
                return dict(eye=(target + torch.tensor(offset, device=env.device)).cpu().tolist(),
                            target=target.cpu().tolist())
            views = dict(overview=pose(center, [3.6,-4.5,3.3]),
                         front=pose(center,[0,-4.8,1.5]), reverse=pose(center,[0,4.8,1.8]),
                         f_pair=pose(fc,[2.3,-2.8,1.4]), f_opposite_low=pose(fc,[-2.2,2.8,.4]),
                         f_opposite_high=pose(fc,[-2.2,2.8,1.8]), u_pair=pose(uc,[2.3,-2.8,1.4]),
                         u_opposite_low=pose(uc,[-2.2,2.8,.4]), u_opposite_high=pose(uc,[-2.2,2.8,1.8]))
            root = self.out / 'multiview' / f'env_{slot:03d}'
            root.mkdir(parents=True, exist_ok=True)
            images = []
            for name, v in views.items():
                # Geometry-only camera-distance fitting, fixed before any replay.
                # Keep each prescribed viewing direction; fit every target sphere
                # with5cm plane clearance before rendering/saving that view.
                fit_mask = (env._sph.arm_id < 2) if name.startswith('f_') else ((env._sph.arm_id >= 2) if name.startswith('u_') else torch.ones_like(env._sph.arm_id,dtype=torch.bool))
                fit_points = centers[slot,fit_mask].cpu().numpy()
                fit_radii = env._sph.radii[fit_mask].cpu().numpy()
                for fit_attempt in range(8):
                    env._viz_cam.set_world_poses_from_view(
                        (torch.tensor(v['eye'],device=env.device)+origin)[None],
                        (torch.tensor(v['target'],device=env.device)+origin)[None])
                    fit_prim = omni.usd.get_context().get_stage().GetPrimAtPath('/World/viz_cam')
                    fit_world = np.array(UsdGeom.Xformable(fit_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=np.float64)
                    fit_camera = np.concatenate([fit_points,np.ones((len(fit_points),1))],1) @ np.linalg.inv(fit_world)
                    fit_K = env._viz_cam.data.intrinsic_matrices[0].cpu().numpy()
                    xx,yy,zz=fit_camera[:,0],fit_camera[:,1],-fit_camera[:,2]
                    fx,fy,cx,cy=fit_K[0,0],fit_K[1,1],fit_K[0,2],fit_K[1,2]
                    fit_planes=np.stack([(fx*xx+cx*zz)/np.hypot(fx,cx)-fit_radii,
                        ((1280-cx)*zz-fx*xx)/np.hypot(fx,1280-cx)-fit_radii,
                        (cy*zz-fy*yy)/np.hypot(fy,cy)-fit_radii,
                        ((720-cy)*zz+fy*yy)/np.hypot(fy,720-cy)-fit_radii,zz-fit_radii],-1)
                    if (fit_planes >= .050).all():
                        break
                    direction = np.array(v['eye'])-np.array(v['target'])
                    v['eye']=(np.array(v['target'])+1.25*direction).tolist()
                else:
                    raise ValueError('camera target-sphere fit failed; retain partial replay')
                v['auto_distance_fit']=dict(geometry_only=True,attempts=fit_attempt+1,
                    target_plane_clearance_m=.050,distance_multiplier=1.25**fit_attempt)
                env._viz_cam.set_world_poses_from_view(
                    (torch.tensor(v['eye'], device=env.device)+origin)[None],
                    (torch.tensor(v['target'], device=env.device)+origin)[None])
                env.sim.render()
                env._viz_cam.update(env.step_dt, force_recompute=True)
                # Read the actual camera prim; keep the erroneous SDK view
                # readback separate, rather than relabel intended coordinates.
                camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath('/World/viz_cam')
                matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world = np.array(matrix, dtype=np.float64)
                actual = np.array(matrix.ExtractTranslation(), dtype=np.float64)
                expected = np.array(v['eye']) + origin.cpu().numpy()
                assert np.allclose(actual, expected, atol=1e-5, rtol=0), ('actual camera pose',actual,expected)
                rotation = matrix.ExtractRotationQuat()
                v['actual_position_world_m'] = actual.tolist()
                v['actual_quat_opengl_wxyz'] = [float(rotation.GetReal()),*map(float,rotation.GetImaginary())]
                v['actual_camera_to_world_row_matrix'] = world.tolist()
                v['actual_pose_source'] = 'USD /World/viz_cam ComputeLocalToWorldTransform'
                v['sdk_position_world_m'] = env._viz_cam.data.pos_w[0].cpu().tolist()
                v['sdk_quat_world_wxyz'] = env._viz_cam.data.quat_w_world[0].cpu().tolist()
                v['sdk_position_matches_actual'] = bool(np.allclose(v['sdk_position_world_m'],actual,atol=1e-5,rtol=0))
                camera = UsdGeom.Camera(camera_prim)
                v['actual_usd_optics'] = dict(focal_length=float(camera.GetFocalLengthAttr().Get()),
                    horizontal_aperture=float(camera.GetHorizontalApertureAttr().Get()),
                    vertical_aperture=float(camera.GetVerticalApertureAttr().Get()),
                    horizontal_aperture_offset=float(camera.GetHorizontalApertureOffsetAttr().Get()),
                    vertical_aperture_offset=float(camera.GetVerticalApertureOffsetAttr().Get()))
                v['actual_intrinsic_matrix'] = env._viz_cam.data.intrinsic_matrices[0].cpu().tolist()
                center_indices = (env._sph.arm_id < 2) if name.startswith('f_') else ((env._sph.arm_id >= 2) if name.startswith('u_') else torch.ones_like(env._sph.arm_id,dtype=torch.bool))
                pts = centers[slot,center_indices].cpu().numpy()
                camera_points = np.concatenate([pts,np.ones((len(pts),1))],1) @ np.linalg.inv(world)
                depth = -camera_points[:,2]
                K = np.array(v['actual_intrinsic_matrix'])
                pixels = np.column_stack([K[0,0]*camera_points[:,0]/depth+K[0,2],K[1,2]-K[1,1]*camera_points[:,1]/depth])
                v['observed_center_projection'] = dict(count=len(pts),depth_min_m=float(depth.min()),
                    pixel_min=pixels.min(0).tolist(),pixel_max=pixels.max(0).tolist(),
                    all_inside=bool((depth>0).all() and (pixels[:,0]>=0).all() and (pixels[:,0]<1280).all() and (pixels[:,1]>=0).all() and (pixels[:,1]<720).all()))
                assert v['observed_center_projection']['all_inside'], ('insufficient view',name,v['observed_center_projection'])
                radii = env._sph.radii[center_indices].cpu().numpy()
                x, y = camera_points[:, 0], camera_points[:, 1]
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                plane_margins = np.stack([
                    (fx*x+cx*depth)/np.hypot(fx,cx)-radii,
                    ((1280-cx)*depth-fx*x)/np.hypot(fx,1280-cx)-radii,
                    (cy*depth-fy*y)/np.hypot(fy,cy)-radii,
                    ((720-cy)*depth+fy*y)/np.hypot(fy,720-cy)-radii,
                    depth-radii], -1)
                v['observed_sphere_frustum'] = dict(spheres=len(radii),
                    all_contained=bool((plane_margins>0).all()),
                    minimum_plane_margin_m=float(plane_margins.min()),
                    scope='represented target-group spheres inside all view planes; silhouette and occlusion not certified')
                assert v['observed_sphere_frustum']['all_contained'], ('sphere clipped',name)
                rgb = env._viz_cam.data.output['rgb'][0].cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb*255,0,255).astype(np.uint8)
                path = root / f'step_{t:04d}_{name}.png'
                Image.fromarray(rgb[:,:,:3]).save(path)
                images.append(dict(path=str(path.relative_to(self.out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            assert all(torch.equal(env._arms[a].data.joint_pos, before[a][0]) and
                       torch.equal(env._arms[a].data.joint_vel, before[a][1]) and
                       torch.equal(env._arms[a].data.root_state_w, before[a][2]) for a in ARM_KEYS)
            assert torch.equal(env._sph.last_centers,centers)
            native_after = {a: {key:getter().clone() for key,getter in dict(
                q=env._arms[a].root_physx_view.get_dof_positions,
                qd=env._arms[a].root_physx_view.get_dof_velocities,
                root=env._arms[a].root_physx_view.get_root_transforms,
                root_vel=env._arms[a].root_physx_view.get_root_velocities).items()} for a in ARM_KEYS}
            assert all(torch.equal(native_before[a][k],native_after[a][k]) for a in ARM_KEYS for k in native_before[a]), 'render changed native PhysX state'
            after_path = self.out / f'native_render_step_{t:04d}_after_env_{slot:03d}.npz'
            np.savez_compressed(after_path, **{a+'_'+k:value.cpu().numpy() for a,values in native_after.items() for k,value in values.items()})
            state = dict(schema='safeduo.pressure_nine_view.v2',env_id=slot,step=t,
                         state_time_s=(t+1)*env.step_dt,environment_count=64,
                         render_state_unchanged_all_64=True,visibility_only=True,views=views,images=images,
                         fresh_native_physx_unchanged_all_64=True,
                         visual_mode=__import__('os').environ['SAFEDUO_JOINT_MODE'],
                         fresh_native_snapshot=f'native_render_step_{t:04d}_before.npz',
                         fresh_native_before_sha256=hashlib.sha256((self.out / f'native_render_step_{t:04d}_before.npz').read_bytes()).hexdigest(),
                         fresh_native_after_snapshot=str(after_path.relative_to(self.out)),
                         fresh_native_after_sha256=hashlib.sha256(after_path.read_bytes()).hexdigest(),
                         fresh_native_all_joint_names={a:env._arms[a].joint_names for a in ARM_KEYS},
                         fresh_native_controlled_joint_indices={a:env._joint_idx[a].cpu().tolist() for a in ARM_KEYS},
                         fresh_native_selected={a:{k:value[slot].cpu().tolist() for k,value in values.items()} for a,values in native_before.items()},
                         origin_world_m=origin.cpu().tolist(),
                         arms={a:dict(q=before[a][0][slot,env._joint_idx[a]].cpu().tolist(),
                                      qd=before[a][1][slot,env._joint_idx[a]].cpu().tolist()) for a in ARM_KEYS},
                         sphere_centers_world_m=centers[slot].cpu().tolist(),
                         full_distances_m=full.dists[slot].cpu().tolist(),
                         full_braking_dmin_m=full.full_dmin[slot].cpu().tolist(),
                         full_closing_m_s=full.closing[slot].cpu().tolist(),
                         full_row_identity_file='full_row_identity.json',
                         pending_actuator_targets=[{a:value[a][slot].cpu().tolist() for a in ARM_KEYS} for value in env._evaluation_actuator_delay.queue.pending],
                         sphere_radii_m=env._sph.radii.cpu().tolist(),
                         full_exempt=full.full_viol_exempt[slot].cpu().tolist(),
                         controller_target={a:env._targets[a][slot].cpu().tolist() for a in ARM_KEYS},
                         actuator_target={a:env._evaluation_actuator_delay.applied[a][slot].cpu().tolist() for a in ARM_KEYS})
            path = root / f'step_{t:04d}_state.json'
            path.write_text(json.dumps(state,indent=2)+'\n')
            self.receipts.append(dict(env_id=slot,step=t,state=str(path.relative_to(self.out)),images=images,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'SEVEN_VIEW step={t} slots={len(SLOTS)}',flush=True)

    def step(self, env, t):
        super().step(env,t)
        if t in CAPTURE_STEPS:
            self.capture(env,t)

    def write(self,*args,**kwargs):
        assert len(self.receipts)==21
        (self.out/'camera_receipts.json').write_text(json.dumps(dict(groups=21,PNG=189,receipts=self.receipts),indent=2)+'\n')
        return super().write(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--ckpt',required=True)
    parser.add_argument('--env-yaml',required=True)
    parser.add_argument('--seeds',type=int,default=1701627244)
    parser.add_argument('--methods',default='system0')
    parser.add_argument('--duration-s',type=float,default=16.)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args).app
    from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
    from safeduo.eval.block1_harness import PolicyDriver
    from safeduo.eval.clutch import wrap_clutch
    from safeduo.eval.clutch_eval import SAFE_LINES
    from safeduo.eval.endurance_eval import ckpt_arm_aware,ckpt_p2_obs
    from safeduo.eval.perturbations import TargetDelayAdapter
    from safeduo.configs import load_config
    cfg=make_duo_env_cfg(num_envs=64,device=args.device,yaml_name=args.env_yaml,coordinator=True,
                         arm_aware_obs=ckpt_arm_aware(args.ckpt),p2_obs=ckpt_p2_obs(args.ckpt))
    cfg.coordinator['terminate_on_violation']=False;cfg.episode_length_s=21.;cfg.seed=0;cfg.enable_viz_camera=True
    env=DuoEnv(cfg)
    # Shared world camera has one sensor row, independent of64 reset IDs.
    assert env.scene.sensors.pop("viz_cam") is env._viz_cam
    env._viz_cam.reset()
    dt=cfg.sim.dt*cfg.decimation;steps=round(args.duration_s/dt)
    env._delta_src=risk.RiskSource(env,'risk_burst',.05,risk.base.battery.cell_seed(args.seeds,'risk_burst',.05),steps)
    env._evaluation_actuator_delay=TargetDelayAdapter(env,6)
    driver=wrap_clutch(PolicyDriver(args.ckpt,device=str(env.device)),True,.5,.2,64,env.device)
    protocol=dict(schema='safeduo.visual_development.v1',args=vars(args),steps=steps,dt=dt,status='running',
                  resolved_config=load_config(args.env_yaml),actor_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
                  scope='separate replay of first fresh heldout bank/tape; each image binds own visual state; no extra unique tapes or object-carry success',
                  slots=SLOTS,capture_steps=CAPTURE_STEPS,observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    risk.base.battery.atomic_json(out/'visual_protocol.json',protocol)
    try:
        trace=VisualTrace(spheres=True)
        acc=risk.base.battery.run_window(env,driver,steps,dict(SAFE_LINES),.08,.005,log_every=300,observer=trace)
        meta=dict(method='system0',flow='risk_burst',seed=args.seeds,amp=.05,dt=dt,actuator_delay_steps=6)
        episodes=trace.write(out/'cell_001.npz',acc,meta,env._delta_src.coverage_metadata())
        protocol.update(status='complete',completed_windows=64,violations=sum(e['violation'] for e in episodes))
        (out/'episodes.json').write_text(json.dumps(episodes,indent=2)+'\n')
    except BaseException as e:
        protocol.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        risk.base.battery.atomic_json(out/'visual_protocol.json',protocol)
        env._evaluation_actuator_delay.close();env.close();app.close()


if __name__=='__main__':main()
