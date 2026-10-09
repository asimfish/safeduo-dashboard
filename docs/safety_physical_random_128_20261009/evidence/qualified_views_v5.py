"""Three actual same-run camera views; rendering never advances physics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


class PilotViews:
    def __init__(self, env, out, settings):
        from pxr import UsdGeom
        import omni.usd
        self.env, self.out, self.settings = env, Path(out), settings
        self.visuals = [UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath(
            f'/World/envs/env_{i}')) for i in range(64)]
        self.receipts = []
        self.frames = 0

    def capture(self, step):
        from pxr import UsdGeom, Usd, Gf
        import omni.usd
        import usdrt
        from PIL import Image
        env = self.env
        getters = dict(q='get_dof_positions', qd='get_dof_velocities',
            root='get_root_transforms', root_velocity='get_root_velocities',
            body_transforms='get_link_transforms')
        before = {a + '_' + k: getattr(art.root_physx_view, fn)().clone()
                  for a, art in env._arms.items() for k, fn in getters.items()}
        render_path = self.out / f'native_render_frame_{self.frames:04d}.npz'
        np.savez_compressed(render_path, **{k: v.cpu().numpy() for k, v in before.items()})
        centers = env._sph.last_centers.clone()
        radii = env._sph.radii.cpu().numpy()
        records = []
        for slot in self.settings['slots']:
            for i, visual in enumerate(self.visuals):
                visual.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if i == slot else UsdGeom.Tokens.invisible)
            points = centers[slot]
            center = (points.min(0).values + points.max(0).values) * .5
            for name, offset in [('top', [0, -.033, 1.]), ('front', [0, -1., .217]), ('side', [1., 0, .217]), ('detail', [.6, -.6, .35])]:
                visible_points = points
                visible_radii = radii
                target_center = center
                if name == 'detail':
                    mask = env._sph.arm_id == 2
                    visible_points = points[mask]
                    visible_radii = radii[mask.cpu().numpy()]
                    target_center = (visible_points.min(0).values + visible_points.max(0).values) * .5
                eye = target_center + torch.tensor(offset, device=env.device)
                for attempt in range(20):
                    env._viz_cam.set_world_poses_from_view(eye[None], target_center[None])
                    prim = omni.usd.get_context().get_stage().GetPrimAtPath('/World/viz_cam')
                    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world = np.array(matrix, dtype=float)
                    camera = UsdGeom.Camera(prim)
                    xyz = np.c_[visible_points.cpu().numpy(), np.ones(len(visible_points))] @ np.linalg.inv(world)
                    x, y, z = xyz[:, 0], xyz[:, 1], -xyz[:, 2]
                    K = env._viz_cam.data.intrinsic_matrices[0].cpu().numpy()
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    near, far = map(float, camera.GetClippingRangeAttr().Get())
                    margins = np.stack([(fx*x+cx*z)/np.hypot(fx,cx)-visible_radii,
                        ((1280-cx)*z-fx*x)/np.hypot(fx,1280-cx)-visible_radii,
                        (cy*z-fy*y)/np.hypot(fy,cy)-visible_radii,
                        ((720-cy)*z+fy*y)/np.hypot(fy,720-cy)-visible_radii,
                        z-near-visible_radii, far-z-visible_radii], -1)
                    if (margins >= .05).all(): break
                    eye = target_center + 1.25 * (eye - target_center)
                else: raise ValueError('four-arm camera frustum fit failed')
                env.sim.render()
                native_fabric_position_error = 0.
                native_fabric_orientation_error = 0.
                correspondence_rows = []
                for arm, art in env._arms.items():
                    transforms = before[arm + '_body_transforms'][slot].cpu().numpy()
                    paths = art.root_physx_view.link_paths[slot]
                    assert len(paths) == len(transforms)
                    for path, native in zip(paths, transforms):
                        stage_id = omni.usd.get_context().get_stage_id()
                        render_stage = usdrt.Usd.Stage.Attach(stage_id)
                        body_prim = render_stage.GetPrimAtPath(path)
                        render_xform = usdrt.Rt.Xformable(body_prim)
                        matrix_value = body_prim.GetAttribute('omni:fabric:worldMatrix').Get()
                        if matrix_value is not None:
                            render_matrix = Gf.Matrix4d(*np.asarray(matrix_value).reshape(-1).tolist())
                            translation = np.asarray(render_matrix.ExtractTranslation())
                            rotation = render_matrix.ExtractRotationQuat()
                            render_pose_source = 'Fabric hierarchy worldMatrix'
                        elif render_xform.HasWorldXform():
                            translation = np.asarray(render_xform.GetWorldPositionAttr().Get())
                            rotation = render_xform.GetWorldOrientationAttr().Get()
                            render_pose_source = 'Fabric Rt world position/orientation'
                        else:
                            diagnostic = dict(step=step,path=path,attributes=[str(attr.GetName()) for attr in body_prim.GetAttributes()],native=native.tolist())
                            (self.out / 'missing_render_transform.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                            usd_prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
                            render_matrix = UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            translation = np.asarray(render_matrix.ExtractTranslation())
                            rotation = render_matrix.ExtractRotationQuat()
                            render_pose_source = 'USD authored transform, no Fabric override; must pass same native equality gate'
                        xyzw = np.r_[np.asarray(rotation.GetImaginary()), rotation.GetReal()]
                        xyzw = xyzw / np.linalg.norm(xyzw)
                        position_error = float(abs(translation - native[:3]).max())
                        orientation_error = float(min(abs(xyzw-native[3:]).max(),abs(xyzw+native[3:]).max()))
                        correspondence_rows.append(dict(path=path,render_pose_source=render_pose_source,native=native.tolist(),fabric_translation=translation.tolist(),fabric_xyzw=xyzw.tolist(),position_max_error_m=position_error,orientation_max_error=orientation_error))
                        native_fabric_position_error = max(native_fabric_position_error, position_error)
                        native_fabric_orientation_error = max(native_fabric_orientation_error, orientation_error)
                correspondence = dict(step=step,lane=slot,view=name,max_position_error_m=native_fabric_position_error,max_orientation_error=native_fabric_orientation_error,bodies=correspondence_rows)
                correspondence_path = self.out / f'render_correspondence_{self.frames:04d}_{slot}_{name}.json'
                correspondence_path.write_text(json.dumps(correspondence,indent=2)+'\n')
                assert native_fabric_position_error <= 2e-5, 'Fabric rendering body position differs from native state'
                assert native_fabric_orientation_error <= 2e-5, 'Fabric rendering body orientation differs from native state'
                env._viz_cam.update(env.step_dt, force_recompute=True)
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                assert np.allclose(np.array(matrix.ExtractTranslation()), eye.cpu().numpy(), atol=1e-5, rtol=0)
                rgb = env._viz_cam.data.output['rgb'][0].cpu().numpy()
                if rgb.dtype != np.uint8: rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                path = self.out / 'native_views' / f'env_{slot:03d}' / name / f'frame_{self.frames:04d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb[..., :3]).save(path)
                records.append(dict(lane=slot, view=name, path=str(path.relative_to(self.out)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    actual_camera_world_row_matrix=np.array(matrix).tolist(),
                    intrinsic_matrix=K.tolist(), clip_m=[near, far],
                    all_selected_spheres_contained=bool((margins >= .05).all()),
                    minimum_plane_margin_m=float(margins.min()), fit_attempts=attempt+1, selected_spheres=len(visible_radii),
                    native_fabric_all_body_position_max_error_m=native_fabric_position_error,
                    native_fabric_all_body_orientation_max_error=native_fabric_orientation_error))
        for a, art in env._arms.items():
            for k, fn in getters.items():
                assert torch.equal(getattr(art.root_physx_view, fn)(), before[a + '_' + k]), 'render changed native state'
        assert torch.equal(env._sph.last_centers, centers)
        self.receipts.append(dict(frame=self.frames, step=step,
            native_state_path=str(render_path.relative_to(self.out)),
            native_state_sha256=hashlib.sha256(render_path.read_bytes()).hexdigest(),
            render_did_not_advance_physics=True, images=records))
        self.frames += 1
        print('PILOT_CAMERAS', step, 'images', len(records), flush=True)

    def close(self):
        (self.out / 'pilot_camera_receipts.json').write_text(json.dumps(dict(status='complete',
            frames=self.frames, images=sum(len(r['images']) for r in self.receipts), slots=self.settings['slots'],
            views=self.settings['views'], receipts=self.receipts,
            scope='Same native run. All represented four-arm spheres inside actual frustum; visibility does not prove geometry/force safety.'), indent=2) + '\n')
