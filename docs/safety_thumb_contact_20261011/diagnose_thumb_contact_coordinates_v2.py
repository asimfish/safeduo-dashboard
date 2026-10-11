"""Posthoc native point ownership and original thumb contact Jacobian."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from diagnose_native60_thumb_velocity_kinematics_v1 import rotation_xyzw, rpy_matrix

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output = H / 'THUMB_CONTACT_COORDINATES_POSTHOC_V2.json'
    assert not output.exists()
    sources = [Path(__file__), H / 'diagnose_native60_thumb_velocity_kinematics_v1.py']
    cases = []
    for version in [66, 69, 70, 71, 73]:
        p = H / f'fourhand_native_probe_v{version}'
        assert read(p / f'ACTUAL_COMPLETION_EXIT_V{version}.json')['actual_outer_exit'] == 0
        raw = Path(read(p / 'RESOURCE_EXECUTION_V1.json')['out'])
        reg = read(p / 'REGISTRATION_V1.json')
        asset = read(reg['coupled_assets'])['assets']['F_L']
        urdf = Path(asset['original_URDF'])
        sources.extend([urdf, raw / 'native_parameters.npz', raw / 'native_task_identity.json'])
        xml = ET.parse(urdf).getroot()
        joints = {j.get('name'): j for j in xml.findall('joint')}
        source_thumb = [f'left_thumb_{i}_joint' for i in range(1, 5)]
        scales = [1., 1., .7222, .7222 * .69754]
        with np.load(raw / 'native_parameters.npz', allow_pickle=False) as params:
            names = params['F_L_joint_names'].tolist()
            bodies = params['F_L_body_names'].tolist()
        partners = read(raw / 'native_task_identity.json')['partners']
        events, positive_points, owned_bodies = 0, [], set()
        for chunk in read(raw / 'physics_chunks.json'):
            file = raw / chunk['path']
            assert sha(file) == chunk['sha256']
            sources.append(file)
            with np.load(file, allow_pickle=False) as z:
                def link_pose(body, poses):
                    if body in bodies:
                        return poses[bodies.index(body)]
                    fixed = next(j for j in joints.values() if j.find('child').get('link') == body)
                    assert fixed.get('type') == 'fixed'
                    parent = link_pose(fixed.find('parent').get('link'), poses)
                    origin = fixed.find('origin')
                    offset = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
                    angles = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
                    rotation = rotation_xyzw(parent[3:])
                    # Keep a matrix for folded fixed parents, no quaternion roundtrip.
                    return (parent[:3] + rotation @ offset, rotation @ rpy_matrix(angles))
                offsets = z['object_normal_event_offsets']
                events += len(z['physics_event'])
                for i, event in enumerate(z['physics_event']):
                    for k in range(int(offsets[i]), int(offsets[i + 1])):
                        path = partners[int(z['object_normal_partner_indices'][k])]
                        if '/F_L/' not in path:
                            continue
                        force = float(z['object_normal_normal_forces'][k, 0])
                        if abs(force) == 0:
                            continue
                        body = path.rsplit('/', 1)[-1]
                        owned_bodies.add(body)
                        assert body == 'left_thumb_4'
                        point = z['object_normal_points'][k].astype(float)
                        normal = z['object_normal_normals'][k].astype(float)
                        poses = z['F_L_native_body_transforms_after_xyzw'][i, 0].astype(float)
                        columns = []
                        for name in source_thumb:
                            joint = joints[name]
                            parent = joint.find('parent').get('link')
                            pose = link_pose(parent, poses)
                            if isinstance(pose, tuple):
                                position, rotation = pose
                            else:
                                position, rotation = pose[:3], rotation_xyzw(pose[3:])
                            origin = joint.find('origin')
                            origin_xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
                            origin_rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
                            pivot = position + rotation @ origin_xyz
                            axis = np.array([float(x) for x in joint.find('axis').get('xyz').split()])
                            axis /= np.linalg.norm(axis)
                            world_axis = rotation @ rpy_matrix(origin_rpy) @ axis
                            columns.append(np.cross(world_axis, point - pivot))
                        reduced = [columns[0], sum(scales[j] * columns[j] for j in range(1, 4))]
                        hand_pd = z['F_L_software_full_motor_PD_preclip'][i, 0]
                        positive_points.append(dict(event=int(event), body=body, signed_native_normal_N=force,
                                                    point_world_m=point.tolist(), separation_m=float(z['object_normal_separations'][k, 0]),
                                                    recorded_normal_world=normal.tolist(),
                                                    original_root_contact_normal_J_m_per_rad={source_thumb[j]: float(reduced[j] @ normal) for j in range(2)},
                                                    actual_before_step_PD_Nm={name: float(hand_pd[names.index(name)]) for name in source_thumb[:2]}))
        cases.append(dict(version=version, physics_events=events, all_F_L_positive_point_bodies=sorted(owned_bodies),
                          positive_point_count=len(positive_points), positive_points=positive_points))
    result = dict(status='CLOSED_POSTHOC_NATIVE_THUMB4_CONTACT_COORDINATE_DIAGNOSIS', cases=cases,
                  contact_points_and_normals_and_forces_are_native_recorded=True,
                  Jacobian_uses_original_URDF_axes_and_after_step_native_parent_poses=True,
                  PD_terms_explicitly_before_same_step=True,
                  model_limits='Kinematic diagnostic only. Tiny source/native thumb1 axis binding mismatch remains; no native local COM witness. Single known initial state and weak contact, not causal proof or qualified grasp.',
                  source_sha256={str(f): sha(f) for f in sources}, formal_holdout_count=0, fullSystem0_accepted=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], [(c['version'], c['physics_events'], c['positive_point_count'], c['all_F_L_positive_point_bodies']) for c in cases], flush=True)


if __name__ == '__main__':
    main()
