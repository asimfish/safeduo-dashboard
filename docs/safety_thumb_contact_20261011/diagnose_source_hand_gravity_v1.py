"""Read-only, posthoc hand gravity oracle from all original URDF links."""
from pathlib import Path
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from diagnose_native60_thumb_velocity_kinematics_v1 import rotation_xyzw, axis_rotation, rpy_matrix
from audit_coupled_hand_targets_native_v1 import matrix

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def xyz(element, attribute='xyz'):
    return np.array([float(x) for x in element.get(attribute, '0 0 0').split()]) if element is not None else np.zeros(3)


def main():
    probe = H / 'fourhand_native_probe_v71'
    output = probe / 'POSTHOC_ORIGINAL_HAND_GRAVITY_ORACLE_V1.json'
    assert not output.exists()
    assert read(probe / 'ACTUAL_COMPLETION_EXIT_V71.json')['actual_outer_exit'] == 0
    raw = Path(read(probe / 'RESOURCE_EXECUTION_V1.json')['out'])
    reg = read(probe / 'REGISTRATION_V1.json')
    asset = read(reg['coupled_assets'])['assets']['F_L']
    source = Path(asset['original_URDF'])
    xml = ET.parse(source).getroot()
    links = {link.get('name'): link for link in xml.findall('link')}
    joints = {joint.get('name'): joint for joint in xml.findall('joint')}
    children = {}
    parents = {}
    for name, joint in joints.items():
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        assert child not in parents
        parents[child] = (parent, name)
        children.setdefault(parent, []).append((child, name))
    root = next(iter(set(links) - set(parents)))
    assert root == 'left_hand_base' and len(links) == 26
    masses, coms = {}, {}
    for name, link in links.items():
        inertial = link.find('inertial')
        assert inertial is not None
        masses[name] = float(inertial.find('mass').get('value'))
        coms[name] = xyz(inertial.find('origin'))
    with np.load(raw / 'native_parameters.npz', allow_pickle=False) as z:
        names = z['F_L_joint_names'].tolist()
        bodies = z['F_L_body_names'].tolist()
        native_masses = z['F_L_link_masses'].reshape(-1).astype(float)
    moving = [name for name, j in joints.items() if j.get('type') == 'revolute']
    assert set(moving) == set(asset['hand_joint_names']) and len(moving) == 12
    hand = asset['hand_joint_names']
    roots, mapping, _ = matrix(hand, str(source))
    hand_ids = [names.index(n) for n in hand]
    groups = {name: [] for name in bodies if name in links}
    for name in links:
        owner = name
        while owner not in groups:
            parent, joint_name = parents[owner]
            assert joints[joint_name].get('type') == 'fixed'
            owner = parent
        groups[owner].append(name)
    folded_mass_rows = [dict(body=name, source_links=members,
                            source_total_kg=sum(masses[n] for n in members),
                            native_kg=float(native_masses[bodies.index(name)]))
                        for name, members in groups.items()]
    sources = [Path(__file__), source, raw / 'native_parameters.npz',
               H / 'diagnose_native60_thumb_velocity_kinematics_v1.py',
               H / 'audit_coupled_hand_targets_native_v1.py']
    rows = []
    for chunk in read(raw / 'physics_chunks.json'):
        file = raw / chunk['path']
        assert sha(file) == chunk['sha256']
        sources.append(file)
        with np.load(file, allow_pickle=False) as z:
            for i, event in enumerate(z['physics_event']):
                q = z['F_L_motor_control_q_before'][i, 0].astype(float)
                body_poses = z['F_L_native_body_transforms_before_xyzw'][i, 0].astype(float)
                base = body_poses[bodies.index(root)]
                poses = {root: (base[:3], rotation_xyzw(base[3:]))}
                joint_world = {}
                lineage = {root: []}
                queue = [root]
                while queue:
                    parent = queue.pop(0)
                    position, rotation = poses[parent]
                    for child, joint_name in children.get(parent, []):
                        joint = joints[joint_name]
                        origin = joint.find('origin')
                        pivot = position + rotation @ xyz(origin)
                        joint_rotation = rotation @ rpy_matrix(xyz(origin, 'rpy'))
                        if joint.get('type') == 'fixed':
                            child_rotation = joint_rotation
                            lineage[child] = list(lineage[parent])
                        else:
                            axis = xyz(joint.find('axis'))
                            axis /= np.linalg.norm(axis)
                            joint_world[joint_name] = (pivot, joint_rotation @ axis)
                            child_rotation = joint_rotation @ axis_rotation(axis, q[names.index(joint_name)])
                            lineage[child] = lineage[parent] + [joint_name]
                        poses[child] = (pivot, child_rotation)
                        queue.append(child)
                assert set(poses) == set(links)
                expected = dict.fromkeys(hand, 0.)
                for link, (position, rotation) in poses.items():
                    com_world = position + rotation @ coms[link]
                    force_world = masses[link] * np.array([0., 0., -9.81])
                    for joint_name in lineage[link]:
                        pivot, axis = joint_world[joint_name]
                        expected[joint_name] -= float(np.cross(axis, com_world - pivot) @ force_world)
                gravity = np.array([expected[name] for name in hand])
                native = z['F_L_gravity_compensation_raw'][i, 0, hand_ids].astype(float)
                pose_checks = []
                for body in groups:
                    pos, rot = poses[body]
                    actual = body_poses[bodies.index(body)]
                    pose_checks.append(dict(body=body, position_error_m=float(np.linalg.norm(pos - actual[:3])),
                                            rotation_frobenius_error=float(np.linalg.norm(rot - rotation_xyzw(actual[3:])))))
                rows.append(dict(event=int(event), source_gravity_compensation_Nm=gravity.tolist(),
                                 native_raw_gravity_compensation_Nm=native.tolist(),
                                 source_minus_native_Nm=(gravity - native).tolist(),
                                 source_reduced_motor_gravity_Nm=(mapping.T @ gravity).tolist(),
                                 actual_reduced_motor_gravity_Nm=(mapping.T @ native).tolist(),
                                 max_pose_position_error_m=max(r['position_error_m'] for r in pose_checks),
                                 max_pose_rotation_frobenius_error=max(r['rotation_frobenius_error'] for r in pose_checks)))
    assert len(rows) == 16
    errors = np.array([row['source_minus_native_Nm'] for row in rows])
    reduced = np.array([np.array(row['source_reduced_motor_gravity_Nm']) - np.array(row['actual_reduced_motor_gravity_Nm']) for row in rows])
    result = dict(status='CLOSED_POSTHOC_ORIGINAL_26_LINK_HAND_GRAVITY_ORACLE',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  known_probe_version=71, known_physics_events=len(rows), arm='F_L',
                  native_root_pose_used=True, all_original_fixed_link_masses_and_COMs_included=True,
                  source_hand_joint_order=hand, original_independent_motor_order=roots,
                  maximum_raw_hand_gravity_error_Nm=float(abs(errors).max()),
                  per_joint_maximum_raw_error_Nm=dict(zip(hand, abs(errors).max(axis=0).tolist())),
                  maximum_reduced_hand_gravity_error_Nm=float(abs(reduced).max()),
                  maximum_native_body_position_error_m=max(row['max_pose_position_error_m'] for row in rows),
                  maximum_native_body_rotation_frobenius_error=max(row['max_pose_rotation_frobenius_error'] for row in rows),
                  folded_mass_checks=folded_mass_rows,
                  maximum_folded_mass_error_kg=max(abs(row['source_total_kg'] - row['native_kg']) for row in folded_mass_rows),
                  rows=rows, source_SHA256={str(p): sha(p) for p in sources},
                  native_local_COM_not_captured=True,
                  interpretation='Diagnostic residuals only. No safety or full-task qualification, no original physical gate replacement, no new native experiment.',
                  all9_jointly_accepted=False, formal_holdout_count=0, fullSystem0_accepted=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k.startswith('maximum_') or k in ['status', 'known_physics_events']}), flush=True)


if __name__ == '__main__':
    main()
