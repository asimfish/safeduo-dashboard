"""Export closed refined-clock captures, retaining every native physics event."""
from pathlib import Path
import hashlib
import json
import shutil

H = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(version, root):
    p = H / f'fourhand_native_probe_v{version}'
    audit = read(p / 'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    actual = read(p / f'ACTUAL_COMPLETION_EXIT_V{version}.json')
    assert actual['actual_outer_exit'] == 0
    assert audit['record_integrity_verified'] and audit['all_owned_native_children_reaped']
    resource = read(p / 'RESOURCE_EXECUTION_V1.json')
    assert resource['actual_exit'] == resource['actual_native_exit'] == 0
    raw = Path(resource['out'])
    dest = Path(root) / f'fourhand_native_v{version}'
    dest.mkdir(exist_ok=False)
    sources = []

    def keep(src, rel):
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists()
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        sources.append(dict(path=rel, source=str(src), sha256=sha(src)))

    for name in ['fourhand_protocol.json', 'native_parameters.npz',
                 'native_task_identity.json', 'native_contact_identity.json',
                 'point_contact_receipts.json', 'native_contact_receipts.json',
                 'physics_chunks.json', 'camera_frames.json', 'native_solver_roots.json',
                 'native_scene_settings.json', 'native_mimic_identity.json']:
        keep(raw / name, name)
    for row in read(raw / 'physics_chunks.json'):
        keep(raw / row['path'], row['path'])
    for row in read(raw / 'point_contact_receipts.json')['chunks']:
        keep(raw / row['path'], row['path'])

    checks = [('INDEPENDENT_CAPTURE_AUDIT_V1.json', 'INDEPENDENT_AUDIT_ACTUAL_EXIT_V1.json'),
              ('INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json', 'INDEPENDENT_NATIVE_MIMIC_WINDOW_ACTUAL_EXIT_V6.json'),
              ('INDEPENDENT_COUPLED_HAND_TARGETS_V2.json', 'INDEPENDENT_COUPLED_HAND_TARGETS_ACTUAL_EXIT_V2.json'),
              ('INDEPENDENT_COUPLED_MOTOR_EFFORT_V1.json', 'INDEPENDENT_COUPLED_MOTOR_EFFORT_ACTUAL_EXIT_V1.json'),
              ('INDEPENDENT_CAMERA_FRAMING_V2.json', 'INDEPENDENT_CAMERA_FRAMING_ACTUAL_EXIT_V2.json'),
              ('INDEPENDENT_FEEDBACK_GRIP_AUDIT_V4.json', 'INDEPENDENT_FEEDBACK_GRIP_ACTUAL_EXIT_V4.json'),
              ('INDEPENDENT_PRELIFT_ADMISSION_V1.json', 'INDEPENDENT_PRELIFT_ACTUAL_EXIT_V1.json'),
              ('INDEPENDENT_PHASE_ADMISSION_V1.json', 'INDEPENDENT_PHASE_ADMISSION_ACTUAL_EXIT_V1.json')]
    prefix = 'TIME_REFINEMENT' if version == 29 else 'MOTOR_DAMPING'
    checks.append((f'INDEPENDENT_{prefix}_COUNTERFACTUAL_V1.json',
                   f'INDEPENDENT_{prefix}_COUNTERFACTUAL_ACTUAL_EXIT_V1.json'))
    for result, receipt in checks:
        assert read(p / receipt)['actual_outer_exit'] == 0, receipt
        keep(p / result, result)
        keep(p / receipt, receipt)
    for name in ['REGISTRATION_V1.json', 'INDEPENDENT_TASK_METRICS_V1.npz',
                 f'ACTUAL_COMPLETION_EXIT_V{version}.json', 'RESOURCE_EXECUTION_V1.json']:
        keep(p / name, name)

    media = []
    for sub, ext in [('video_evidence_v5', 'mp4'), ('video_webm_v3', 'webm')]:
        custody = read(p / sub / 'VIDEO_CUSTODY_V1.json')
        assert read(p / sub / 'ACTUAL_ENCODER_EXIT_V1.json')['actual_outer_exit'] == 0
        assert custody['native_probe_version'] == version and len(custody['views']) == 4
        for row in custody['views']:
            assert sha(row['path']) == row['sha256']
            assert row['source_frames'] * 4 == audit['camera_frames']
            keep(Path(row['path']), row['view'] + '.' + ext)
            media.append(dict(view=row['view'], format=ext,
                              frames=row['source_frames'],
                              maximum_timestamp_error_s=row['max_timestamp_error_s']))
        keep(p / sub / 'VIDEO_CUSTODY_V1.json', ext + '_VIDEO_CUSTODY_V1.json')
        keep(p / sub / 'ACTUAL_ENCODER_EXIT_V1.json', ext + '_ACTUAL_ENCODER_EXIT_V1.json')
    for name in ['top', 'front', 'side', 'overview']:
        keep(raw / 'frames' / f"{name}_{audit['completed_controls']:05d}.png", name + '_last.png')
    bundle = dict(status='CLOSED_KNOWNCASE_NATIVE_CAPTURE_WITH_ALL_PHYSICS_EVENTS',
                  version=version, controls=audit['completed_controls'],
                  physics_events=audit['physics_events'],
                  physics_substeps_per_control=audit['physics_substeps_per_control'],
                  actual_native_duration_s=audit['observed_duration_s'],
                  all9021_geometry_arrays_public=True, all82_raw_point_arrays_public=True,
                  all74_state_and_actual_targets_public=True,
                  object_normal_and_friction_points_public=True,
                  all_native_frame_timestamps_public=True, all_source_PNG_frames_public=False,
                  final_display_hold_is_not_additional_physics=True, media=media,
                  system0_active=False, actor_active=False, formal_holdout_count=0,
                  task_pass=audit['four_hand_task_development_pass'],
                  model_pass=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')['model_window_pass'],
                  original_physical_gates_pass=audit['physical_gates_pass'],
                  fullSystem0_accepted=False, sources=sources)
    (dest / 'BUNDLE_RECEIPT_V2.json').write_text(json.dumps(bundle, indent=2) + '\n')
    return bundle
