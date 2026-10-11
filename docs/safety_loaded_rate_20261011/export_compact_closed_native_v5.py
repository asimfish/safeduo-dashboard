"""Export every captured free-predictor step without fabricated observer/query fields."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

H=Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(version,root):
    p=H/f'fourhand_native_probe_v{version}'
    actual=read(p/f'ACTUAL_COMPLETION_EXIT_V{version}.json')
    assert actual['actual_outer_exit']==actual['actual_launcher_subprocess_exit']==actual['actual_native_exit']==actual['actual_finite_exit']==0
    a=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    assert a['record_integrity_verified'] and a['all_owned_native_children_reaped']
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out'])
    dest=Path(root)/f'native_v{version}';dest.mkdir(exist_ok=False)
    sources=[]
    def copy(f,name):
        f=Path(f);t=dest/name;t.parent.mkdir(parents=True,exist_ok=True)
        assert not t.exists();shutil.copyfile(f,t);assert sha(t)==sha(f)
        sources.append(dict(path=name,source=str(f),sha256=sha(f)))
    for f in sorted(p.glob('*.json')):copy(f,f.name)
    for f in sorted(p.glob('*.py')):copy(f,'source/'+f.name)
    copy(p/'INDEPENDENT_TASK_METRICS_V1.npz','INDEPENDENT_TASK_METRICS_V1.npz')
    for name in ['native_parameters.npz','fourhand_protocol.json','physics_chunks.json','camera_frames.json',
        'native_mimic_identity.json','native_scene_settings.json','native_contact_identity.json']:
        # Same-named local protocol files are absent; custody copies are never silently overwritten.
        if not (dest/name).exists():copy(raw/name,name)
    selected=['physics_event','state_time_s','phase','object_state','object_size_m','four_hand_normal_N',
        'support_normal_N','feedback_grip_mode','feedback_grip_finger_fraction','feedback_grip_thumb_fraction',
        'pregrasp_position_error_rad','grasp_position_error_rad','grasp_arm_speed_rad_s','grasp_settle_dwell_s',
        'prelift_ready','prelift_native_hand_min_margin_rad','prelift_max_hand_speed_rad_s']
    for arm in ['F_L','F_R','U_L','U_R']:
        selected.extend(arm+suffix for suffix in ['_native_q','_native_qd','_actual_full_position_target','_native_external_actuation_effort','_hand_nominal_position_target','_hand_filtered_position_target','_software_full_motor_PD_preclip'])
    pieces={key:[] for key in selected};chunks=[]
    for row in read(raw/'physics_chunks.json'):
        f=raw/row['path'];assert sha(f)==row['sha256']
        chunks.append(dict(source=str(f),sha256=row['sha256'],start=row['start'],stop=row['stop']))
        with np.load(f,allow_pickle=False) as d:
            for key in selected:pieces[key].append(d[key].copy())
    arrays={key:np.concatenate(value,axis=0) for key,value in pieces.items()}
    assert len(arrays['physics_event'])==a['physics_events']
    np.testing.assert_array_equal(arrays['physics_event'].reshape(-1),np.arange(a['physics_events']))
    target=dest/'SELECTED_EVERY_STEP_NATIVE_FIELDS_V4.npz'
    np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as d:
        for key,value in arrays.items():np.testing.assert_array_equal(d[key],value)
    media=[]
    for folder,ext in [('video_evidence_v5','mp4'),('video_webm_v3','webm')]:
        assert read(p/folder/'ENCODER_ACTUAL_EXIT_V1.json')['actual_outer_exit']==0
        custody=read(p/folder/'VIDEO_CUSTODY_V1.json')
        assert custody['native_probe_version']==version and len(custody['views'])==4
        for row in custody['views']:
            assert sha(row['path'])==row['sha256']
            copy(row['path'],row['view']+'.'+ext)
            media.append(dict(view=row['view'],format=ext,frames=row['source_frames'],max_timestamp_error_s=row['max_timestamp_error_s']))
        copy(p/folder/'VIDEO_CUSTODY_V1.json',ext+'_VIDEO_CUSTODY_V1.json')
        copy(p/folder/'ENCODER_ACTUAL_EXIT_V1.json',ext+'_ENCODER_ACTUAL_EXIT_V1.json')
    frames=read(raw/'camera_frames.json')
    for view in ['top','front','side','overview']:
        last=[f for f in frames if f['view']==view][-1]
        frame=raw/last['path']
        assert sha(frame)==last['sha256'];copy(frame,view+'_last.png')
    result=dict(version=version,physics_events=a['physics_events'],controls=a['completed_controls'],
        status='CLOSED_CAPTURE_WITH_EVERY_STEP_SELECTED_NATIVE_FIELDS_AND_METRICS_PUBLIC',
        selected_native_fields_public=selected,every_recorded_step_included=True,
        selected_arrays_unchanged_from_registered_raw_chunks=True,
        full_raw_mass_body_pose_geometry_and_point_contact_arrays_public=False,
        full_original_raw_archive_location=str(raw),full_raw_source_chunks=chunks,
        selected_npz_sha256=sha(target),sources=sources,media=media,
        physical_pass=a['physical_gates_pass'],full_task_pass=a['four_hand_task_development_pass'],
        actual_outer_exit=actual['actual_outer_exit'],outer_tool_receipt_unknown=False,diagnostic_capture_not_fulltask=True,candidate_adopted=False,actor=False,system0=False,unique_known_input_cases=1,formal_holdout_count=0,fullSystem0_accepted=False)
    (dest/'COMPACT_PUBLIC_CUSTODY_V5.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
