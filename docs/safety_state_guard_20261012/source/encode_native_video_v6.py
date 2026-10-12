"""Encode every registered native frame with exact variable native timestamps."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import sys
sys.path.insert(0,str(Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')))
from microstep_contract_v1 import microsteps
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(version):
    p=H/f'fourhand_native_probe_v{version}'
    audit=json.loads((p/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    resource=json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text());assert resource['actual_native_exit']==resource['actual_exit']==0
    root=Path(resource['out']);frames=json.loads((root/'camera_frames.json').read_text());out=p/'video_evidence_v6';out.mkdir(exist_ok=False)
    reg=json.loads((p/'REGISTRATION_V1.json').read_text());clock=reg['physics_dt_s'];substeps=microsteps(reg);assert reg['control_dt_s']==1/60.
    receipt=dict(status='running',native_probe_version=version,views=[],continuous_capture_of_observed_window=True,
                 full_maximum_window_complete=audit['maximum_window_complete'],task_status=audit['task_status'],task_pass=audit['four_hand_task_development_pass'],system0_active=False,physics_substeps_per_control=substeps,physics_dt_s=clock,source_encoder_sha256=sha(__file__),fullSystem0_accepted=False)
    try:
        for view in reg['camera_views']:
            group=[f for f in frames if f['view']==view];times=np.array([f['state_time_s'] for f in group]);assert len(group)>1
            assert all(sha(root/f['path'])==f['sha256'] for f in group)
            sequence=out/(view+'_source_sequence');sequence.mkdir(exist_ok=False)
            for i,f in enumerate(group):(sequence/f'frame_{i:06d}.png').symlink_to(root/f['path'])
            expected=np.array([f['control'] for f in group])*reg['control_dt_s']
            assert np.max(abs(times-expected))<=1e-12
            assert [f['control'] for f in group[:-1]]==list(range(0,4*(len(group)-1),4))
            last_us=round(times[-1]*1000000);stride_us=4*reg['control_dt_s']*1000000
            pts_filter=f"settb=expr=1/1000000,setpts='min(round(N*{stride_us}),{last_us})'"
            video=out/(view+'.mp4')
            argv=['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-framerate','125000/8333',
                  '-i',str(sequence/'frame_%06d.png'),'-vf',pts_filter,
                  '-an','-c:v','libx264','-threads','2','-preset','medium','-crf','20','-pix_fmt','yuv420p',
                  '-vsync','vfr','-enc_time_base','1:1000000','-video_track_timescale','1000000','-movflags','+faststart',str(video)]
            with (out/(view+'_encoder.log')).open('xb') as log:
                result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            assert result.returncode==0,'native PNG encoding failed'
            result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)],capture_output=True,text=True,timeout=30)
            assert result.returncode==0
            observed=np.array([float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']])
            assert observed.shape==times.shape,'missing/duplicated output frames'
            error=float(abs(observed-times).max());assert error<=1.1e-6,'encoded clock differs from native frame time'
            metadata=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,timeout=30);assert metadata.returncode==0
            meta=json.loads(metadata.stdout);stream=meta['streams'][0];assert stream['width']==1280 and stream['height']==720
            receipt['views'].append(dict(view=view,path=str(video),sha256=sha(video),bytes=video.stat().st_size,
                source_frames=len(group),native_frame_controls=[f['control'] for f in group],native_frame_times_s=times.tolist(),
                actual_decoded_frame_times_s=observed.tolist(),max_timestamp_error_s=error,actual_encoder_exit=0,actual_probe_exits=[0,0],
                duration_s=float(meta['format']['duration']),source_last_time_s=float(times[-1]),native_camera_settings=reg['camera_views'][view]))
            print('NATIVE_VIDEO_ENCODED',view,len(group),'frames','timestamp_error',error,flush=True)
        receipt['status']='PASS_ALL4_NATIVE_FRAMES_AND_EXACT_VIDEO_TIMESTAMPS'
    except BaseException as e:receipt.update(status='failed',error=repr(e));raise
    finally:(out/'VIDEO_CUSTODY_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);main(parser.parse_args().version)
