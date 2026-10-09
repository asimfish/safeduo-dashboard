"""Encode and fully decode native camera samples; each view binds to its own run."""
import hashlib
import json
from pathlib import Path
import subprocess

H=Path(__file__).resolve().parent
REG=json.loads((H/'PAIRED_REGISTRATION_V1.json').read_text())
R=Path(REG['raw'])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(str(args)+': '+p.stderr[-2000:])
    return p


def main():
    manifest=json.loads((H/'PAIRED128_RESULT_V1.json').read_text())
    assert manifest['all4_native_actual_exit0'] and manifest['all4_independent_oracles_pass']
    output=R/'encoded_v1'
    output.mkdir(exist_ok=False)
    videos=[]
    for batch in range(2):
        for method in ['raw','multirow']:
            root=R/f'paired_batch{batch}_{method}_v8'
            cameras=json.loads((root/'pilot_camera_receipts.json').read_text())
            assert cameras['status']=='complete' and cameras['frames']==31
            assert cameras['images']==31*16
            with __import__('numpy').load(root/'resolved_native_parameters.npz') as z:
                inputs=z['global_input_id']
            for receipt in cameras['receipts']:
                assert receipt['render_did_not_advance_physics']
                assert sha(root/receipt['native_state_path'])==receipt['native_state_sha256']
                for im in receipt['images']:
                    assert im['all_selected_spheres_contained']
                    assert im['native_fabric_all_body_position_max_error_m']<=2e-5
                    assert im['native_fabric_all_body_orientation_max_error']<=2e-5
                    assert sha(root/im['path'])==im['sha256']
            for lane in REG['camera']['slots']:
                for view in REG['camera']['views']:
                    prefix=f'batch{batch}_input{int(inputs[lane]):03d}_{method}_{view}'
                    source=root/'native_views'/f'env_{lane:03d}'/view
                    assert len(list(source.glob('frame_*.png')))==31
                    formats={}
                    for ext in ['webm','mp4']:
                        target=output/f'{prefix}.{ext}'
                        cmd=['ffmpeg','-v','error','-nostdin','-framerate','3.75','-i',str(source/'frame_%04d.png'),'-vf','scale=960:540','-an','-threads','2']
                        if ext=='webm':
                            cmd+=['-c:v','libvpx-vp9','-row-mt','1','-cpu-used','5','-crf','32','-b:v','0']
                        else:
                            cmd+=['-c:v','libx264','-preset','veryfast','-crf','25','-movflags','+faststart']
                        cmd+=['-pix_fmt','yuv420p',str(target)]
                        encoded=run(cmd)
                        decoded=run(['ffmpeg','-v','error','-nostdin','-i',str(target),'-f','null','-'])
                        probe=json.loads(run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,codec_name','-of','json',str(target)]).stdout)['streams'][0]
                        assert int(probe['nb_read_frames'])==31 and probe['width']==960 and probe['height']==540
                        formats[ext]=dict(path=str(target),sha256=sha(target),encoder_actual_exit=encoded.returncode,full_decoder_actual_exit=decoded.returncode,probe=probe)
                    videos.append(dict(batch=batch,method=method,lane=lane,input_id=int(inputs[lane]),view=view,
                        native_camera_receipt_sha256=sha(root/'pilot_camera_receipts.json'),frames=31,
                        native_macro_boundaries=[x['step']+1 for x in cameras['receipts']],
                        playback_fps=3.75,native_first_frame_control=1,native_last_frame_control=480,
                        note='Firstinterval15controls; remaining16controls. Encoded CFR frame clock is a sample-index clock, not exact native event time. Force and geometry scored every native microstep.',
                        poster=str(source/'frame_0015.png'),formats=formats))
                    print('ENCODED',prefix,flush=True)
    assert len(videos)==64
    result=dict(status='PASS_64_NATIVE_VIEWS_128_FORMATS_FULL_DECODE',logical_clips=64,encoded_files=128,
        selected_before_outcomes=True,independent_paired_inputs_shown=8,frames_per_clip=31,width=960,height=540,
        actual_same_run=True,native_all_body_Fabric_bound=True,render_advances_physics=False,
        force_peak_frame_exact_claim=False,videos=videos)
    (H/'PAIRED_VIDEO_RECEIPT_V1.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
