from pathlib import Path
import json,subprocess,hashlib
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');out=p/'videos';out.mkdir(exist_ok=False);sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();videos=[]
for run,env,label,kind in [('matched_scene',0,'旧V3 · 原生近景 · 部分遮挡','direct_native_capture'),('matched_scene',1,'几何准入 · 原生近景 · 部分遮挡','direct_native_capture'),('camera_replay',0,'旧V3 · 存档状态俯视重渲染','archived_state_render_only'),('camera_replay',1,'几何准入 · 存档状态俯视重渲染','archived_state_render_only'),('fresh_scene_batch',0,'新姿态 · 左手原生俯视 · 左手拒绝、右手部分入镜','direct_native_capture'),('camera_replay_ur',0,'新姿态 · 准入右手闭合 · 存档状态俯视重渲染','archived_state_render_only')]:
    receipt=json.loads((raw/run/'recording_receipt.json').read_text())
    if run.startswith('camera_replay'):frames=[dict(file=x['file'],sha256=x['sha256'],step=x['source_step'],time_s=x['source_native_state_time_s']) for x in receipt['images'] if x['source_env']==env]
    else:frames=[dict(file=x['file'],sha256=x['sha256'],step=x['step'],time_s=x['state_time_s']) for x in receipt['images'] if x['env']==env and x['view']=='detail' and x['step']<720 and x['step']%12==0]
    frames.sort(key=lambda x:x['step']);assert [x['step'] for x in frames]==list(range(0,720,12))
    for x in frames:assert sha(raw/run/x['file'])==x['sha256']
    tag=f'{run}_e{env}';lst=out/(tag+'.txt');lst.write_text(''.join("file '"+str(raw/run/x['file'])+"'\nduration 0.1\n" for x in frames))
    infos={}
    for ext,codec in [('mp4',['-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart']),('webm',['-c:v','libvpx-vp9','-crf','30','-b:v','0','-row-mt','1'])]:
        dest=out/(tag+'.'+ext);subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(lst),'-frames:v','60','-r','10','-threads','2',*codec,str(dest)],check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate,duration','-of','json',str(dest)]))['streams'][0];assert int(info['nb_read_frames'])==60 and info['width']==1280 and info['height']==720;infos[ext]=info
    videos.append(dict(run=run,env=env,label=label,visualization=kind,source_receipt_sha256=sha(raw/run/'recording_receipt.json'),source_frames=frames,mp4='videos/'+tag+'.mp4',webm='videos/'+tag+'.webm',mp4_sha256=sha(out/(tag+'.mp4')),webm_sha256=sha(out/(tag+'.webm')),poster=run+'/'+frames[0]['file'],decode=infos));print('ENCODED',tag,flush=True)
(p/'VIDEO_RECEIPT.json').write_text(json.dumps(dict(status='PASS_3_NATIVE_AND_3_LABELLED_RECONSTRUCTED_VIDEOS',native_clips=3,supplementary_render_clips=3,retouched=False,videos=videos),ensure_ascii=False,indent=2)+'\n')
