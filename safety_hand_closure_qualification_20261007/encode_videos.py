from pathlib import Path
import json,hashlib,subprocess
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/validation');receipt=json.loads((raw/'recording_receipt.json').read_text());videos=[];out=p/'videos';out.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
items=[('validation',0,view,'validation_'+view) for view in ['top','front','side','detail']]+[('paired',0,'detail','unsafe_bypass_detail'),('paired',1,'detail','guard_rejected_detail')]
for run,env_id,view,label in items:
 raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007')/run
 receipt=json.loads((raw/'recording_receipt.json').read_text())
 imgs=sorted([x for x in receipt['images'] if x['env']==env_id and x['view']==view and x['step']<720 and x['step']%12==0],key=lambda x:x['step']);assert len(imgs)==60 and [x['step'] for x in imgs]==list(range(0,720,12))
 frames=out/label;frames.mkdir();lines=[]
 for i,img in enumerate(imgs):
  src=raw/img['file'];assert sha(src)==img['sha256'];(frames/f'{i:04d}.png').symlink_to(src)
 mp4=out/f'{label}.mp4';webm=out/f'{label}.webm'
 base=['ffmpeg','-y','-nostdin','-loglevel','error','-framerate','10','-i',str(frames/'%04d.png')]
 subprocess.run(base+['-c:v','libx264','-crf','21','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
 subprocess.run(base+['-c:v','libvpx-vp9','-crf','32','-b:v','0','-row-mt','1','-threads','2',str(webm)],check=True)
 for dest in [mp4,webm]:
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,codec_name,nb_frames:format=duration','-of','json',str(dest)]));assert probe['streams'][0]['width']==1280 and probe['streams'][0]['height']==720 and abs(float(probe['format']['duration'])-6)<.02
 videos.append(dict(view=label,run=run,env=env_id,mp4='videos/'+mp4.name,webm='videos/'+webm.name,poster=run+'/'+imgs[0]['file'],frames=60,fps=10,mp4_sha256=sha(mp4),webm_sha256=sha(webm),original_frames=imgs))
(p/'VIDEO_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ENCODINGS_FROM_UNEDITED_NATIVE_FRAMES',videos=videos),indent=2)+'\n');print('VIDEO_PASS',len(videos))
