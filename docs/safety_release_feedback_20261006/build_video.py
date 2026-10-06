"""Encode actual native samples, preserving original images and all source IDs."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006/block_0')
rec=json.loads((RAW/'recording_receipt.json').read_text());init=json.loads((RAW/'native_initial.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
work=RAW.parent/'video_encoding';work.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=[]
for c in init['cases']:
    if c['layout']!=0:continue
    frames=sorted([i for i in rec['video_frames'] if i['env']==c['env']],key=lambda i:i['step']);assert len(frames)==61
    assert np.allclose(np.diff([i['state_time_s'] for i in frames]),.099996,atol=1e-9,rtol=0)
    directory=work/c['method'];directory.mkdir(exist_ok=True)
    for k,item in enumerate(frames):
        p=RAW/item['file'];assert sha(p)==item['sha256'];src=Image.open(p).convert('RGB');assert src.size==(1280,720)
        canvas=Image.new('RGB',(1280,800),(13,20,34));canvas.paste(src,(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,727),f"{c['method']} | block0 env{c['env']} | native t={item['state_time_s']:.6f}s | phase={item['phase_s']:.6f}s | stage={item['stage']}",font=font,fill='white')
        draw.text((12,756),'Actual camera frames every 6 physics-control steps (~10Hz); no motion interpolation; originals preserved.',font=font,fill=(179,198,217))
        canvas.save(directory/f'{k:04d}.png')
    codecs=[('mp4',['-c:v','libx264','-crf','28','-preset','fast','-movflags','+faststart']),('webm',['-c:v','libvpx-vp9','-crf','35','-b:v','0','-row-mt','1'])]
    versions=[]
    for ext,codec in codecs:
        path=R/(c['method']+'.'+ext)
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','10','-i',str(directory/'%04d.png'),*codec,'-pix_fmt','yuv420p',str(path)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        assert abs(float(probe['format']['duration'])-6.1)<.05
        decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
        image_array=np.frombuffer(decoded,dtype=np.uint8).reshape(-1,800,1280,3);assert len(image_array)==61
        rms=[]
        for k,item in enumerate(frames):
            source=np.asarray(Image.open(RAW/item['file']).convert('RGB'),dtype=np.float32)
            rmse=float(np.sqrt(np.mean((image_array[k,:720].astype(np.float32)-source)**2)));assert rmse<12,(ext,k,rmse);rms.append(rmse)
        versions.append(dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,all61_source_frames_verified=True,max_source_rmse=max(rms),duration_s=float(probe['format']['duration']),codec=probe['streams'][0]['codec_name']))
    out.append(dict(method=c['method'],env=c['env'],mp4=c['method']+'.mp4',webm=c['method']+'.webm',frames=frames,versions=versions))
receipt=dict(status='PASS_ALL_183_NATIVE_SOURCE_FRAMES_BOTH_CODECS',videos=out,scope='Actual source cadence6 controlsteps .099996s; encoding10fps, last frame held .1s. No interpolation; raw original camera pixels above annotation band retained.',source_receipt_sha256=sha(RAW/'recording_receipt.json'))
(R/'VIDEO_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],versions=[v['versions'] for v in out]),indent=2))
