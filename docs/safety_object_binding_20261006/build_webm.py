"""Codec fallback for the same labeled discrete review, not new observations."""
import json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
    receipt=json.loads((R/'KEYFRAME_VIDEO.json').read_text());src=Path(receipt['file']);dest=src.with_suffix('.webm')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-threads','1','-i',str(src),'-an','-c:v','libvpx-vp9','-threads','4','-row-mt','1','-cpu-used','4','-crf','32','-b:v','0','-pix_fmt','yuv420p',str(dest)],check=True)
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)],capture_output=True,text=True,check=True).stdout)
    assert probe['streams'][0]['codec_name']=='vp9' and abs(float(probe['format']['duration'])-88)<.05
    v=dict(status='COMPLETE_VP9_CODEC_FALLBACK',file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_mp4_sha256=receipt['sha256'],duration_s=88,width=1920,height=450,scope='WebM codec fallback for exact same discrete review; originalMP4 and nativePNGs retained; not new physical observations')
    (R/'WEBM_COMPATIBILITY.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
