"""Check every review group's source identity after lossy video encoding."""
from pathlib import Path
import json,subprocess,hashlib
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent

def main():
    receipt=json.loads((R/'KEYFRAME_VIDEO.json').read_text());video=Path(receipt['file'])
    assert hashlib.sha256(video.read_bytes()).hexdigest()==receipt['sha256']
    result=json.loads((R/'native_results.json').read_text());native=Path(result['native_source_directory'])
    raw=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-vf',"select='eq(mod(n,50),25)'",'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout
    data=np.frombuffer(raw,np.uint8).reshape(-1,450,1920,3);assert len(data)==receipt['groups']==44
    rows=[]
    for i,group in enumerate(receipt['frames']):
        for j,source in enumerate(group['source_images']):
            with Image.open(native/source) as image:expected=np.asarray(image.convert('RGB').resize((640,360),Image.Resampling.BICUBIC),dtype=np.float32)
            actual=data[i,50:410,j*640:(j+1)*640].astype(np.float32);diff=actual-expected
            mae=float(np.abs(diff).mean());rmse=float(np.sqrt((diff**2).mean()))
            assert mae<8 and rmse<12,(i,source,mae,rmse)
            rows.append(dict(group=i,source=source,mean_absolute_rgb_error=mae,root_mean_square_rgb_error=rmse))
    verdict=dict(status='PASS_ALL_44_GROUPS_132_IMAGE_IDENTITIES',max_mae=max(x['mean_absolute_rgb_error'] for x in rows),max_rmse=max(x['root_mean_square_rgb_error'] for x in rows),scope='center encoded frame of each discrete2s group compared with original native source image resized bicubically; lossy review fidelity only, no claim of continuous motion or altered original images',checks=rows)
    (R/'VIDEO_VERIFICATION.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps({k:v for k,v in verdict.items() if k!='checks'}))
if __name__=='__main__':main()
