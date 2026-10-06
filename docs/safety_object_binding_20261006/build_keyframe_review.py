"""All recorded stills in a labeled review; deliberately not continuous footage."""
import json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((R/'native_results.json').read_text());native=Path(result['native_source_directory'])
    out=native.parent/'keyframe_review';out.mkdir(exist_ok=True)
    groups=sorted({(i['env'],i['step']) for i in result['native_images']})
    bykey={(i['env'],i['step'],i['view']):i for i in result['native_images']}
    frames=[];inputs=[];hold=2
    for view in ('top','front','side'):
        entries=[]
        for e,s in groups:
            item=bykey[e,s,view];path=native/item['file'];assert sha(path)==item['sha256']
            assert "'" not in str(path);entries.extend([f"file '{path}'",f'duration {hold}'])
        entries.append(f"file '{path}'")
        playlist=out/(view+'.txt');playlist.write_text('\n'.join(entries)+'\n')
        inputs.extend(['-f','concat','-safe','0','-i',str(playlist)])
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters=[f'[{i}:v]scale=640:360,setsar=1[v{i}]' for i in range(3)]
    filters.append('[v0][v1][v2]hstack=inputs=3,pad=1920:450:0:50:color=0x0b1425,drawtext=fontfile='+font+":text='RECORDED KEYFRAMES - DISCRETE STATES - NOT CONTINUOUS MOTION':fontsize=26:fontcolor=white:x=20:y=10[base]")
    prev='base'
    for i,(e,s) in enumerate(groups):
        item=bykey[e,s,'top'];mode='pose binding' if result['cases'][e]['binding'] else 'fixed replay'
        label=f"env {e} | {mode} | step {s} | actual state {item['state_time_s']:.6f}s | Top / Front / Side"
        nxt=f'caption{i}'
        filters.append(f"[{prev}]drawtext=fontfile={font}:text='{label}':fontsize=24:fontcolor=white:x=20:y=415:enable='gte(t,{i*hold})*lt(t,{(i+1)*hold})'[{nxt}]")
        prev=nxt;frames.append(dict(env=e,step=s,original_state_time_s=item['state_time_s'],video_start_s=i*hold,video_end_s=(i+1)*hold,source_images=[bykey[e,s,v]['file'] for v in ('top','front','side')]))
    graph=out/'filters.txt';graph.write_text(';\n'.join(filters)+'\n')
    video=out/'keyframe_review.mp4';duration=len(groups)*hold
    cmd=['ffmpeg','-hide_banner','-loglevel','warning','-y','-threads','1',*inputs,'-filter_complex_threads','1','-filter_complex_script',str(graph),'-map',f'[{prev}]','-t',str(duration),'-r','25','-c:v','libx264','-threads','1','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    subprocess.run(cmd,check=True)
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)
    assert abs(float(probe['format']['duration'])-duration)<.05
    subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
    receipt=dict(status='PASS_COMPLETE_DISCRETE_KEYFRAME_REVIEW',file=str(video),sha256=sha(video),duration_s=duration,groups=len(groups),source_image_count=len(result['native_images']),all_cases=list(range(8)),frames=frames,scope='all132 native stills resized into44 tri-view keyframes, held2s each; no interpolation, no continuous motion claim, no replacement for the original states/images',ffmpeg_version=subprocess.run(['ffmpeg','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0])
    (R/'KEYFRAME_VIDEO.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='frames'}))
if __name__=='__main__':main()
