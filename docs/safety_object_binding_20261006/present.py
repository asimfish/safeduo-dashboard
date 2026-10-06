"""Publish only fully audited native development evidence."""
import argparse,json,hashlib,shutil,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    out=a.repo/'docs'/R.name;out.mkdir(parents=True,exist_ok=True)
    result=json.loads((R/'native_results.json').read_text());assert result['status']=='PASS_NATIVE_TRACE_AUDIT'
    native=Path(result['native_source_directory']);old=json.loads((R/'retrospective.json').read_text())
    data=dict(result=result,retrospective={k:v for k,v in old.items() if k not in ('examples','cases','sources')},goals=[[.37,0,.84],[-.5,0,.84]],phase=json.loads((R/'PHASE_DIAGNOSIS.json').read_text()),keyframe_video=json.loads((R/'KEYFRAME_VIDEO.json').read_text()),media_archive=json.loads((R/'MEDIA_ARCHIVE.json').read_text()),curves=[])
    with np.load(R/'native_curves.npz',allow_pickle=False) as z:
        data['dt']=float(z['time'][1]-z['time'][0])
        for e in range(8):
            data['curves'].append(dict(time=z['time'].tolist(),xyz=z['states'][:,e,:,:3].tolist(),lift=z['lift'][:,e].tolist(),tilt=z['tilt'][:,e].tolist(),clearance=z['bbox_clearance'][:,e].tolist(),hand=z['hand_normal'][:,e].tolist(),table=z['own_table_normal'][:,e].tolist()))
    fixed=result['totals']['fixed_replay'];bound=result['totals']['object_pose_binding']
    data['summary']=f"固定轨迹任务通过 {fixed['passed']}/4，物体位姿绑定通过 {bound['passed']}/4；几何违规窗口分别 {fixed['violating']}/4 与 {bound['violating']}/4。6/8任务失败；两条原门槛通过轨迹的U抬升均只发生在退离阶段，且没有抬升中的双手接触。物体位姿绑定没有提高本轮任务通过数，未批准新策略。"
    (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    (out/'retrospective_summary.json').write_text(json.dumps({k:v for k,v in old.items() if k!='examples'},indent=2)+'\n')
    shutil.copyfile(R/'panel.html',out/'index.html')
    video=data['keyframe_video'];assert sha(video['file'])==video['sha256'];shutil.copyfile(video['file'],out/'keyframe_review.mp4')
    webm=json.loads((R/'WEBM_COMPATIBILITY.json').read_text());assert sha(webm['file'])==webm['sha256'];shutil.copyfile(webm['file'],out/'keyframe_review.webm')
    for p in R.iterdir():
        if p.suffix in ('.py','.md','.json','.csv') and p.name not in ('retrospective.json','delivery.json','PUBLIC_MANIFEST.json'):
            shutil.copyfile(p,out/p.name)
    for name in ('native_initial.json','recording_receipt.json','bound_reference.npz','native_limits_before_planning.json','effective_registration.json'):shutil.copyfile(native/name,out/name)
    shutil.copyfile(R/'task_qualified_two_pair.npz',out/'task_qualified_two_pair.npz')
    # Full originals/raw bytes live in the immutable archive commit outside Pages.
    archive=data['media_archive'];assert archive['status']=='PASS_ALL_REMOTE_MEDIA_SHA'
    for item in result['native_images']:
        assert sha(native/item['file'])==item['sha256']==archive['files'][item['file']]
    for item in json.loads((native/'recording_receipt.json').read_text())['chunks']:
        assert sha(native/item['file'])==item['sha256']==archive['files']['raw/'+item['file']]
    log_manifest=[]
    for p in R.glob('*.log'):
        raw=p.read_bytes();name=p.name+'.gz';(out/name).write_bytes(gzip.compress(raw,mtime=0));log_manifest.append(dict(file=name,raw_sha256=hashlib.sha256(raw).hexdigest()))
    (out/'LOG_MANIFEST.json').write_text(json.dumps(log_manifest,indent=2)+'\n')
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,(o,v) in zip(axes,old['totals'].items()):
        counts=[v['lifted_50mm_steps'],v['lifted_50mm_both_assigned_hands_gt_0_1n_steps']]
        ax.bar(['All lifted states','Lifted + both hands'],counts,color=['#3d92cc','#26a78e']);ax.set_title(o+' — known development data');ax.set_ylabel('Sampled states (not trials)');ax.set_ylim(0,max(counts)*1.2)
        for i,c in enumerate(counts):ax.text(i,c,str(c),ha='center',va='bottom')
    for suffix in ('png','pdf','svg'):fig.savefig(out/('development_contact.'+suffix),dpi=180)
    plt.close(fig)
    svg=out/'development_contact.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    root=a.repo/'index.html';html=root.read_text();id='objectBindingEvidence';assert id not in html
    card=f'<section class="sci-note" id="{id}"><h2>真实四臂物体任务：输入绑定与夹持诊断</h2><p>新增8个原生物理开发对照，{len(result["native_images"])}张同次执行俯视/正视/侧视实图。固定轨迹通过{fixed["passed"]}/4，物体位姿绑定通过{bound["passed"]}/4；两方案共6/8失败，原门槛通过也未证明稳定双手抓持；未通过全面安全准入。</p><p>旧128个开发样本逐步复核：U物体抬升时双手接触约33%，F约96%。原19/192任务结果及FAIL_UNCALIBRATED保留；没有新增正式留出或广域覆盖宣称。</p><p><a href="docs/{R.name}/">查看全部8案例、接触曲线、原图与诊断证据 →</a></p></section>\n'
    marker='<section class="sci-note" id="benchmarkProtocolEvidence">';assert html.count(marker)==1;root.write_text(html.replace(marker,card+marker,1))
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PUBLIC_MANIFEST.json'}
    (out/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(files=manifest,scope='complete development diagnostic, not full safety acceptance'),indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),images=len(result['native_images']),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),summary=data['summary']),ensure_ascii=False))
if __name__=='__main__':main()
