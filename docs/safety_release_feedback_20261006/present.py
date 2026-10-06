import argparse,json,hashlib,shutil,gzip,csv
from pathlib import Path
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
dest=a.repo/'docs/safety_release_feedback_20261006';dest.mkdir(exist_ok=False)
results=json.loads((R/'RESULTS.json').read_text());archive=json.loads((R/'MEDIA_ARCHIVE.json').read_text());video=json.loads((R/'VIDEO_RECEIPT.json').read_text())
ts=results['totals'];names={'scheduled_debt':'原计划','stationary_release':'静止释放','clearance_feedback':'退离反馈'};summary='；'.join(names[m]+' 原门槛'+str(ts[m]['pass_original_task'])+'/8，合格完成'+str(ts[m]['accepted_task'])+'/8，中止'+str(ts[m]['aborted'])+'/8' for m in ts)+'。原安全球指标不等同物体夹持安全；候选尚未通过全面准入。'
results['summary']=summary
# Explicit presentation-only extra field; original scorer RESULTS bytes remain frozen.
curves={}
with (R/'timeseries.csv').open() as f:
    for index,row in enumerate(csv.DictReader(f)):
        key=row['block']+'_'+row['env'];d=curves.setdefault(key,{k:[] for k in ('time_s','phase_s','beam700_lift_m','beam700_tilt_deg','beam700_hand_n','beam700_table_n','beam300_lift_m','beam300_tilt_deg','beam300_hand_n','beam300_table_n')})
        # 1800 steps per case, keep every6 and terminal.
        step=index%1800
        if step%6==0 or step==1799:
            for k in d:d[k].append(round(float(row[k]),6))
images=[]
reg=json.loads((R/'REGISTRATION.json').read_text())
for b in range(2):
    receipt=json.loads((RAW/f'block_{b}/recording_receipt.json').read_text())
    for item in receipt['images']:
        if item['step'] in reg['capture_steps']:
            images.append({**item,'block':b,'url':archive['raw_base_url']+'safety_release_feedback_20261006/'+f'block_{b}/'+item['file']})
assert len(images)==360
payload=dict(results=results,curves=curves,images=images,videos=[{k:v[k] for k in ('env','method','mp4','webm')} for v in video['videos']],archive={k:archive[k] for k in ('raw_base_url','archive_zip_url','commit')})
(dest/'data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
shutil.copyfile(R/'panel.html',dest/'index.html')
public=['RESULTS.json','REGISTRATION.json','DIAGNOSIS.json','HYPOTHESIS_UPDATE.json','POSTRUN_DIAGNOSIS.json','postrun_diagnosis.py','GUARD_INPUT_AUDIT.json','diagnose_gate_inputs.py','HAND_ASSET_LIMITS.json','SHARED_SOURCE_BEFORE.json','ASSET_BEFORE.json','SOURCE_AFTER.json','VIDEO_RECEIPT.json','MEDIA_ARCHIVE.json','clearance_gate.py','native_runner.py','analyze.py','test_gate.py','build_runner.py','diagnose.py','register.py','build_video.py','archive_media.py','verify_archive.py','REPRODUCE.md','TEST_CONTRACT.md','gate_tests.log','gate_tests_v2.log','diagnose.log','diagnose_execution.log','registration.log','TERMINAL_CONTRACT.json']
for rel in public:shutil.copyfile(R/rel,dest/rel)
for v in video['videos']:
    for k in ('mp4','webm'):shutil.copyfile(R/v[k],dest/v[k])
logs=[]
for b in range(2):
    path=R/f'native_block{b}.log';rel=path.name+'.gz'
    (dest/rel).write_bytes(gzip.compress(path.read_bytes(),mtime=0));logs.append(dict(file=rel,original_sha256=sha(path),gzip_sha256=sha(dest/rel)))
(dest/'LOG_MANIFEST.json').write_text(json.dumps(logs,indent=2)+'\n')
for b in range(2):
    for name in ('recording_receipt.json','native_initial.json'):
        shutil.copyfile(RAW/f'block_{b}'/name,dest/(f'block{b}_'+name))
root=a.repo/'index.html';s=root.read_text();assert 'id="releaseFeedbackEvidence"' not in s
card='<section class="sci-note" id="releaseFeedbackEvidence"><h2>最新物体任务：静止释放与退离反馈</h2><p>24条完整开发任务，4布局×三条件×两种子并轮换槽位。'+summary+'</p><p>新增实际连续录像3段，全部24案例的360张三视角原图、完整接触及物体曲线可查。原门槛不变，中止与实际运动量单独列出；四臂各成对搬两件物体，尚未测试四臂共持或扩大随机范围。</p><p><a href="docs/safety_release_feedback_20261006/">查看三组实际录像、全部任务与安全失败证据 →</a></p></section>\n'
needle='<section class="sci-note" id="objectBindingEvidence">';assert s.count(needle)==1;root.write_text(s.replace(needle,card+needle))
manifest={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
(dest/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(files=manifest,scope='Current small Pages package; all heavy original media and dense data are fixed-commit archive references'),indent=2)+'\n')
print(json.dumps(dict(bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),files=len(manifest)+1,images=len(images),summary=summary),ensure_ascii=False,indent=2))
