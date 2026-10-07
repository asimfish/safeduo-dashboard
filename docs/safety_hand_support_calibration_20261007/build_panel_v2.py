import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive-commit',required=True);a=p.parse_args()
RAW=Path('/mnt/nas/data/lyf/double_hand/safety_hand_support_calibration_20261007')
dest=a.repo/'docs/safety_hand_support_calibration_20261007';dest.mkdir(exist_ok=True)
reg=json.loads((R/'REGISTRATION_V2.json').read_text());cases=reg['cases']
labels=['初始姿态（物体移开）','取物姿态（物体移开）','放置姿态（物体移开）','退离姿态（物体移开）','起点静态支撑','终点静态支撑']
for c,label in zip(cases,labels):c['label_cn']=label
prefix='safety_hand_support_calibration_20261007/'
base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+a.archive_commit+'/'+prefix
archive=json.loads((R/'MEDIA_ARCHIVE.json').read_text())
assert archive['commit']==a.archive_commit
series={e:{arm:{} for arm in ('F_L','F_R','U_L','U_R')} for e in range(len(cases))};images=[];results={}
for name,rawfolder in [('baseline_v2','baseline_v2_gpu0'),('self_off_v2','self_off_v2_gpu0')]:
 result=json.loads((R/(name.upper()+'_RESULTS.json')).read_text());results[name]=result
 folder=RAW/rawfolder;rec=json.loads((folder/'recording_receipt.json').read_text());meta=json.loads((folder/'native_metadata.json').read_text());data={}
 for chunk in rec['chunks']:
  with np.load(folder/chunk['file']) as d:
   for key in d.files:
    if key.endswith((':q',':target',':hand_net')) or key=='time_s':data.setdefault(key,[]).append(d[key])
 data={key:np.concatenate(val) for key,val in data.items()};idx=np.unique(np.r_[np.arange(0,reg['steps'],6),reg['steps']-1])
 for e in range(len(cases)):
  for arm in series[e]:
   hid=meta['arms'][arm]['hand_ids'];default=np.asarray(meta['arms'][arm]['hand_default_rad'])[e]
   q=data[arm+':q'][:,e,hid];target=data[arm+':target'][:,e,hid]
   series[e][arm][name]=dict(time=np.round(data['time_s'][idx],6).tolist(),error=np.round(np.abs(q-default).max(-1)[idx].astype(float),6).tolist(),
    tracking=np.round(np.abs(q-target).max(-1)[idx].astype(float),6).tolist(),contact=np.round(np.linalg.norm(data[f'e{e}:{arm}:hand_net'],axis=-1).sum(-1)[idx].astype(float),6).tolist())
 for img in rec['images']:images.append(dict(**img,condition=name,url=base+rawfolder+'/'+img['file']))
core=['REGISTRATION_V2.json','REGISTRATION_SELF_OFF_V2.json','ANALYSIS_SEAL.json','BASELINE_V2_RESULTS.json','SELF_OFF_V2_RESULTS.json',
 'NATIVE_DIAGNOSIS.json','FAILED_ATTEMPT_V1.json','FAILED_ATTEMPT_GPU1.json','ASSET_READBACK.json','SOURCE_AFTER.json','MEDIA_ARCHIVE.json',
 'FAILURE_CONTRACT.md','REPRODUCE.md','analyze.py','native_probe_v2.py','REGISTRATION_OPEN_POSE.json','OPEN_POSE_ANALYSIS_SEAL.json','OPEN_POSE_RESULTS.json','analyze_open_pose.py','native_probe_v3.py','build_panel_v2.py','BUILD_V1_FAILURE.json','NEXT.md']
for name in core:shutil.copyfile(R/name,dest/name)
links=[dict(label='原配置结果JSON',url='BASELINE_V2_RESULTS.json'),dict(label='自碰诊断结果JSON',url='SELF_OFF_V2_RESULTS.json'),
 dict(label='原配置预登记',url='REGISTRATION_V2.json'),dict(label='诊断对照预登记',url='REGISTRATION_SELF_OFF_V2.json'),
 dict(label='机制诊断',url='NATIVE_DIAGNOSIS.json'),dict(label='复现与边界',url='REPRODUCE.md'),
 dict(label='完整原始数据ZIP',url=archive['zip_url']),dict(label='逐文件SHA清单',url=base+'MEDIA_MANIFEST.json')]
data=dict(curve_display_rounding_decimal_places=6,cases=cases,series=series,images=images,results=results,capture_steps=reg['capture_steps'],archive=archive,
 verdict=json.loads((R/'NATIVE_DIAGNOSIS.json').read_text())['display_paragraphs'],links=links)
opening=json.loads((R/'OPEN_POSE_RESULTS.json').read_text());data['opening_pose']=opening
ore=json.loads((RAW/'open_pose_v3_gpu0/recording_receipt.json').read_text());data['opening_images']=[dict(**img,url=base+'open_pose_v3_gpu0/'+img['file']) for img in ore['images']]
(dest/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
shutil.copyfile(R/'panel.html',dest/'index.html');shutil.copyfile(R/'panel.js',dest/'panel.js')
card='<section class="sci-note" id="handSupportCalibrationEvidence"><h2>最新机制校准：真实开手与完整支撑归因</h2>'
card+=''.join('<p>'+v+'</p>' for v in data['verdict'])
card+='<p>20组静态原生诊断，全部180张俯视/正视/侧视原图、关节与接触曲线、预登记和原始数据公开。没有增加任务或正式留出样本，关闭自碰仅用于诊断；真实抓持、四臂共持、广域随机和全面安全仍待准入。</p><p><a href="docs/safety_hand_support_calibration_20261007/">查看物理对照、完整受力伙伴与全部原图 →</a></p></section>\n'
before=(R/'ROOT_BEFORE.html').read_text();assert 'handSupportCalibrationEvidence' not in before
root=before.replace('<section class="sci-note" id="releaseFeedbackEvidence">',card+'<section class="sci-note" id="releaseFeedbackEvidence">',1)
assert root!=before;(a.repo/'index.html').write_text(root)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(dest)):sha(path) for path in sorted(dest.rglob('*')) if path.is_file() and path.name!='PUBLIC_MANIFEST.json'}
(dest/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(files=files,scope='Small presentation/diagnostic evidence only; all original media/raw diagnostics held at fixed external branch commit.'),indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=sum(path.stat().st_size for path in dest.rglob('*') if path.is_file()),images=len(images),archive=a.archive_commit)))
