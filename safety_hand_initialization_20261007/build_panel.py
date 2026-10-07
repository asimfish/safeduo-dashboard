import argparse,gzip,json,shutil,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive-commit',required=True);a=p.parse_args()
R=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand/safety_hand_initialization_20261007')
dest=a.repo/'docs/safety_hand_initialization_20261007';dest.mkdir(exist_ok=True)
base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+a.archive_commit+'/safety_hand_initialization_20261007/'
series={};images=[];results={}
for run in ['old_default','neutral_default']:
    r=json.loads((R/f'REGISTRATION_{run}.json').read_text());results[run]=json.loads((R/(run.upper()+'_RESULTS.json')).read_text())
    folder=RAW/run;rec=json.loads((folder/'recording_receipt.json').read_text());meta=json.loads((folder/'native_metadata.json').read_text());data={}
    for chunk in rec['chunks']:
        with np.load(folder/chunk['file'],allow_pickle=False) as d:
            for key in d.files:
                if key.endswith((':q',':target',':hand_partner_normal')) or key=='time_s':data.setdefault(key,[]).append(d[key])
    data={k:np.concatenate(v) for k,v in data.items()};series[run]={}
    # Preserve every first-second state. Thereafter plot a max envelope per six physical steps.
    blocks=[(i,i+1) for i,t in enumerate(data['time_s']) if t<=1]
    start=blocks[-1][1]
    blocks += [(i,min(i+6,r['steps'])) for i in range(start,r['steps'],6)]
    times=[float(data['time_s'][j-1]) for i,j in blocks]
    for e in range(6):
        series[run][e]={}
        for arm,m in meta['arms'].items():
            ids=m['hand_ids'];q=data[arm+':q'][:,e][:,ids];tgt=data[arm+':target'][:,e][:,ids];op=np.asarray(m['proposed_hand_open_rad'])
            fields=dict(error=np.abs(q-op).max(-1),tracking=np.abs(q-tgt).max(-1),contact=np.linalg.norm(data[f'e{e}:{arm}:hand_partner_normal'],axis=-1).max(axis=(1,2)))
            series[run][e][arm]={'time':np.round(times,6).tolist(),**{k:[round(float(v[i:j].max()),6) for i,j in blocks] for k,v in fields.items()}}
    images += [dict(**img,run=run,url=base+run+'/'+img['file']) for img in rec['images']]
links=[dict(label=label,url=base+file) for label,file in [
    ('原默认完整结果JSON','OLD_DEFAULT_RESULTS.json'),('中性默认完整结果JSON','NEUTRAL_DEFAULT_RESULTS.json'),
    ('原默认预登记','REGISTRATION_old_default.json'),('中性默认预登记','REGISTRATION_neutral_default.json'),
    ('启动前脚本封存','REGISTRATION_SEAL.json'),('原生初始状态：默认0','old_default/initial_states.json'),
    ('原生初始状态：默认0.35','neutral_default/initial_states.json'),('跨运行逐值复核','CROSS_RUN_COMPARISON.json'),
    ('手内全豁免盲区审计','SEMANTICS_GAP.json'),('源文件与资产复核','SOURCE_AFTER.json'),
    ('复现方法与边界','REPRODUCE.md'),('下一步准入','NEXT.md'),('原始逐文件SHA清单','MEDIA_MANIFEST.json')]]
archive=json.loads((R/'MEDIA_ARCHIVE.json').read_text());assert archive['commit']==a.archive_commit
links.append(dict(label='固定提交完整原始ZIP',url=archive['zip_url']))
payload=dict(results=results,summary=json.loads((R/'SUMMARY.json').read_text()),series=series,images=images,links=links,
    series_note='Every state <=1s; per-six-step maximum thereafter. Display rounded to six decimals. Full raw arrays preserved.',archive=archive)
raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
(R/'DISPLAY_PAYLOAD.json').write_bytes(raw)
(dest/'payload.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
for src,dst in [('panel.html','index.html'),('panel.js','panel.js'),('SUMMARY.json','SUMMARY.json'),('MEDIA_ARCHIVE.json','MEDIA_ARCHIVE.json')]:shutil.copyfile(R/src,dest/dst)
card='<section class="sci-note" id="handInitializationEvidence"><h2>最新手部诊断：启动冲击与不合格闭合目标</h2>'
card+=''.join('<p>'+v+'</p>' for v in payload['summary']['paragraphs'])
card+='<p><a href="docs/safety_hand_initialization_20261007/">查看初始化对照、全部原生曲线与180张三视角原图 →</a></p></section>\n'
before=(R/'ROOT_BEFORE.html').read_text();assert 'handInitializationEvidence' not in before
root=before.replace('<section class="sci-note" id="handSupportCalibrationEvidence">',card+'<section class="sci-note" id="handSupportCalibrationEvidence">',1)
assert root!=before;(a.repo/'index.html').write_text(root)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files={str(path.relative_to(dest)):sha(path) for path in sorted(dest.rglob('*')) if path.is_file() and path.name!='PUBLIC_MANIFEST.json'}
(dest/'PUBLIC_MANIFEST.json').write_text(json.dumps(dict(files=files,payload_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),original_images=180),indent=2)+'\n')
print(json.dumps(dict(public_files=len(files),bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),original_images=len(images))))
