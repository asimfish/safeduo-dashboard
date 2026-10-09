"""Re-render the existing plotting block only; preserve native and metric bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

P=Path(__file__).resolve().parent
H=P.parent
OLD=H/'strong_long_delivery_v1'
W=Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    assert not (P/'PUBLIC_PAYLOAD_MANIFEST.json').exists()
    assert read(OLD/'FINISH_DELIVERY_EXECUTION_V2.json')['status']=='complete'
    gate=read(OLD/'FINAL_DELIVERY_GATE.json');assert gate['status']=='PASS_PUBLIC_STRONG_LONG128_DELIVERY_FULL_SYSTEM0_REJECTED'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();assert base==gate['commit']
    previous=read(OLD/'PUBLIC_PAYLOAD_MANIFEST.json');D=W/previous['slug']
    preserved={**read(OLD/'OLDER_EVIDENCE_PRESERVATION_V1.json')['files'],
        **{k:v for k,v in previous['files'].items() if not k.endswith(('.png','.svg','.pdf')) and k!=previous['slug']+'/index.html'}}
    for name,digest in {**read(OLD/'OLDER_EVIDENCE_PRESERVATION_V1.json')['files'],**previous['files']}.items():assert sha(W/name)==digest,name
    (P/'OLDER_EVIDENCE_PRESERVATION_V1.json').write_text(json.dumps(dict(files=preserved),indent=2)+'\n')
    (P/'PUBLIC_PAYLOAD_BEFORE_DISPLAY_FIX_V1.json').write_text(json.dumps(previous,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('closed_builder',OLD/'build_report_v1.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    result=read(builder.L/'STRONG_LONG128_RESULT_V1.json');reg=read(builder.L/'REGISTRATION_V1.json');root=Path(reg['bank']).parent
    metrics={};bindings={}
    for batch in range(2):
        for method in builder.METHODS:
            path=builder.L/f'PAIRED_METRICS_batch{batch}_{method}_V1.npz';metrics[(batch,method)]=builder.load(path);bindings[str(path)]=sha(path)
    bindings[str(builder.L/'STRONG_LONG128_RESULT_V1.json')]=sha(builder.L/'STRONG_LONG128_RESULT_V1.json')
    rows=[{method+'_controlled_path_rad':result['states'][method][i]['controlled_path_rad'] for method in builder.METHODS} for i in range(128)]
    contacts=read(root/'paired_batch0_raw_v8/point_contact_receipts.json')
    with builder.np.load(root/'paired_batch0_raw_v8'/contacts['chunks'][0]['path']) as z:dt=float(builder.np.unique(z['physics_dt'])[0])
    source=(OLD/'build_report_v1.py').read_text();block=textwrap.dedent(source[source.index('    plotdir=D/'):source.index('    default=cases[-1]')])
    block=block.replace('plotdir.mkdir();','plotdir.mkdir(exist_ok=True);')
    block=block.replace("fig.legend(loc='outside upper center',ncol=3)","handles, labels = (axes[0] if isinstance(axes, np.ndarray) else ax).get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)")
    block=block.replace('figsize=(11,8)','figsize=(11,10)')
    block=block.replace("'Minimum raw gap (mm)'","'Minimum raw gap\\n(mm)'").replace("'Scalar normal force (N)'","'Scalar normal force\\n(N)'").replace("'Hard limit excess (rad)'","'Hard limit excess\\n(rad)'").replace("'Speed excess (rad/s)'","'Speed excess\\n(rad/s)'")
    namespace={**vars(builder),'metrics':metrics,'result':result,'reg':reg,'bank':builder.load(Path(reg['bank'])),'allrows':rows,'dt':dt,'cases':[],'D':D}
    exec(compile(block,'closed_native_display_only_repair','exec'),namespace)
    assert namespace['cases']==read(OLD/'PUBLIC_CASE_BINDINGS.json')
    for path,digest in bindings.items():assert sha(Path(path))==digest
    for name,digest in preserved.items():assert sha(W/name)==digest,name
    receipt=dict(status='PASS_DISPLAY_ONLY_REPAIR_NATIVE_METRICS_CASE_BINDINGS_UNCHANGED',changes=['Use one legend per method instead of repeated legends from every subplot','Increase case figure height and put units on a separate label line'],
        native_or_scoring_reexecuted=False,all6_closed_metric_SHA256_unchanged=bindings,all16_case_bindings_unchanged=True,full128_counts_unchanged=True,fullSystem0_accepted=False,source_sha256=sha(Path(__file__)))
    (P/'DISPLAY_ONLY_REPAIR_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ['repair_display_v1.py','DISPLAY_ONLY_REPAIR_V1.json']:
        shutil.copyfile(P/name,D/'evidence'/name)
    index=D/'index.html';text=index.read_text();needle='<a href="trace_subset_receipts.json">';assert text.count(needle)==1
    text=text.replace(needle,'<p><a href="evidence/DISPLAY_ONLY_REPAIR_V1.json">图例与标签展示修复：原生数据、指标与16案例绑定未改变</a> · <a href="evidence/repair_display_v1.py">图表修复代码</a></p>'+needle,1);index.write_text(text)
    changed=[str(p.relative_to(W)) for p in sorted((D/'plots').iterdir()) if p.is_file()]+[str(index.relative_to(W)),previous['slug']+'/evidence/repair_display_v1.py',previous['slug']+'/evidence/DISPLAY_ONLY_REPAIR_V1.json']
    files={name:sha(W/name) for name in changed if previous['files'].get(name)!=sha(W/name)};manifest=dict(slug=previous['slug'],base_commit=base,files=files,total_files=len(files),total_bytes=sum((W/name).stat().st_size for name in files),scope='Display-only repair; all native and numerical results preserved',fullSystem0_accepted=False)
    (P/'PUBLIC_PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copyfile(OLD/'PUBLIC_CASE_BINDINGS.json',P/'PUBLIC_CASE_BINDINGS.json')
    print('DISPLAY_ONLY_REPAIR_READY',len(files),manifest['total_bytes'],flush=True)
if __name__=='__main__':main()
