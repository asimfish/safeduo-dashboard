"""Expose every measured hold failure; original systematic16 bindings stay intact."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import textwrap
P=Path(__file__).resolve().parent
H=P.parent
OLD=H/'strong_long_delivery_v1'
W=Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert not (P/'FAILURE_CASE_SUPPLEMENT_V1.json').exists()
    manifest=read(P/'PUBLIC_PAYLOAD_MANIFEST.json');D=W/manifest['slug'];oldcases=read(P/'PUBLIC_CASE_BINDINGS.json');assert len(oldcases)==16
    for name,digest in manifest['files'].items():assert sha(W/name)==digest,name
    result=read(H/'strong_long128_v1/STRONG_LONG128_RESULT_V1.json');failed=[r['input_id'] for r in result['states']['multirow_hold_fallback'] if r['primary_failure']];assert len(failed)==10
    spec=importlib.util.spec_from_file_location('closed_builder',OLD/'build_report_v1.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    reg=read(builder.L/'REGISTRATION_V1.json');root=Path(reg['bank']).parent;bank=builder.load(Path(reg['bank']));present={c['input_id'] for c in oldcases};newids=[i for i in failed if i not in present]
    pos=[int(builder.np.flatnonzero(bank['selected_input_id']==i)[0]) for i in newids];metrics={};bindings={}
    for batch in range(2):
        for method in builder.METHODS:
            path=builder.L/f'PAIRED_METRICS_batch{batch}_{method}_V1.npz';metrics[(batch,method)]=builder.load(path);bindings[str(path)]=sha(path)
    contacts=read(root/'paired_batch0_raw_v8/point_contact_receipts.json')
    with builder.np.load(root/'paired_batch0_raw_v8'/contacts['chunks'][0]['path']) as z:dt=float(builder.np.unique(z['physics_dt'])[0])
    source=(OLD/'build_report_v1.py').read_text()
    helpers=textwrap.dedent(source[source.index('    plotdir=D/'):source.index('    fig,axes=plt.subplots(1,2')]).replace('plotdir.mkdir();','plotdir.mkdir(exist_ok=True);')
    block=textwrap.dedent(source[source.index("    for pos in presentation['positions']:"):source.index('    default=cases[-1]')])
    block=block.replace('figsize=(11,8)','figsize=(11,10)').replace("fig.legend(loc='outside upper center',ncol=3)","handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)")
    for before,after in [('Minimum raw gap (mm)','Minimum raw gap\\n(mm)'),('Scalar normal force (N)','Scalar normal force\\n(N)'),('Hard limit excess (rad)','Hard limit excess\\n(rad)'),('Speed excess (rad/s)','Speed excess\\n(rad/s)')]:block=block.replace("'"+before+"'","'"+after+"'")
    namespace={**vars(builder),'D':D,'bank':bank,'reg':reg,'metrics':metrics,'dt':dt,'cases':[],'presentation':{'positions':pos}}
    exec(compile(helpers+'\n'+block,'all_closed_hold_failures_case_figures','exec'),namespace)
    newcases=namespace['cases'];assert {c['input_id'] for c in newcases}==set(newids)
    cases=newcases+oldcases;assert cases[-16:]==oldcases and set(failed)<={c['input_id'] for c in cases}
    index=D/'index.html';page=index.read_text();literal='const cases='+json.dumps(oldcases,ensure_ascii=False);assert page.count(literal)==1;page=page.replace(literal,'const cases='+json.dumps(cases,ensure_ascii=False),1)
    options=''.join(f'<option value="{c["input_id"]}">回退失败 · 组合{c["cell"]} · 输入{c["input_id"]}（事后）</option>' for c in newcases)
    page=page.replace('<select id="case">','<select id="case"><optgroup label="补充回退失败（事后诊断）">'+options+'</optgroup><optgroup label="原先固定规则的16例">',1).replace('</select>','</optgroup></select>',1)
    lookup={c['input_id']:c for c in cases};failurelinks=''.join(f'<li>输入 {i} · 组合 {lookup[i]["cell"]}：<a href="{lookup[i]["plot"]}">PNG</a> · <a href="{lookup[i]["plot"].replace(".png",".svg")}">SVG</a> · <a href="{lookup[i]["plot"].replace(".png",".pdf")}">PDF</a></li>' for i in failed)
    note='<section class="card" id="failure-cases"><h2>全部10条回退失败：事后诊断</h2><p>另外补充所有未在原16例中的回退失败曲线，未删任何原病例或原始数据，也未称这些失败例在结果前选择。输入1887的UR左腕/TableU接触峰值为111.743N；其余失败指标见全部128条账本。未启用JavaScript也可逐个打开曲线。</p><ul>'+failurelinks+'</ul><p><a href="evidence/FAILURE_CASE_SUPPLEMENT_V1.json">补充规则、全部10个失败ID与数据不变证明</a></p></section>'
    needle='<h2>如何保证比较一致</h2>';assert page.count(needle)==1;page=page.replace(needle,note+needle,1);index.write_text(page)
    for path,digest in bindings.items():assert sha(Path(path))==digest
    receipt=dict(status='PASS_ALL10_NATIVE_HOLD_FAILURE_CASES_EXPOSED_ORIGINAL16_PRESERVED',all_failed_inputs=failed,newly_added_inputs=newids,total_case_options=len(cases),selection='Exhaustive posthoc hold-primary failures, in addition to original systematic16; not before-outcome preregistration',original16_case_bindings_unchanged=True,all128_metrics_and_counts_unchanged=True,native_or_scoring_reexecuted=False,unchanged_metric_SHA256=bindings,source_sha256=sha(Path(__file__)))
    (P/'FAILURE_CASE_SUPPLEMENT_V1.json').write_text(json.dumps(receipt,indent=2)+'\n');(P/'PUBLIC_CASE_BINDINGS.json').write_text(json.dumps(cases,indent=2)+'\n')
    for name in ['supplement_failure_cases_v1.py','FAILURE_CASE_SUPPLEMENT_V1.json']:shutil.copyfile(P/name,D/'evidence'/name)
    for name in ['FAILURE_CASE_SUPPLEMENT_V1.json','supplement_failure_cases_v1.py']:
        marker='<a href="trace_subset_receipts.json">';text=index.read_text();text=text.replace(marker,f'<p><a href="evidence/{name}">{name}</a></p>'+marker,1);index.write_text(text)
    (P/'PUBLIC_PAYLOAD_BEFORE_FAILURE_SUPPLEMENT_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={name:sha(W/name) for name in manifest['files']}
    for c in newcases:
        for ext in ['png','svg','pdf']:
            name=manifest['slug']+'/'+c['plot'].replace('.png','.'+ext);files[name]=sha(W/name)
    for name in ['supplement_failure_cases_v1.py','FAILURE_CASE_SUPPLEMENT_V1.json']:files[manifest['slug']+'/evidence/'+name]=sha(D/'evidence'/name)
    manifest.update(files=files,total_files=len(files),total_bytes=sum((W/name).stat().st_size for name in files),scope='Display repair and exhaustive posthoc10 hold-failure figures; every original numerical result preserved')
    (P/'PUBLIC_PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(receipt['status'],len(cases),len(files),flush=True)
if __name__=='__main__':main()
