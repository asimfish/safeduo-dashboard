"""Report06 overlay: actual fixed supplemental MODEL_ONLY data, no native actions."""
import argparse, ast, copy, csv, datetime, hashlib, html, json, os, types
from html.parser import HTMLParser
from pathlib import Path

H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent
O=R/'astra_report_increment06_v2'
Z=O/'build_attempt1'
OUT=O/'build/candidate06'
PREFIX='model_supplement_increment06_v1'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BASE='ef5f9e676dffb248bb145d6bc1433808f921b9a0'
BIND_SHA='bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'
PREP_SHA='523ff5a27d75768b077449a190fd19ec131fe9e566c01d7b6962f1daf1d07a53'
BUILDER05_SHA='8a8d677c6bda098393cc94ad19a398a601226b8d8de58adbf4796b7783bb5992'
PREPARED=dict(path=str(R/'astra_report_increment06_v1/cpu_attempt1/CLOSED_PREPARED_FACT_SECTION_HANDOFF_V1.json'),sha256='7c6e565f4ca20b358db876afd1d56527ff15a191ca63aed57e5862d8dc6ef8d2')
STATES=('MODEL_NEGATIVE','MODEL_PREDICTED_INITIAL_CLEAR','MODEL_UNKNOWN')

def module(path, sha, name):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('authenticated same source buffer')
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def presentation_helpers(b):
    path=H/'astra/report_increment05_v4/build_report_v1.py'
    raw=b.buffer(dict(path=str(path),sha256=BUILDER05_SHA),1024**2)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('replace_id','table')]
    b.need({n.name for n in nodes}=={'replace_id','table'},'exact two unchanged HTML helpers')
    ns=dict(HTMLParser=HTMLParser,html=html,need=b.need)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['replace_id'],ns['table']

def copy_artifact(b,q,path,cap=64*1024**2,newlines=None):
    src=b.owned_path(q['path']);b.need(src.stat().st_size<=cap,'narrow bounded artifact copy')
    path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=total=0
    with src.open('rb') as f,path.open('xb') as g:
        a=os.fstat(f.fileno())
        for raw in iter(lambda:f.read(1024**2),b''):
            total+=len(raw);b.need(total<=cap,'stream copy bound');h.update(raw);count+=raw.count(b'\n');g.write(raw)
        c=os.fstat(f.fileno());g.flush();os.fsync(g.fileno())
    b.need((a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(c.st_dev,c.st_size,c.st_mtime_ns,c.st_ctime_ns) and h.hexdigest()==q['sha256'],'stable source FD and exact copied SHA')
    if newlines is not None:b.need(count==newlines,'complete authenticated line count')
    return b.ref(path)

def plot(b,oldcells,newcells,overlay,refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'report06-fixed-two-denominators'})
    fig,axes=plt.subplots(2,1,figsize=(13.5,8.2),sharex=True)
    labels=[f's{s}/f{f}' for s in range(4) for f in range(4)]
    for ax,cells,n,title in zip(axes,[oldcells,newcells],[512,2048],['Original8192:16 cells x512; original schedule EMPTY/HELD','Fixed supplement32768:16 cells x2048; 32 DERIVED_NOT_ADOPTED']):
        neg=[c['counts']['MODEL_NEGATIVE'] for c in cells];clear=[c['counts']['MODEL_PREDICTED_INITIAL_CLEAR'] for c in cells];unknown=[c['counts']['MODEL_UNKNOWN'] for c in cells]
        xs=list(range(16));ax.bar(xs,neg,color='#a6542d',label='MODEL_NEGATIVE')
        ax.bar(xs,clear,bottom=neg,color='#216487',label='PREDICTED_INITIAL_CLEAR')
        ax.bar(xs,unknown,bottom=[a+c for a,c in zip(neg,clear)],color='#999999',label='MODEL_UNKNOWN')
        for i,v in enumerate(clear):ax.text(i,n*1.022,str(v),ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,n*1.16);ax.set_ylabel('All model states in each cell');ax.set_title(title,pad=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_xticks(xs,labels);ax.tick_params(labelbottom=True);ax.text(.995,.015,'Numbers above bars: prediction CLEAR counts',transform=ax.transAxes,ha='right',fontsize=9)
    axes[0].patches[13].set_edgecolor('#ab2222');axes[0].patches[13].set_linewidth(2)
    axes[0].annotate('s3/f1:CLEAR1<quota2',xy=(13,512),xytext=(9.5,512*1.13),fontsize=9,color='#9d2222',arrowprops={'arrowstyle':'->','color':'#9d2222'})
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.0))
    fig.text(.015,.016,'Cumulative accounting40960=8192+32768:NEG40004 / prediction CLEAR956 / UNKNOWN0. Separate denominators; not pooled IID physical experiments.\nNo new native or camera; BANK0; drive peak and global physical error bound UNKNOWN.',fontsize=10)
    fig.subplots_adjust(top=.90,bottom=.14,hspace=.40,left=.075,right=.99)
    dest=overlay/PREFIX/'figures';dest.mkdir(parents=True)
    for ext in ('svg','png'):fig.savefig(dest/('original8192_fixed32768.'+ext),dpi=170,metadata={'Creator':'Astra; source-bound MODEL_ONLY report06'})
    plt.close(fig)
    return b.write(overlay/PREFIX/'data/FIGURE_SOURCES_V1.json',dict(original_source=refs['old_facts'],supplement_source=refs['data_peer'],original_cells=oldcells,supplement_cells=newcells,denominators=[8192,32768],cumulative_accounting_only=40960,counts_labels=STATES,numeric_gap_replay_performed_by_plot=0,photo_or_native_image_generated=False))

def build(b):
    before=b.ref(__file__);prepared=b.read(PREPARED);b.closed_wait(prepared['actual_worker_wait']);b.closed_wait(prepared['actual_wrapper_wait'])
    tool=b.read(b.ref(R/'astra_report_increment06_v1/cpu_attempt1/ACTUAL_TOOL_CLOSURE_V1.json'));b.need(tool['actual_exit']==0 and tool['session_closed'],'actual prepared06 top tool0')
    prep=module(H/'astra/report_increment06_v1/prepare_v1.py',PREP_SHA,'_closed06_preparer')
    pub=b.read(prep.PUBLIC05);base=b.read(pub['handoff']);b.need(pub['commit']==base['commit']==BASE,'actual published05 parent')
    oldplan=b.read(base['plan']);oldmanifest=b.read(base['manifest']);oldfacts=b.read(base['facts']);facts=b.read(prepared['facts'])
    data=b.read(prep.DATA);proto=b.read(data['protocol']);source=b.read(data['source_input_data_peer']);schedule=b.read(data['prospective32']);cohort=b.read(data['supplemental_cohort'])
    prep.check(b,data,proto,source,schedule)
    for q in facts['actual_waits']:b.closed_wait(q)
    b.need(cohort['denominator']==32768 and cohort['protocol']==data['protocol'] and cohort['source_graph']==data['source_graph'] and cohort['duplicates_not_resampled'],'same frozen supplement INPUT metadata')
    oldcells=oldfacts['model8192']['cells'];newcells=data['accounting']['cells']
    b.need([(c['seed'],c['family']) for c in oldcells]==[(c['seed'],c['family']) for c in newcells] and all(c['denominator']==512 for c in oldcells),'separate original16x512 and supplement16x2048')
    b.need(oldfacts['model_total_counts']==facts['original8192']['counts'] and oldfacts['schedule']['candidates']==[] and facts['original8192']['schedule']=='EMPTY_REMAINS_EMPTY','old8192 unchanged and EMPTY')
    OUT.mkdir(parents=True,exist_ok=False);overlay=OUT/'overlay';overlay.mkdir()
    refs=dict(old_facts=base['facts'],data_peer=prep.DATA,source_input_peer=data['source_input_data_peer'],protocol=data['protocol'],supplemental_cohort=data['supplemental_cohort'],prospective32=data['prospective32'],parent_actual_closed=data['parent_actual_screen_closed'],MODEL_STREAM=data['stream'])
    figure_sources=plot(b,oldcells,newcells,overlay,refs)
    for name,q in refs.items():copy_artifact(b,q,overlay/PREFIX/'evidence'/(name+'.json'))
    for name,q in [('ORIGINAL8192_INPUT_REPLAY_EXACT_V1.json',cohort['original_reproduction']),('INPUT_BOUNDS_V1.npz',cohort['input_bounds']),('NATIVE_PARAMETERS_V1.npz',cohort['parameter']),('PROSPECTIVE32_Q_QD_IDENTITIES_V1.npz',data['prospective32_input'])]:copy_artifact(b,q,overlay/PREFIX/'inputs'/name)
    for name,sha in cohort['files'].items():copy_artifact(b,dict(path=name,sha256=sha),overlay/PREFIX/'inputs'/Path(name).name)
    for name,key,lines in [('ALL32768_AUTHENTICATED_ROWS_V1.jsonl','all32768_authenticated_ROWS',32768),('ALL1024_COMMIT_V1.jsonl','all1024_COMMIT',1024),('ALL1024_RAW9021_HEADERS_SHA_V1.json','all1024_headers',None)]:copy_artifact(b,data['evidence_refs'][key],overlay/PREFIX/'data'/name,newlines=lines)
    graph=b.read(data['source_graph']);sf=H/'astra/full74_supplemental_development_sampling_v2'
    selected=[sf/n for n in ('sampling_v1.py','generate_inputs_v1.py','common_v1.py','extended_rules_v1.py','screen_runner_v1.py','model_adapter_v1.py','MODEL_DATA_DEPENDENCIES_V1.json','DEPENDENCY_PINS_V1.json')]
    selected+=[Path(proto['original_sampler']['path'])]
    copied=[]
    for p in selected:
        q=dict(path=str(p),sha256=graph['files'][str(p)]);v=copy_artifact(b,q,overlay/PREFIX/'source/sampling'/p.name);copied.append(dict(original=q,public_copy=v))
    for name in ('index.html','panel_manifest.json','summary.json'):
        q=base['manifest'] if name=='panel_manifest.json' else base['HTML'] if name=='index.html' else oldplan['file_sources'][name]
        copy_artifact(b,q,overlay/PREFIX/'archive_candidate05'/name)
    copy_artifact(b,b.ref(__file__),overlay/PREFIX/'source/build_v1.py');copy_artifact(b,b.ref(D/'SOURCE_GRAPH_V1.json'),overlay/PREFIX/'source/REPORT_SOURCE_GRAPH_V1.json')
    epoch=data['fresh_absence_utc'];facts.update(schema='astra.report.increment06.actual_facts.v1',status='ACTUAL_CLOSED_MODEL_DATA_PENDING_DISTINCT_REPORT06_SCIENTIFIC_PEER',snapshot_utc=epoch,prepared_fact_section=PREPARED,report06_scientific_peer='PENDING_HOOKE',figure_sources=figure_sources,input_generation_metadata=cohort['original_reproduction'],duplicate_full148_bytes_observed=cohort['duplicate_full148_bytes_observed'],duplicates_not_resampled=True,old7figures_15RGB_SHA_preserved=True,full_model_gap_data_bytes_NAS=32768*9021*8,large_model_gaps_copied_to_public=False)
    fact_ref=b.write(overlay/PREFIX/'FACTS_INCREMENT06_V1.json',facts)
    with (overlay/PREFIX/'data/ALL16_CELLS_V1.csv').open('x',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['batch','seed','family','denominator',*STATES])
        for label,cells in [('original8192',oldcells),('supplement32768',newcells)]:
            for c in cells:w.writerow([label,c['seed'],c['family'],c['denominator'],*(c['counts'][s] for s in STATES)])
    with (overlay/PREFIX/'data/DERIVED32_NOT_ADOPTED_IDENTITIES_V1.csv').open('x',newline='') as f:
        names=['seed','family','proposal_id','global_id','storage_row'];w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader();w.writerows(data['candidate_IDs'])
    readme='Exact source/input/evidence copies for fixed supplemental MODEL_ONLY development.\nOriginal reviewed source preserves absolute environment dependencies; these source copies are not claimed as a standalone relocated runtime.\nThe original8192 portable model kernels/frames remain downloadable under model_point0_increment05_v1/recompute. Supplement source graph/protocol and input arrays are exact bytes.\nNo complete distance matrix is copied:2365063168 float64 payload bytes remain on NAS with1024 fullfile SHA/header/COMMIT records.\nThis report does not rerun models, physics, or cameras. DATA peer replay was fixed12 blocks/384 rows, not all32768 numerical gaps.\nNew32 are derived only, not adopted or native tested. Old8192 schedule remains EMPTY.\n'
    (overlay/PREFIX/'source/REPRODUCTION_SCOPE.txt').write_text(readme)
    b.write(overlay/PREFIX/'source/EXACT_FACT_INPUTS_V1.json',dict(prepared=PREPARED,actual05=prep.PUBLIC05,inputs=refs,copied_sampling_sources=copied,source_only_no_runtime_relocation_claim=True))
    replace_id,table=presentation_helpers(b);page=b.buffer(base['HTML']).decode()
    header='<p class="asof" id="reportAsOf">报告06证据截至 <time datetime="'+epoch+'">'+epoch.replace('T',' ').replace('Z',' UTC')+'</time>：固定新增32768 MODEL_ONLY已全部闭合；782预测初态CLEAR、新32仅NOT_ADOPTED。原8192分母与EMPTY保持；累计40960只是两批台账。原05截至19:16:27及旧04截至17:53:40各自历史正文可下载，原7图与15张INITIAL_ONLY RGB保持原时点。</p>'
    page=replace_id(page,'p','reportAsOf',header)
    notice='<p id="supplement06Executive"><strong>完整安全验收仍未通过。</strong>固定新增32768模型分类31986 NEG／782预测CLEAR／0 UNKNOWN；新32未adopt，新增native／camera0、BANK0。<a href="#modelSupplementIncrement06">独立分母、16格与证据</a></p>'
    b.need(page.count('<div class="executive">')==1,'unique main results');page=page.replace('<div class="executive">','<div class="executive">'+notice,1)
    rows=[[c['seed'],c['family'],2048,*[c['counts'][s] for s in STATES]] for c in newcells]
    idrows=[[x[k] for k in ('seed','family','proposal_id','global_id','storage_row')] for x in data['candidate_IDs']]
    section=b.buffer(prepared['section']).decode().replace('新报告仍待distinct科学内容peer。','报告发布由独立科学内容peer和正常公共验证闭环约束。',1)
    first=section.index('<table id="supplement06CellsTable">');last=section.index('</table>',first)+len('</table>')
    section=section[:first]+table('supplement06CellsTable','固定新增32768，全16格各2048，全部状态保留',['seed','family','分母','NEG','预测CLEAR','UNKNOWN'],rows)+section[last:]
    totals=table('supplement06DenominatorsTable','分母分别报告；累计仅台账，不是合成IID物理验收',['台账批次','分母','MODEL_NEGATIVE','预测INITIAL CLEAR','MODEL_UNKNOWN'],[['原8192',8192,8018,174,0],['固定新增32768',32768,31986,782,0],['累计台账',40960,40004,956,0]])
    figure='<figure id="model06SupplementFigure"><img src="'+PREFIX+'/figures/original8192_fixed32768.svg" alt="原8192各512与固定新增32768各2048分别计数；非物理安全或IID验收"><figcaption>图8：两面板各自分母，蓝色仅预测初态CLEAR；原8192 EMPTY不变，新32仅DERIVED_NOT_ADOPTED。累计40960不增加原生、相机或policy bank。</figcaption></figure>'
    scope='<p>来源与输入不同作者复现：原8192重新生成逐bit／canonical NPY SHA一致，新增32768 q74／qd74及family／IDs一致；重复完整148维bit串观测'+str(cohort['duplicate_full148_bytes_observed'])+'，不因重复而重抽。该复现是SOURCE+INPUT DATA，不是新增物理试验。</p>'
    candidates=table('supplement06CandidatesTable','32个模型派生候选；SUPPLEMENTAL32768_ALONE／NOT_ADOPTED／NOT_TESTED',['seed','family','proposal ID','global ID','storage row'],idrows)
    downloads='<p>可下载本次完整32768逐项处置、全部1024 COMMIT／gaps headers及SHA、原始q74／qd74与身份、固定采样源码。超过2GiB派生距离矩阵留NAS，原native RAW不删；这里只复制输入和审计元数据，未重新解码该矩阵。</p><ul>'
    for path,label in [('FACTS_INCREMENT06_V1.json','本次事实与来源'),('data/ALL32768_AUTHENTICATED_ROWS_V1.jsonl','完整32768逐项MODEL_ONLY处置'),('data/ALL1024_COMMIT_V1.jsonl','全部1024闭合COMMIT／SHA'),('data/ALL1024_RAW9021_HEADERS_SHA_V1.json','全部9021矩阵headers／SHA'),('data/ALL16_CELLS_V1.csv','两批全部16格计数'),('data/DERIVED32_NOT_ADOPTED_IDENTITIES_V1.csv','32派生身份，未adopt'),('evidence/prospective32.json','派生32完整receipt'),('evidence/data_peer.json','不同作者SCREEN_DATA审查'),('evidence/source_input_peer.json','SOURCE+INPUT复现审查'),('evidence/protocol.json','固定补充协议与四原seed'),('evidence/supplemental_cohort.json','完整输入SHA／block索引'),('inputs/q74.npy','完整32768 q74原字节'),('inputs/qd74.npy','完整32768 qd74原字节'),('inputs/ORIGINAL8192_INPUT_REPLAY_EXACT_V1.json','原8192逐bit再生成证据'),('source/EXACT_FACT_INPUTS_V1.json','可复现sources／inputs exact refs'),('source/sampling/sampling_v1.py','原采样API source'),('source/sampling/generate_inputs_v1.py','固定输入生成source'),('source/sampling/screen_runner_v1.py','原MODEL_ONLY runner source'),('source/REPRODUCTION_SCOPE.txt','复现范围和原环境依赖'),('archive_candidate05/index.html','历史05已发布快照')]:downloads+='<li><a href="'+PREFIX+'/'+path+'" download>'+html.escape(label)+'</a></li>'
    downloads+='</ul>'
    section=section.replace('</section>',scope+totals+figure+candidates+downloads+'</section>')
    b.need(page.count('<section id="latestExperiments">')==1,'original05 section unique');page=page.replace('<section id="latestExperiments">',section+'<section id="latestExperiments">',1)
    needle='<a href="#latestExperiments">数据与模型状态</a>';b.need(page.count(needle)==1,'old TOC entry');page=page.replace(needle,'<a href="#modelSupplementIncrement06">新增32768 / 新32未adopt</a>'+needle,1)
    (overlay/'index.html').write_text(page)
    summary=b.read(oldplan['file_sources']['summary.json']);summary['asof_utc']=epoch;summary['increment06']=facts;summary['historical_candidate05_summary']=PREFIX+'/archive_candidate05/summary.json';b.write(overlay/'summary.json',summary)
    sources={p.relative_to(overlay).as_posix():b.ref(p) for p in overlay.rglob('*') if p.is_file()}
    updated=copy.deepcopy(oldmanifest);updated['files'].update({n:q['sha256'] for n,q in sources.items()});updated['snapshot_updated_utc']=epoch;updated['increment06']=dict(facts=PREFIX+'/FACTS_INCREMENT06_V1.json',scope='fixed development supplement MODEL_ONLY; separate8192 and32768 denominators; bank0',data_peer=prep.DATA,new32_status='DERIVED_PROSPECTIVE32_NOT_ADOPTED',native_tested=0,camera_tested=0,bank_count=0,formal_holdout=False,safety_acceptance=False)
    b.write(overlay/'panel_manifest.json',updated);sources['panel_manifest.json']=b.ref(overlay/'panel_manifest.json')
    changed={n:q['sha256'] for n,q in sources.items()};inherited={n:s for n,s in oldmanifest['files'].items() if n not in changed}
    b.need(all(updated['files'][n]==s for n,s in oldmanifest['files'].items() if n not in ('index.html','summary.json')),'all old05 evidence/7fig/15RGB metadata SHA preserved')
    plan=dict(schema='astra.report.increment06.overlay_plan.v1',base_commit=BASE,base_manifest=base['manifest'],before_sha256={n:base['manifest']['sha256'] if n=='panel_manifest.json' else oldmanifest['files'][n] for n in changed if n=='panel_manifest.json' or n in oldmanifest['files']},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=sources,changed_files=changed,file_bytes={n:Path(q['path']).stat().st_size for n,q in sources.items()},inherited_files=inherited,manifest=b.ref(overlay/'panel_manifest.json'),original7fig_and15RGB_preserved=True,science_figures=8,original_RGB=15,archive_decodes=0,model_gap_arrays_copied=False,source_facts=prepared['facts'],scientific_peer='PENDING_DISTINCT_HOOKE_REPORT06_PEER',browser_publication='NOT_RUN')
    b.write(OUT/'REPORT_OVERLAY_PLAN_V1.json',plan)
    b.need(b.ref(__file__)==before,'own build source unchanged')
    hand=dict(status='BUILT_REPORT06_ACTUAL_MODEL_DATA_PENDING_CPU_WAIT_HOOKE_PEER',candidate='candidate06',base_public_commit=BASE,plan=b.ref(OUT/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],HTML=b.ref(overlay/'index.html'),facts=fact_ref,overlay=str(overlay),source_graph=b.ref(D/'SOURCE_GRAPH_V1.json'),actual_DATA_peer=prep.DATA,source_INPUT_peer=data['source_input_data_peer'],prepared_source_fact_handoff=PREPARED,actual05=prep.PUBLIC05,new_paths=len(changed),delta_bytes=sum(plan['file_bytes'].values()),manifest_objects=len(updated['files'])+1,old7fig_and15RGB_SHA_preserved=True,source_before_after_unchanged=True,bank_count=0,safety_acceptance=False,task_acceptance=False,formal_holdout=False,source_native_GPU_Git_W_actions=0,model_numeric_replay_here=0,large_gap_arrays_read_here=0,distinct_scientific_reviewer_requested='Hooke',report06_scientific_peer=None,publish_performed=False)
    return b.write(OUT/'CANDIDATE06_BUILD_HANDOFF_V1.json',hand)

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--source-sha256',required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--child',action='store_true');a=ap.parse_args()
    b=module(H/'astra/report_increment05_v4/binding_v1.py',BIND_SHA,'_closed_report_binding')
    b.need(b.sha(__file__)==a.source_sha256 and os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in b.THREADS),'own source and threads1 CPU-only environment')
    if a.child:print(json.dumps(build(b)),flush=True);return 0
    argv=[PY,'-B',str(Path(__file__).resolve()),*os.sys.argv[1:]]
    if a.run:
        loader=module(H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py','8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512','_report06_loader')
        core=loader.load_reviewed_core('82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626')
        w=core.bounded_child(argv+['--child'],dict(os.environ,MPLCONFIGDIR=str(Z/'matplotlib_config')),Z/'worker',backend='CPU_REPORT06_PLOT_OVERLAY_BUILD',max_wall=300,cpu_limit=True);b.closed_wait(b.ref(Z/'worker/ACTUAL_WAIT.json'));return w['actual_wait_exit']
    O.mkdir(exist_ok=True);Z.mkdir(exist_ok=False)
    observer=module(H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py','65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc','_report06_observer')
    observer.observe(argv+['--run'],Z/'external')
    worker=b.ref(Z/'worker/ACTUAL_WAIT.json');outer=b.ref(Z/'external/ACTUAL_WAIT.json');b.closed_wait(worker);b.closed_wait(outer)
    d=b.read(b.ref(OUT/'CANDIDATE06_BUILD_HANDOFF_V1.json'));d.update(status='CLOSED_REPORT06_CPU_CANDIDATE_PENDING_DISTINCT_HOOKE_SCIENTIFIC_PEER',actual_worker_wait=worker,actual_wrapper_wait=outer,CPU_AS_bytes=1024**3,threads=1)
    hand=b.write(Z/'CLOSED_CANDIDATE06_HANDOFF_V1.json',d)
    print(json.dumps(dict(handoff=hand,worker_wait=worker,wrapper_wait=outer)),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
