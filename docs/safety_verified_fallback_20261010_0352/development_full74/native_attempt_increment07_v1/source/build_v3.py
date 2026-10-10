"""Small historical report07 metadata overlay. No model, native, or Git actions."""
import argparse, ast, copy, datetime, hashlib, html, json, os, types
from html.parser import HTMLParser
from pathlib import Path

H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent
O=R/'astra_report_increment07_v1'
OUT=O/'build/candidate07_v3'
Z=O/'build_attempt3'
PREFIX='native_attempt_increment07_v1'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BASE='275b0ca1f8da6af6b3c53c986d620c6a40968454'
BIND_SHA='bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'
BUILDER05_SHA='8a8d677c6bda098393cc94ad19a398a601226b8d8de58adbf4796b7783bb5992'

def ref(root,path,sha):return dict(path=str(root/path),sha256=sha)

INPUTS={
 'public06':ref(R,'parent_report_publish_increment06_v1/closeout01/CLOSED_PUBLICATION_HANDOFF_V1.json','222358504dd0f8072df4b7cfb952af63bb53101fbd7b1c247b2f03a33e7d7652'),
 'failed_v3':ref(R,'parent_supplement_native32_intake_v1/campaign_v1/actual_native01/PARENT_ACTUAL_NATIVE_FAILED_CLOSED_V1.json','4bcca58b887de76b7bac12d305af700831c2c8499aba3589f81268e647d74549'),
 'failed_metadata_peer':ref(R,'astra_full74_supplement_native_semantic_camera_peer_v1/failed_metadata_peer_v1/CLOSED_FAILED_METADATA_PEER_V1.json','87cf0e7103298349d3a0d4d818cad79ec5e71a5d3958dceb95b2df1da6a42e72'),
 'repair_handoff':ref(H,'astra/full74_supplemental_mixed32_native_v4/release4/FINAL_HANDOFF_V4.json','bb16c1d06e58036bebdeaa008bf7bc826a1a12bb331b2761c7a948ab845fc2e1'),
 'repair_peer':ref(R,'astra_full74_supplemental_mixed32_native_v4_independent_review_v1/REVIEW_V2.json','04ee57398debcfd088836484e7a04e59df3751f3136428eba06fdcf6103211e9'),
 'helper_handoff':ref(R,'parent_supplement_native32_retry_intake_v1/CLOSED_HELPER_HANDOFF_V1.json','3bcb15a24dc365a2cb20fcc27e29ce19b52d8918b8ebe150e200eae11311e205'),
 'helper_peer':ref(R,'astra_parent_supplement_native32_retry_intake_independent_review_v1/REVIEW_V1.json','001b48a364ca7b4f5b08bfa3467722c6d006e089744f5c17a68d891f76ff47c6'),
 'parent_freeze':ref(R,'parent_supplement_native32_retry_intake_v1/CPU/parent_prepare01/PARENT_ACTUAL_FREEZE_CLOSED_V1.json','2f053f0b38c09561404bb4f1d5c265dddd379537753462c20775d185118e1962'),
 'request_v4':ref(R,'parent_supplement_native32_retry_intake_v1/campaign_v1/FROZEN_NATIVE_REQUEST_V1.json','9fd9091297d882d0a27bb9aa360346ffb5fd31f622cba6d68d59c52e20a77b7f'),
 'review_v4':ref(R,'parent_supplement_native32_retry_intake_v1/campaign_v1/PARENT_REVIEW_V1.json','900ed7f4641465ad2a6fdee363e8adb240264204961384865bdce5924b9ff33f'),
 'native_API_CPU_check':ref(R,'parent_supplement_native32_retry_intake_v1/CPU/parent_request_check01/PARENT_ACTUAL_NATIVE_API_CPU_CHECK_CLOSED_V1.json','2c3b1c5164f480135f97ce4a3efd3ab810c01d88fd773904601c034e83218011'),
}

def module(path,sha,name):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('authenticated exact source buffer')
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def presentation_helpers(b):
    p=H/'astra/report_increment05_v4/build_report_v1.py'
    raw=b.buffer(ref(H,'astra/report_increment05_v4/build_report_v1.py',BUILDER05_SHA),1024**2)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('replace_id','table')]
    b.need({n.name for n in nodes}=={'replace_id','table'},'exact inherited HTML helpers')
    ns=dict(HTMLParser=HTMLParser,html=html,need=b.need)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns['replace_id'],ns['table']

def copy_artifact(b,q,path):
    p=b.owned_path(q['path']);b.need(p.stat().st_size<=4*1024**2,'small metadata only')
    raw=b.buffer(q,4*1024**2);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return b.ref(path)

def wait_metadata(b,q,exit_code):
    """Preserve failed exits; source reuse does not reread native logs or arrays."""
    w=b.read(q)
    b.need(w['waitpid_observed'] is True and w['child_pid']==w['waited_pid'] and type(w['actual_wait_exit']) is int and w['actual_wait_exit']==exit_code and w['raw_wait_status']==exit_code*256,'actual typed owned wait4')
    b.need(not w['error'] and not w['resource_abort'] and w['signals']==[] and w['owned_identities'] and all(b.fresh_absent(i) for i in w['owned_identities']),'recorded clean wait and fresh exact PID/starttick absence')
    b.need(len(w['log_sha256'])==64 and Path(w['log']).is_absolute(),'preserve original log SHA; no log liveness inference')
    return w

def tool(b,q,profile):
    t=b.read(q)
    if profile=='author':ok=t['actual_tool_exit']==0 and t['session_closed'] is True and t['actual_terminal_result']['exit_code']==0
    elif profile=='producer_peer':ok=t['actual_tool_exit']==0 and t['terminal_completion_observed'] is True and t['actual_tool_terminal']['exit_code']==0 and t['active_tool_session'] is None
    elif profile=='metadata_peer':ok=t['exit_code']==0 and t['reaped'] is True and t['session_id'] is None
    else:raise ValueError('explicit actual tool schema')
    b.need(ok,'real closed CPU tool, explicit schema');return t

def facts(b):
    ds={k:b.read(q) for k,q in INPUTS.items()};f=ds['failed_v3'];m=ds['failed_metadata_peer'];a=ds['repair_handoff'];ap=ds['repair_peer'];h=ds['helper_handoff'];hp=ds['helper_peer'];z=ds['parent_freeze'];q=ds['request_v4'];v=ds['review_v4'];c=ds['native_API_CPU_check']
    b.need(ds['public06']['commit']==BASE and ds['public06']['science_figures']==8 and ds['public06']['original_RGB']==15,'actual frozen public06 base')
    b.need(f['native_actual_exit']==0 and f['outer_actual_exit']==f['actual_parent_tool_exit']==1 and f['ENTRY_status']=='failed' and f['producer'] is None and f['error']=='ValueError: selected dependency declared','v3 native0 is a failed pre-challenge attempt')
    b.need(f['completed_challenge_states']==f['camera_groups_collected']==f['bank_count']==0 and f['physical_safety_acceptance'] is False and f['cleanup_errors']==[],'no failure promotion')
    nw=wait_metadata(b,f['native_wait'],0);ow=wait_metadata(b,f['outer_wait'],1);et=b.read(f['entry']);ft=b.read(f['tool_closure'])
    b.need(ft['actual_tool_exit']==1 and ft['session_closed'] is True and ft['actual_terminal_result']['exit_code']==1 and et['status']=='failed' and et['request_sha256']==f['request']['sha256'],'real failed outer/tool and exact ENTRY')
    b.need(m['parent_failure_closure']==INPUTS['failed_v3'] and m['status']=='FAILED_ATTEMPT_METADATA_VERIFIED' and m['native_actual_exit']==0 and m['outer_actual_exit']==m['parent_tool_actual_exit']==1 and m['ENTRY_status']=='failed','independent failed metadata scope')
    b.need(m['registered_related_states']==m['not_evaluated_states']==32 and m['complete_challenge_states']==m['captured_challenge_rows']==m['actual_camera_groups']==m['bank_count']==0 and m['actual_challenge_prefix']=='NOT_REACHED' and m['complete_DATA_PASS'] is False and m['safety_acceptance'] is False and m['all_review_CPU_tools_reaped'] is True,'32 NOT_EVAL, no rows or cameras')
    b.closed_wait(m['actual_reviewer_worker_wait']);tool(b,m['actual_reviewer_tool_terminal'],'metadata_peer')
    b.need(ap['author_handoff']==INPUTS['repair_handoff'] and ap['source_graph']==a['source_graph'] and ap['status']=='COND_PASS' and ap['blocking_findings']==[] and ap['author_new_checks']==30 and ap['independent_CPU_checks']==43,'final producer SOURCE engineering-only peer')
    b.need(a['same32_related_retry_not_new_IID'] and a['original_budget_1200_and_camera_V3_and_floors_unchanged'] and a['native_launches']==0,'same32 source repair with original gates')
    for t in [a['actual_CPU_green_worker_wait'],a['actual_CPU_outer_wait'],ap['actual_worker_wait']]:b.closed_wait(t)
    tool(b,a['actual_CPU_tool_closure'],'author');tool(b,ap['actual_tool_wait'],'producer_peer')
    b.need(hp['helper_handoff']==INPUTS['helper_handoff'] and hp['source_graph']==h['source_graph'] and hp['status']=='COND_PASS' and hp['blocking_findings']==[] and hp['independent_CPU_checks']==42 and h['CPU_checks']==35,'helper metadata engineering-only peer')
    for t in h['actual_worker_waits']:b.closed_wait(t['wait'])
    b.closed_wait(h['actual_outer_wait']);b.closed_wait(hp['actual_worker_wait']);tool(b,h['actual_tool_wait'],'author');tool(b,hp['actual_tool_wait'],'author')
    b.need(z['request']==INPUTS['request_v4'] and z['native_dispatched'] is False and z['related_same32_not_new_IID'] is True and z['actual_worker_exit']==z['actual_outer_exit']==z['actual_tool_exit']==0,'actual parent CPU freeze only')
    b.closed_wait(z['worker_wait']);b.closed_wait(z['outer_wait']);tool(b,z['tool_closure'],'author')
    b.need(q['parent_frozen'] is True and q['draft_only'] is False and q['cases']==32 and len(q['candidates'])==32 and q['prefix_microsteps']==12 and q['max_wall_seconds']==1200 and q['policy_actions']==0 and q['bank_allowed'] is False and q['formal_holdout'] is False,'unchanged prospective32/12 and no bank or holdout')
    oldq=b.read(f['request']);b.need(q['candidates']==oldq['candidates']==a['reviewed_conditional_candidate_IDs'],'all32 frozen identities unchanged')
    b.need(q['registered_denominator']==32768 and q['original_denominator_preserved']==8192 and q['related_observations_not_new_IID'] is True and q['same32_pre_challenge_retry']['failure_closure']==INPUTS['failed_v3'],'both model denominators and related attempt history')
    b.need(v['request_sha256']==INPUTS['request_v4']['sha256'] and v['source_graph']==q['source_graph']==a['source_graph'] and v['independent_review']==INPUTS['repair_peer'] and v['helper_independent_review']==INPUTS['helper_peer'] and v['helper_source_graph']==h['source_graph'] and v['actual_parent_freeze']==INPUTS['parent_freeze'],'exact new request/review/source/peer chain')
    b.need(c['actual_parent_tool_exit']==c['actual_worker_exit']==0 and c['session_closed'] is True and c['native_started'] is False and c['new_native_root_exists'] is False and c['GPU0_free_MiB']==9400 and c['original_initial_min_free_MiB']==q['resource']['min_initial_free_MiB']==11000 and c['read_only_GPU_snapshot_is_not_permission'] is True,'CPU native API check and historical resource snapshot')
    cw=b.closed_wait(c['worker_wait']);b.need(INPUTS['request_v4']['path'] in cw['argv'] and INPUTS['request_v4']['sha256'] in cw['argv'] and INPUTS['review_v4']['path'] in cw['argv'],'actual check argv binds final frozen request')
    epoch=c['closed_utc'];b.need(epoch=='2026-10-10T21:30:27.935059+00:00' and c['tool_following_clock_UTC']=='2026-10-10 21:29:22 UTC','fixed historical boundary; never a live status')
    evidence=dict(INPUTS);evidence.update(native_wait_v3=f['native_wait'],outer_wait_v3=f['outer_wait'],tool_v3=f['tool_closure'],ENTRY_v3=f['entry'],request_v3=f['request'],metadata_result=m['metadata_peer'],repair_SOURCE_graph=a['source_graph'],helper_SOURCE_graph=h['source_graph'],parent_API_CPU_wait=c['worker_wait'])
    x=dict(schema='astra.report.increment07.actual_historical_facts.v1',status='ACTUAL_METADATA_BOUND_PENDING_DISTINCT_REPORT07_PEER',snapshot_utc=epoch,snapshot_Asia_Shanghai=datetime.datetime.fromisoformat(epoch).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),historical_snapshot_not_live=True,first_v3=dict(native_actual_exit=0,native_raw_wait_status=nw['raw_wait_status'],ENTRY_status='failed',outer_actual_exit=1,outer_raw_wait_status=ow['raw_wait_status'],tool_actual_exit=1,registered_related_states=32,NOT_EVAL=32,completed_challenge_rows=0,camera_groups=0,challenge_prefix='NOT_REACHED',error=f['error'],failure_rootcause_scope=m['omissions_scope']),repair_v4=dict(status='SOURCE_CPU_REVIEWED_READY_NOT_NATIVE_MEASURED',author_CPU_engineering_checks=30,independent_CPU_engineering_checks=43,helper_CPU_metadata_checks=35,helper_independent_CPU_metadata_checks=42,engineering_checks_are_not_experimental_samples=True,final_request=INPUTS['request_v4'],parent_review=INPUTS['review_v4'],native_status_at_cutoff='NOT_STARTED',native_root_absent_at_historical_CPU_check=True,native_wait=None,native_wait_missing_is_not_zero=True,same32_related_not_new_IID=True),resource_snapshot=dict(asof_utc='2026-10-10T21:29:22+00:00',physical_gpu=0,free_MiB=9400,original_initial_floor_MiB=11000,is_live=False,is_native_permission=False),original8192=dict(denominator=8192,negative=8018,predicted_initial_clear=174,unknown=0,schedule='EMPTY_REMAINS_EMPTY'),fixed32768=dict(denominator=32768,negative=31986,predicted_initial_clear=782,unknown=0,model_clear_is_not_safe=True),cumulative_accounting_only=40960,evidence=evidence,source_actions=dict(native_GPU=0,Git=0,model_replay=0,large_gap_matrix_decodes=0,new_figures=0,new_IID_samples=0),bank_count=0,safety_acceptance=False,task_acceptance=False,formal_holdout_opened=False,CONTACT='UNKNOWN',actual_drive_peak='UNKNOWN',wrench_frame='UNKNOWN',global_error_bound='UNKNOWN',TERMINAL_SET='UNKNOWN',tasks='NOT_RUN',holdout='UNOPENED',science_figures=8,original_RGB=15,old06_original_assets_preserved=True,report07_scientific_peer=None)
    return x,ds,evidence

def build(b):
    before=b.ref(__file__);x,ds,evidence=facts(b);base=ds['public06'];oldplan=b.read(base['plan']);oldmanifest=b.read(base['manifest'])
    OUT.mkdir(parents=True,exist_ok=False);overlay=OUT/'overlay';overlay.mkdir()
    for name,q in evidence.items():copy_artifact(b,q,overlay/PREFIX/'evidence'/(name+'.json'))
    for name in ('index.html','panel_manifest.json','summary.json'):
        q=base['manifest'] if name=='panel_manifest.json' else base['HTML'] if name=='index.html' else oldplan['file_sources'][name]
        copy_artifact(b,q,overlay/PREFIX/'archive_candidate06'/name)
    for name in ('build_v3.py','SOURCE_GRAPH_V3.json'):copy_artifact(b,b.ref(D/name),overlay/PREFIX/'source'/name)
    factref=b.write(overlay/PREFIX/'FACTS_INCREMENT07_V1.json',x)
    replace_id,table=presentation_helpers(b);page=b.buffer(base['HTML'],4*1024**2).decode();epoch=x['snapshot_utc'];local=x['snapshot_Asia_Shanghai']
    header='<p class="asof" id="reportAsOf">报告07科学快照截至 <time datetime="'+epoch+'">2026-10-10 21:30:27.935059 UTC</time>（Asia/Shanghai：2026-10-11 05:30:27.935059）。这是固定历史证据边界，不是实时运行状态；之后的启动或结果另版记录。报告06截至2026-10-10 19:54:39 UTC，05截至19:16:27 UTC，04截至17:53:40 UTC；各历史正文、8图和15张原生RGB按原时点保留。</p>'
    page=replace_id(page,'p','reportAsOf',header)
    notice='<p id="native07Executive"><strong>安全验收仍未通过，BANK0。</strong>同32首次v3原生attempt真实失败：native0／ENTRY failed／outer1／tool1，32 NOT_EVAL、挑战行0、相机0。v4修复及父级请求CPU检查已闭合；截至本版快照v4 NOT_STARTED，没有原生测量结果。782仅模型预测CLEAR。<a href="#nativeAttemptIncrement07">失败、修复与历史资源快照</a>；以下报告06及更早各段是对应截止时点的历史记录。</p>'
    b.need(page.count('<div class="executive">')==1,'single executive');page=page.replace('<div class="executive">','<div class="executive">'+notice,1)
    states=table('native07AttemptTable','同32 related retry；不增加新IID样本',['阶段','native实际wait','ENTRY','outer / tool','挑战行 / 相机','32状态'],[['首次v3真实attempt','0（raw0）','failed：selected dependency declared','1 / 1','0 / 0','32 NOT_EVAL'],['v4修复后的历史快照','NOT_STARTED；wait缺失，不填写0','未产生；无测量','未启动；不填写0','未产生','同32等待采集']])
    checks=table('native07EngineeringTable','工程检查不属于实验样本或安全验收',['工程边界','作者CPU checks','独立CPU checks','结果与范围'],[['v4采集源码修复',30,43,'SOURCE条件通过；非native DATA'],['父级retry helper',35,42,'metadata／source条件通过；非物理试验'],['父级freeze及最终native API check','worker／outer／tool真0','已绑定final request／review','只做CPU检查；未启动Kit／native']])
    section='<section id="nativeAttemptIncrement07"><h2>报告07：真实失败与同32修复就绪</h2><p>科学截止：<strong>2026-10-10 21:30:27.935059 UTC</strong>／Asia/Shanghai 2026-10-11 05:30:27.935059。首次v3的native进程clean退出0不代表测量成功：ENTRY failed，外层及工具真实退出1；32项全部NOT_EVAL，未到挑战前缀，无挑战状态行、无相机组、无新bank。失败记录和独立FAILED_ATTEMPT_METADATA_VERIFIED审查完整保留。</p>'+states
    section+='<p>实际异常停在GPU helper依赖未被READ_PINS声明。telemetry helper也缺pin是源码检查发现，不能写成第二个实际原生异常。v4 final release补齐选定依赖及真实buffer工程验证；科学参数、同32身份、12micro预算、原相机协议与资源门槛保持。修复就绪不等于原生通过。</p>'+checks
    section+='<p id="native07ResourceSnapshot"><strong>历史资源读数，不是实时状态或启动许可：</strong>2026-10-10 21:29:22 UTC，GPU0 free 9400 MiB &lt; 原initial floor 11000 MiB。随后闭合的最终请求CPU检查记录v4 native NOT_STARTED、attempt root不存在。该读取不代替以后实际启动前fresh资源、串行global lease及owned terminal检查；本版不推断后续运行。</p>'
    section+='<p>最终父级请求已freeze并绑定独立SOURCE/helper审查；它是同32 pre-challenge失败后的related retry，不是新随机采样或新IID实验。原8192：8018 NEG／174预测CLEAR／0 UNKNOWN，原schedule EMPTY/HELD保留；固定新增32768：31986 NEG／782预测CLEAR／0 UNKNOWN。16×512与16×2048两套分母、累计40960台账保持。模型预测CLEAR不升级为安全，也不删首次失败或用CPU工程checks扩充实验样本。</p>'
    section+='<p>当前完整requirements仍未达成：整体安全验收false、BANK0；实际Drive peak、CONTACT、wrench frame、全局误差界与terminal set仍UNKNOWN；tasks未运行，formal holdout未打开。没有新增科学图，本版不重算模型gaps、不解码原native ZIP。</p><ul>'
    labels={'failed_v3':'真实首次v3失败闭环','failed_metadata_peer':'独立失败metadata审查','native_wait_v3':'真实native wait0（不代表测量成功）','outer_wait_v3':'真实outer wait1','tool_v3':'真实tool1','ENTRY_v3':'真实failed ENTRY','repair_handoff':'v4 final采集修复SOURCE／CPU闭合','repair_peer':'v4独立SOURCE审查30＋43工程checks','helper_handoff':'retry helper作者35 metadata checks','helper_peer':'retry helper独立42 metadata checks','parent_freeze':'真实父级freeze CPU闭合','request_v4':'同32最终冻结请求','review_v4':'最终父级review及原门槛条件','native_API_CPU_check':'最终native API CPU闭合与历史资源读数','repair_SOURCE_graph':'修复SOURCE SHA图','helper_SOURCE_graph':'父级helper SOURCE SHA图'}
    for name,label in labels.items():section+='<li><a href="'+PREFIX+'/evidence/'+name+'.json" download>'+html.escape(label)+'</a></li>'
    section+='<li><a href="'+PREFIX+'/FACTS_INCREMENT07_V1.json" download>本版全部typed事实与exact来源</a></li><li><a href="'+PREFIX+'/archive_candidate06/index.html">冻结的已发布报告06历史正文</a></li></ul></section>'
    needle='<section id="modelSupplementIncrement06">';b.need(page.count(needle)==1,'unique historical06 section');page=page.replace(needle,section+needle,1)
    needle='<a href="#modelSupplementIncrement06">新增32768 / 新32未adopt</a>';b.need(page.count(needle)==1,'existing06 TOC');page=page.replace(needle,'<a href="#nativeAttemptIncrement07">同32真实失败 / 修复就绪快照</a>'+needle,1)
    (overlay/'index.html').write_text(page)
    summary=b.read(oldplan['file_sources']['summary.json']);summary['asof_utc']=epoch;summary['asof_AsiaShanghai']=datetime.datetime.fromisoformat(epoch).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat();summary['increment07']=x;summary['historical_candidate06_summary']=PREFIX+'/archive_candidate06/summary.json';b.need(summary['asof_utc']==x['snapshot_utc'] and summary['asof_AsiaShanghai']==x['snapshot_Asia_Shanghai'] and datetime.datetime.fromisoformat(summary['asof_utc'])==datetime.datetime.fromisoformat(summary['asof_AsiaShanghai']),'summary UTC/local derived same frozen instant as typed facts');b.write(overlay/'summary.json',summary)
    sources={p.relative_to(overlay).as_posix():b.ref(p) for p in overlay.rglob('*') if p.is_file()}
    updated=copy.deepcopy(oldmanifest);updated['files'].update({n:q['sha256'] for n,q in sources.items()});updated['snapshot_updated_utc']=epoch;updated['increment07']=dict(facts=PREFIX+'/FACTS_INCREMENT07_V1.json',scope='historical failed nativeattempt and SOURCE_CPU repair; not realtime',native_first_v3='FAILED_ENTRY_NATIVE0_OUTER1_TOOL1',repair_v4_native_at_cutoff='NOT_STARTED',same32_related_not_new_IID=True,science_figures=8,original_RGB=15,bank_count=0,safety_acceptance=False,formal_holdout=False)
    b.write(overlay/'panel_manifest.json',updated);sources['panel_manifest.json']=b.ref(overlay/'panel_manifest.json')
    changed={n:q['sha256'] for n,q in sources.items()};inherited={n:s for n,s in oldmanifest['files'].items() if n not in changed}
    b.need(all(updated['files'][n]==s for n,s in oldmanifest['files'].items() if n not in ('index.html','summary.json')),'all prior assets/8fig/15RGB metadata SHA unchanged')
    plan=dict(schema='astra.report.increment07.overlay_plan.v1',base_commit=BASE,base_manifest=base['manifest'],before_sha256={n:base['manifest']['sha256'] if n=='panel_manifest.json' else oldmanifest['files'][n] for n in changed if n=='panel_manifest.json' or n in oldmanifest['files']},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=sources,changed_files=changed,file_bytes={n:Path(q['path']).stat().st_size for n,q in sources.items()},inherited_files=inherited,manifest=b.ref(overlay/'panel_manifest.json'),old8fig_and15RGB_SHA_preserved=True,science_figures=8,original_RGB=15,new_science_figures=0,archive_decodes=0,model_gap_arrays_read=False,source_facts=factref,scientific_peer='PENDING_DISTINCT_HOOKE_REPORT07_PEER',browser_publication='NOT_RUN')
    b.write(OUT/'REPORT_OVERLAY_PLAN_V1.json',plan);b.need(b.ref(__file__)==before,'author source unchanged')
    return b.write(OUT/'CANDIDATE07_BUILD_HANDOFF_V1.json',dict(status='BUILT_REPORT07_PENDING_CPU_CLOSURE_DISTINCT_HOOKE_PEER',candidate='candidate07',base_public_commit=BASE,plan=b.ref(OUT/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],HTML=b.ref(overlay/'index.html'),facts=factref,overlay=str(overlay),source_graph=b.ref(D/'SOURCE_GRAPH_V3.json'),actual06=INPUTS['public06'],new_paths=len(changed),delta_bytes=sum(plan['file_bytes'].values()),manifest_objects=len(updated['files'])+1,old8fig_and15RGB_SHA_preserved=True,source_before_after_unchanged=True,snapshot_utc=epoch,bank_count=0,safety_acceptance=False,task_acceptance=False,formal_holdout=False,source_native_GPU_Git_W_actions=0,model_numeric_replay_here=0,large_gap_arrays_read_here=0,new_science_figures=0,distinct_scientific_reviewer_requested='Hooke',report07_scientific_peer=None,publish_performed=False))

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--source-sha256',required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--child',action='store_true');a=ap.parse_args()
    b=module(H/'astra/report_increment05_v4/binding_v1.py',BIND_SHA,'_closed_report_binding')
    b.need(b.sha(__file__)==a.source_sha256 and os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in b.THREADS),'bound -B threads1 CPU-only source')
    if a.child:print(json.dumps(build(b)),flush=True);return 0
    argv=[PY,'-B',str(Path(__file__).resolve()),*os.sys.argv[1:]]
    if a.run:
        loader=module(H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py','8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512','_report07_loader');core=loader.load_reviewed_core('82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626')
        w=core.bounded_child(argv+['--child'],dict(os.environ),Z/'worker',backend='CPU_REPORT07_METADATA_ONLY_BUILD',max_wall=180,cpu_limit=True);b.closed_wait(b.ref(Z/'worker/ACTUAL_WAIT.json'));return w['actual_wait_exit']
    O.mkdir(exist_ok=True);Z.mkdir(exist_ok=False)
    observer=module(H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py','65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc','_report07_observer');observer.observe(argv+['--run'],Z/'external')
    worker=b.ref(Z/'worker/ACTUAL_WAIT.json');outer=b.ref(Z/'external/ACTUAL_WAIT.json');b.closed_wait(worker);b.closed_wait(outer)
    d=b.read(b.ref(OUT/'CANDIDATE07_BUILD_HANDOFF_V1.json'));d.update(status='CLOSED_REPORT07_CPU_BUILD_PENDING_LOCAL_QA_HOOKE_PEER',actual_worker_wait=worker,actual_wrapper_wait=outer,CPU_AS_bytes=1024**3,threads=1)
    hand=b.write(Z/'CLOSED_BUILD_HANDOFF_V1.json',d);print(json.dumps(dict(handoff=hand,worker_wait=worker,wrapper_wait=outer)),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
