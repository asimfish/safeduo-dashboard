"""Report08: closed failed-attempt metadata only; no RAW replay or native actions."""
import argparse, copy, csv, datetime, hashlib, html, io, json, os, types
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent;O=R/'astra_report_increment08_v1';OUT=O/'build/candidate08_v2';Z=O/'build_attempt2'
PREFIX='native_attempt_increment08_v1';BASE='91703e7c4e8c12e0e65def6e50a8940de1e55bdb';PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BIND_SHA='bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'
BUILD07_SHA='dce7ccdfc222a27053194b6db2f9fb4d2b8adfed6682ac1ee50eaf3fe0274026'

def ref(root,path,sha):return dict(path=str(root/path),sha256=sha)
INPUTS={
 'public07':ref(R,'astra_report_increment07_v1/publication_v2/closeout01/CLOSED_PUBLICATION_HANDOFF_V1.json','9d14af1a44c8fc2d012e6eefb22f2cff56600fe344294b74c0c7f5452759ac60'),
 'public07_parent':ref(R,'astra_report_increment07_v1/publication_v2/closeout01/PARENT_ACTUAL_PUBLICATION_CLOSED_V1.json','3bf3a8e8f2ca99498167f1a5967c6de28cfe781f3c36886b1575f3ea214f827e'),
 'parent_draft':ref(R,'parent_report_increment08_facts_preparation_v1/FACTS_DRAFT_INCREMENT08_V1.json','dcf5a2cd3cc573fa907e7ca949927566de67a4806c06720d74d1419d14b295d8'),
 'failed_v4':ref(R,'parent_supplement_native32_retry_intake_v1/campaign_v1/actual_native01/PARENT_ACTUAL_NATIVE_FAILED_CLOSED_V1.json','a60dd2c1c26b0e78d6b1c922ba488692ce4e171ee2aab04b23a0f8d1c787677d'),
 'failed_metadata_peer':ref(R,'astra_full74_supplement_native_semantic_camera_peer_v1/v4_failed_metadata_peer_v1/CLOSED_FAILED_METADATA_PEER_V1.json','56eb9b9ba69f7b6e22aa7458d1e6df1a18bb8ae61e91a8961527b88e7730e80a'),
 'diagnosis':ref(R,'astra_full74_supplemental_mixed32_native_v4_crash_diagnosis_v1/DIAGNOSIS_V1.json','6943805fbf4ad726e59e592b426e831b6900b377c6c0609b7a4ec6cd0d000f65'),
 'diagnosis_closed':ref(R,'astra_full74_supplemental_mixed32_native_v4_crash_diagnosis_v1/ACTUAL_CLOSED_HANDOFF_V1.json','c0cd5b84163b319ad865ed85caccd6ec92e9603eb478a983ee804aba2429e1e9'),
}

def module(path,sha,name):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('authenticated exact source buffer '+str(path))
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def copy_artifact(b,q,path):
 raw=b.buffer(q,4*1024**2);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return b.ref(path)

def exact_absent(b,identity):
 # Original root snapshots have no live field; closed terminal receipts are checked separately.
 b.need(type(identity['pid']) is int and type(identity['startticks']) is int,'typed original PID/startticks')
 try:
  fields=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return int(fields[19])!=identity['startticks']
 except FileNotFoundError:return True

def failed_wait(b,q,code,rawstatus):
 w=b.read(q);b.need(w['waitpid_observed'] and w['waited_pid']==w['child_pid'] and type(w['actual_wait_exit']) is int and w['actual_wait_exit']==code and w['raw_wait_status']==rawstatus and os.waitstatus_to_exitcode(rawstatus)==code,'actual exact failed wait4 retained')
 b.need(not w['resource_abort'] and not w['error'] and w['signals']==[] and w['owned_identities'] and all(i['live'] is False and b.fresh_absent(i) for i in w['owned_identities']),'clean guardian record and fresh exact owned absence; failure unchanged')
 b.need(Path(w['log']).is_absolute() and len(w['log_sha256'])==64,'preserved native log SHA metadata; no logfile liveness inference');return w

def facts(b):
 ds={k:b.read(q) for k,q in INPUTS.items()};draft=ds['parent_draft'];f=ds['failed_v4'];c=ds['failed_metadata_peer'];d=ds['diagnosis'];dc=ds['diagnosis_closed'];base=ds['public07']
 b.need(base['commit']==BASE and base['publish_paths']==29 and base['science_figures']==8 and base['original_RGB']==15 and ds['public07_parent']['handoff']==INPUTS['public07'],'actual closed public07 REPAIR1 base')
 b.closed_wait(base['closeout_worker_wait']);b.closed_wait(base['closeout_wrapper_wait'])
 b.need(draft['actual_native_closure']==INPUTS['failed_v4'] and draft['snapshot_utc']=='2026-10-10T22:03:37.385121+00:00' and draft['snapshot_AsiaShanghai']=='2026-10-11T06:03:37.385121+08:00','exact parent historical data cutoff')
 b.need(f['native_actual_exit']==draft['native_actual_exit']==-11 and f['outer_actual_exit']==f['actual_parent_tool_exit']==1 and f['entry'] is None and f['producer'] is None and f['registered_candidates']==32 and f['related_same32_retry'] and f['all32_denominator_retained'],'real failed same32 native attempt; no missing receipt invented')
 nw=failed_wait(b,f['native_wait'],-11,139);failed_wait(b,f['outer_wait'],1,256)
 b.need(nw['max_wall_seconds']==1200 and nw['elapsed_seconds']==f['native_elapsed_seconds']==draft['native_elapsed_seconds'] and 0<nw['elapsed_seconds']<1200 and draft['native_timeout'] is False,'SIGSEGV before original finite deadline; not timeout')
 t=b.read(f['tool_closure']);b.need(t['actual_tool_exit']==1 and t['session_closed'] and t['actual_terminal_result']['exit_code']==1,'actual parent tool1 retained')
 b.need(f['committed_numeric_micro_ids']==[0] and f['committed_physical_advances']==0 and len(f['initial_camera_directories'])==24 and f['completed_camera_group_receipts']==0 and f['crash_root_cause']=='UNKNOWN' and f['bank_count']==0 and not f['physical_safety_acceptance'],'partial durable commit metadata; no camera acceptance')
 b.need(c['status']=='FAILED_ATTEMPT_METADATA_VERIFIED' and c['blocking_findings']==[] and c['parent_failure_closure']==INPUTS['failed_v4'] and c['native_wait']==f['native_wait'] and c['outer_wait']==f['outer_wait'] and c['parent_tool_terminal']==f['tool_closure'],'distinct actual failed metadata chain')
 b.need(c['native_actual_exit']==-11 and c['native_raw_status']==139 and c['outer_actual_exit']==c['parent_tool_actual_exit']==1 and c['ENTRY']==c['PRODUCER']=='MISSING' and c['committed_micro_ids']==[0] and c['committed_advances']==0 and c['initial_camera_directories']==24 and c['complete_camera_groups']==c['complete_full12_prefixes']==0 and c['not_evaluated_full_native_states']==32,'complete32 denominator and failure states')
 b.need(c['initial_row_numeric_results']=='NOT_INDEPENDENTLY_RECOMPUTED' and not c['complete_DATA_PASS'] and not c['scientific_PASS'] and not c['RAW_binding_emitted'] and c['same32_related_not_new_IID'] and c['all_review_CPU_tools_reaped'] and c['all_parent244_and_reviewer4_owned_fresh_absent'],'closed metadata is not raw numeric/camera scientific acceptance')
 review=b.read(c['review']);metadata=b.read(c['metadata_peer'])
 for x in (review,metadata):
  b.need(x['camera_pixels_and_angles']=='NOT_REVIEWED' and x['all27_and_latest42_held']=='UNKNOWN_NO_COMPLETE_CAPTURE_RECEIPT' and x['native_measurements_before_after']=='UNKNOWN_NO_ENTRY_OR_PRODUCER_AFTER_CLOSURE' and x['possible_uncommitted_physical_activity']=='UNKNOWN' and x['RAW_arrays_decoded']==x['camera_outputs_decoded']==x['large_model_matrices_opened']==0,'explicit independently unmeasured boundaries preserved')
 b.closed_wait(c['actual_reviewer_worker_wait']);b.closed_wait(c['actual_reviewer_outer_wait'])
 for key in ('actual_reviewer_tool_terminal','actual_closer_tool_terminal','fresh_final_absence_tool'):
  z=b.read(c[key]);b.need(z['exit_code']==0 and z['reaped'] and z['session_id'] is None and z['actual_terminal_observed'],'real metadata reviewer tool schema0')
 for i in review['reviewer_owned_identities']:b.need(exact_absent(b,i),'fresh exact reviewer PIDSTART absence')
 b.need(dc['diagnosis']==INPUTS['diagnosis'] and dc['actual_native_closed']==INPUTS['failed_v4'] and dc['root_cause']==d['root_cause']=='UNKNOWN' and dc['actual_worker_exit']==dc['actual_tool_exit']==0 and dc['all_owned_closed'] and dc['source_and_evidence_before_after_sha_equal'],'actual closed independent diagnosis; rootcause UNKNOWN')
 b.closed_wait(dc['actual_worker_wait']);dt=b.read(dc['actual_tool_wait']);b.need(dt['actual_tool_exit']==0 and dt['session_closed'] and dt['terminal_completion_observed'] and dt['actual_terminal_result']['exit_code']==0 and exact_absent(b,dc['wrapper_identity']),'actual diagnosis tool0 and wrapper absence')
 b.need(d['native_exit']==-11 and d['raw_wait_status']==139 and d['python_stack']['fatal_header_count']==0 and d['python_stack']['timeout_dump_count']==18 and d['python_stack']['fatal_CPP_instruction_pointer']=='UNKNOWN' and d['native_launches']==d['Kit_imports']==d['GPU_queries']==d['signals_sent']==0,'periodic SyntheticData stack not fatal stack or demonstrated cause')
 request=b.read(draft['request']);b.need(f['request']==draft['request'] and request['cases']==32 and request['prefix_microsteps']==12 and request['max_wall_seconds']==1200 and not request['bank_allowed'] and not request['formal_holdout'],'same frozen32/12/1200 and ineligible policy')
 rows=list(csv.DictReader(io.StringIO(b.buffer(draft['all32_rows'],1024**2).decode())))
 keys=('seed','proposal_id','global_id','family');b.need(len(rows)==32 and [{k:int(row[k]) for k in keys} for row in rows]==request['candidates'],'exact all32 requested IDs retain order and denominator')
 b.need(all(row['expected_physical_microsteps']=='12' and row['committed_physical_microsteps']=='0' and row['full_prefix_status']=='NOT_EVALUATED' and row['bank_accepted']=='False' for row in rows),'all32 same related states; no absent result promoted')
 b.need(draft['original8192']==dict(denominator=8192,model_NEG=8018,model_CLEAR=174,model_UNKNOWN=0,prospective32='EMPTY_ORIGINAL_QUOTA_SHORTFALL_PRESERVED') and draft['supplemental32768']==dict(denominator=32768,model_NEG=31986,model_CLEAR=782,model_UNKNOWN=0) and draft['cumulative40960']['denominator']==40960,'frozen separate model accounting; no numeric replay')
 evidence=dict(INPUTS);evidence.update(native_wait_v4=f['native_wait'],outer_wait_v4=f['outer_wait'],tool_v4=f['tool_closure'],request_v4=f['request'],parent_review_v4=f['review'],parent_receipt_v4=f['parent_receipt'],native_API_CPU_wait=f['CPU_check_wait'],metadata_review=c['review'],metadata_result=c['metadata_peer'],metadata_worker=c['actual_reviewer_worker_wait'],metadata_outer=c['actual_reviewer_outer_wait'],metadata_tool=c['actual_reviewer_tool_terminal'],diagnosis_worker=dc['actual_worker_wait'],diagnosis_tool=dc['actual_tool_wait'],producer_SOURCE_graph=draft['source_graph'])
 epoch=draft['snapshot_utc'];local=datetime.datetime.fromisoformat(epoch).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 x=dict(schema='astra.report.increment08.closed_failed_metadata_facts.v1',status='CLOSED_NATIVE_FAILURE_METADATA_BOUND_PENDING_DISTINCT_REPORT08_SCIENTIFIC_PEER',snapshot_utc=epoch,snapshot_Asia_Shanghai=local,historical_snapshot_not_live=True,metadata_verification_closed_utc=c['closed_utc'],metadata_verification_later_than_fact_cutoff_not_new_native_event=True,native_actual_exit=-11,native_raw_status=139,outer_actual_exit=1,tool_actual_exit=1,native_elapsed_seconds=nw['elapsed_seconds'],original_native_wall_cap_seconds=1200,native_timeout=False,resource_abort=None,guardian_error=None,guardian_signals=[],ENTRY='MISSING',PRODUCER='MISSING',registered_related_states=32,full_native_NOT_EVAL=32,committed_micro_ids=[0],committed_physical_advances=0,possible_uncommitted_physical_activity='UNKNOWN',complete_full12_prefixes=0,initial_camera_directories=24,complete_camera_group_receipts=0,all27_and_latest42_held='UNKNOWN_NO_COMPLETE_CAPTURE_RECEIPT',native_measurements_before_after='UNKNOWN_NO_ENTRY_OR_PRODUCER_AFTER_CLOSURE',camera_pixels_and_angles='NOT_REVIEWED',initial_row_numeric_results='NOT_INDEPENDENTLY_RECOMPUTED',crash_root_cause='UNKNOWN',periodic_python_stacks_not_fatal_or_causal=True,same32_related_not_new_IID=True,new_IID_states=0,original8192=draft['original8192'],fixed32768=draft['supplemental32768'],cumulative40960=draft['cumulative40960'],all32_parent_draft=draft['all32_rows'],evidence=evidence,complete_DATA_PASS=False,bank_count=0,safety_acceptance=False,task_acceptance=False,all_requirements_completed=False,formal_holdout_opened=False,CONTACT='UNKNOWN',actual_drive_peak='UNKNOWN',wrench_frame='UNKNOWN',global_error_bound='UNKNOWN',TERMINAL_SET='UNKNOWN',tasks='NOT_RUN',holdout='UNOPENED',science_figures=8,original_RGB=15,source_actions=dict(native_GPU=0,Git=0,RAW_numeric_decodes=0,RAW_camera_decodes=0,model_gap_arrays_read=0,new_figures=0),report08_scientific_peer=None)
 return x,ds,evidence,rows

def build(b):
 before=b.ref(__file__);x,ds,evidence,rows=facts(b);base=ds['public07'];oldplan=b.read(base['plan']);oldmanifest=b.read(base['manifest'])
 OUT.mkdir(parents=True,exist_ok=False);overlay=OUT/'overlay';overlay.mkdir()
 for name,q in evidence.items():copy_artifact(b,q,overlay/PREFIX/'evidence'/(name+'.json'))
 for n,q in [('index.html',base['HTML']),('panel_manifest.json',base['manifest']),('summary.json',oldplan['file_sources']['summary.json'])]:copy_artifact(b,q,overlay/PREFIX/'archive_candidate07'/n)
 copy_artifact(b,x['all32_parent_draft'],overlay/PREFIX/'ALL32_PARENT_DRAFT_ROWS_V1.csv')
 for name in ('build_v2.py','SOURCE_GRAPH_V2.json'):copy_artifact(b,b.ref(D/name),overlay/PREFIX/'source'/name)
 factref=b.write(overlay/PREFIX/'FACTS_INCREMENT08_V1.json',x)
 helper=module(H/'astra/report_increment07_v1/build_v3.py',BUILD07_SHA,'_immutable07_presentation');replace_id,table=helper.presentation_helpers(b)
 page=b.buffer(base['HTML'],4*1024**2).decode();epoch=x['snapshot_utc'];local=x['snapshot_Asia_Shanghai']
 header='<p class="asof" id="reportAsOf">报告08事实快照截至 <time datetime="'+epoch+'">2026-10-10 22:03:37.385121 UTC</time>（Asia/Shanghai：2026-10-11 06:03:37.385121）；不是实时状态。失败metadata核验于22:04:35 UTC闭合，只核验该快照内的事实，不增加新实验。报告07保留21:30:27.935059 UTC／2026-10-11 05:30:27.935059的历史快照；报告06截至19:54:39 UTC，05截至19:16:27 UTC，04截至17:53:40 UTC。</p>'
 page=replace_id(page,'p','reportAsOf',header)
 notice='<p id="native08Executive"><strong>完整安全验收尚未通过，BANK0。</strong>同32 v4 related retry真实SIGSEGV：native −11（raw139）／outer1／tool1；ENTRY与PRODUCER缺失，只有row000提交metadata、24个相机目录、0完整相机组收据。全部32项完整挑战未评估，原因UNKNOWN；没有独立RAW数值／像素通过。<a href="#nativeAttemptIncrement08">失败与证据缺口</a>；下方07修复就绪状态是原冻结时点的历史记录。</p>'
 b.need(page.count('<div class="executive">')==1,'single executive');page=page.replace('<div class="executive">','<div class="executive">'+notice,1)
 states=table('native08AttemptTable','同32 related retry；时间行、重试不增加新IID',['记录','native wait / ENTRY','outer / tool','提交步 / 相机','完整32挑战'],[['首次v3（07已保留）','0 / failed','1 / 1','0步 / 0组','32 NOT_EVAL'],['同32 v4实际失败','−11 / raw139；ENTRY MISSING','1 / 1','row000；提交物理步0；24目录 / 完整组0','32 NOT_EVAL']])
 gaps=table('native08GapsTable','闭合metadata核验与未完成的RAW／物理证据分别列出',['证据范围','已闭合事实','仍未测得'],[['进程与guardian','native−11/raw139，1086.5235147180501s < 原1200s；resource_abort / error为空，supervisor signals=[]','崩溃原因UNKNOWN；无fatal C++指令／stack'],['原生数值与时间','持久化micro ID只有[0]；完整12步前缀0','初始数值未独立复算；未提交物理活动UNKNOWN'],['原始相机与held状态','24个初始camera目录；完整group receipt0','像素／角度未审；all27/latest42 held UNKNOWN'],['测量与验收','ENTRY、PRODUCER缺失；32分母保留','before/after native测量UNKNOWN；完整DATA未通过；BANK0']])
 ids=table('native08All32Table','全部32预登记身份保留；以下为失败metadata标签，不是RAW数值重算',['seed','family','proposal','global ID','期望步','提交步','完整挑战','RAW / 相机','bank'],[[r['seed'],r['family'],r['proposal_id'],r['global_id'],12,0,'NOT_EVALUATED','未复算 / 未接受',0] for r in rows])
 section='<section id="nativeAttemptIncrement08"><h2>报告08：同32原生崩溃，完整挑战仍未评估</h2><p>事实截止：<strong>2026-10-10 22:03:37.385121 UTC</strong>／Asia/Shanghai 2026-10-11 06:03:37.385121。之后完成的独立失败metadata核验（22:04:35 UTC）只核验这些既存记录；没有追加后续native或profiling事件。</p>'+states
 section+='<p>v4真实native SIGSEGV −11／raw139；outer1、tool1已reaped。ENTRY和PRODUCER均缺失。1086.5235147180501秒小于原1200秒预算；guardian未记录resource_abort、error或supervisor signals。本次记录不是guardian超时终止，但不能据此推断崩溃原因或所有资源风险已排除，原因仍UNKNOWN。</p>'+gaps
 section+='<p><strong>24个目录不等于24个合格相机状态。</strong>只有初始row000持久化提交metadata，提交物理advance0，完整挑战12步前缀0；可能未提交的物理活动UNKNOWN。原始PNG／NPZ保留，但独立失败metadata审查没有解码像素、角度或原生数值；完整camera DATA、held27/latest42、参数before/after闭合不能补造。</p>'
 section+='<p>独立诊断只完成有界只读观察：周期性Python SyntheticData stack不是fault-time fatal stack，也未识别崩溃C++指令或因果机制。没有执行新原生复现，没有已证修复；本版不纳入未闭合的未来profiling、重试或物理结果。</p>'+ids
 section+='<p>仍保留原8192：8018 NEG／174预测CLEAR／0 UNKNOWN，原schedule EMPTY/HELD；新增32768：31986 NEG／782预测CLEAR／0 UNKNOWN，16×512及16×2048分母分开保留。累计40960仅MODEL_ONLY台账；模型预测CLEAR不是安全，same32相关重试不是新IID。完整requirements仍false，BANK0；Drive、CONTACT、wrench frame、terminal set、global error bound均UNKNOWN；tasks NOT_RUN、formal holdout UNOPENED。旧8图与15张原生RGB及全部失败不删改。</p><ul>'
 labels={'failed_v4':'父级真实v4失败闭环（−11／outer1／tool1）','native_wait_v4':'实际native wait4 −11／raw139','outer_wait_v4':'实际outer wait1','tool_v4':'实际tool1终态','failed_metadata_peer':'独立FAILED_ATTEMPT_METADATA_VERIFIED闭合','metadata_review':'完整metadata审查与明确缺口','metadata_result':'原metadata输出（非RAW数值／像素oracle）','diagnosis':'独立只读崩溃诊断：root cause UNKNOWN','diagnosis_closed':'诊断worker／tool0真实闭合','request_v4':'相同32身份、原12步／1200秒冻结请求','parent_review_v4':'父级原门槛review','producer_SOURCE_graph':'原冻结生产SOURCE图（metadata复用）'}
 for name,label in labels.items():section+='<li><a href="'+PREFIX+'/evidence/'+name+'.json" download>'+html.escape(label)+'</a></li>'
 section+='<li><a href="'+PREFIX+'/FACTS_INCREMENT08_V1.json" download>本版typed事实及全部exact来源</a></li><li><a href="'+PREFIX+'/ALL32_PARENT_DRAFT_ROWS_V1.csv" download>父级原32行metadata草稿CSV（原PENDING标签保留，非RAW验收）</a></li><li><a href="'+PREFIX+'/archive_candidate07/index.html">冻结已发布报告07 REPAIR1历史正文</a></li></ul></section>'
 needle='<section id="nativeAttemptIncrement07">';b.need(page.count(needle)==1,'unique frozen07 section');page=page.replace(needle,section+needle,1)
 needle='<a href="#nativeAttemptIncrement07">同32真实失败 / 修复就绪快照</a>';b.need(page.count(needle)==1,'existing07 TOC');page=page.replace(needle,'<a href="#nativeAttemptIncrement08">同32崩溃 / 32未评与证据缺口</a>'+needle,1)
 (overlay/'index.html').write_text(page)
 summary=b.read(oldplan['file_sources']['summary.json']);summary.update(asof_utc=epoch,asof_AsiaShanghai=local,increment08=x,historical_candidate07_summary=PREFIX+'/archive_candidate07/summary.json')
 b.need(datetime.datetime.fromisoformat(summary['asof_utc'])==datetime.datetime.fromisoformat(summary['asof_AsiaShanghai']),'same frozen UTC/local instant, no inherited timestamp');b.write(overlay/'summary.json',summary)
 sources={p.relative_to(overlay).as_posix():b.ref(p) for p in overlay.rglob('*') if p.is_file()};updated=copy.deepcopy(oldmanifest);updated['files'].update({n:q['sha256'] for n,q in sources.items()});updated.update(snapshot_updated_utc=epoch,increment08=dict(facts=PREFIX+'/FACTS_INCREMENT08_V1.json',scope='closed failed metadata only; no RAW acceptance',native_exit=-11,native_raw_status=139,outer_exit=1,tool_exit=1,full_native_NOT_EVAL=32,camera_complete_groups=0,bank_count=0,safety_acceptance=False,formal_holdout=False))
 b.write(overlay/'panel_manifest.json',updated);sources['panel_manifest.json']=b.ref(overlay/'panel_manifest.json');changed={n:q['sha256'] for n,q in sources.items()}
 b.need(all(updated['files'][n]==s for n,s in oldmanifest['files'].items() if n not in ('index.html','summary.json')),'all old assets/8fig15RGB/failure SHA preserved')
 plan=dict(schema='astra.report.increment08.overlay_plan.v1',base_commit=BASE,base_manifest=base['manifest'],before_sha256={n:base['manifest']['sha256'] if n=='panel_manifest.json' else oldmanifest['files'][n] for n in changed if n=='panel_manifest.json' or n in oldmanifest['files']},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=sources,changed_files=changed,file_bytes={n:Path(q['path']).stat().st_size for n,q in sources.items()},inherited_files={n:s for n,s in oldmanifest['files'].items() if n not in changed},manifest=b.ref(overlay/'panel_manifest.json'),old8fig_and15RGB_SHA_preserved=True,science_figures=8,original_RGB=15,new_science_figures=0,archive_decodes=0,RAW_numeric_camera_decodes=0,model_gap_arrays_read=False,source_facts=factref,scientific_peer='PENDING_DISTINCT_HOOKE_REPORT08_PEER',browser_publication='NOT_RUN')
 b.write(OUT/'REPORT_OVERLAY_PLAN_V1.json',plan);b.need(b.ref(__file__)==before,'author source unchanged')
 return b.write(OUT/'CANDIDATE08_BUILD_HANDOFF_V1.json',dict(status='BUILT_REPORT08_PENDING_TRUE_CPU_LOCAL4MODES_DISTINCT_HOOKE_PEER',candidate='candidate08',base_public_commit=BASE,plan=b.ref(OUT/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],HTML=b.ref(overlay/'index.html'),facts=factref,overlay=str(overlay),source_graph=b.ref(D/'SOURCE_GRAPH_V2.json'),actual07=INPUTS['public07'],new_paths=len(changed),delta_bytes=sum(plan['file_bytes'].values()),manifest_objects=len(updated['files'])+1,old8fig_and15RGB_SHA_preserved=True,source_before_after_unchanged=True,snapshot_utc=epoch,bank_count=0,safety_acceptance=False,task_acceptance=False,formal_holdout=False,source_native_GPU_Git_W_actions=0,model_numeric_replay_here=0,large_gap_arrays_read_here=0,RAW_decodes_here=0,new_science_figures=0,distinct_scientific_reviewer_requested='Hooke',report08_scientific_peer=None,publish_performed=False))

def main():
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--source-sha256',required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--child',action='store_true');a=ap.parse_args();b=module(H/'astra/report_increment05_v4/binding_v1.py',BIND_SHA,'_closed_report_binding')
 b.need(b.sha(__file__)==a.source_sha256 and os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in b.THREADS),'bound -B CPU thread1 source')
 if a.child:print(json.dumps(build(b)),flush=True);return 0
 argv=[PY,'-B',str(Path(__file__).resolve()),*os.sys.argv[1:]]
 if a.run:
  loader=module(H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py','8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512','_report08_loader');core=loader.load_reviewed_core('82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626')
  w=core.bounded_child(argv+['--child'],dict(os.environ),Z/'worker',backend='CPU_REPORT08_CLOSED_FAILED_METADATA_BUILD',max_wall=180,cpu_limit=True);b.closed_wait(b.ref(Z/'worker/ACTUAL_WAIT.json'));return w['actual_wait_exit']
 O.mkdir(exist_ok=True);Z.mkdir(exist_ok=False);observer=module(H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py','65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc','_report08_observer');observer.observe(argv+['--run'],Z/'external')
 worker=b.ref(Z/'worker/ACTUAL_WAIT.json');outer=b.ref(Z/'external/ACTUAL_WAIT.json');b.closed_wait(worker);b.closed_wait(outer);h=b.read(b.ref(OUT/'CANDIDATE08_BUILD_HANDOFF_V1.json'));h.update(status='CLOSED_REPORT08_CPU_BUILD_PENDING_LOCAL_QA_HOOKE_PEER',actual_worker_wait=worker,actual_wrapper_wait=outer,CPU_AS_bytes=1024**3,threads=1)
 q=b.write(Z/'CLOSED_BUILD_HANDOFF_V1.json',h);print(json.dumps(dict(handoff=q,worker_wait=worker,wrapper_wait=outer)),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
