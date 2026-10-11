"""REPORT12 partial-gallery overlay; terminal V8 evidence is required to build.

Only compact closed metadata and four sealed PNGs are read. No SDK, native, Git,
public-W writes, raw-matrix replay, image editing, or scientific self-approval.
"""
import argparse
import copy
import datetime as dt
import hashlib
import html
import json
import os
import re
import types
from pathlib import Path

H = Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D = Path(__file__).resolve().parent
O = R/'astra_report_increment12_v1'
PREFIX = 'partial_gallery_increment12_v1'
BASE_COMMIT = '403c3f4c5b9c172b056b1e9ab674fc8a2cd61388'
PY = '/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BPY = '/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/bin/python'
CORE_SHA = '82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626'
LOADER = H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py'
LOADER_SHA = '8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512'
OBSERVER = H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py'
OBSERVER_SHA = '65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc'
BIND = H/'astra/report_increment05_v4/binding_v1.py'
BIND_SHA = 'bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'

def rr(path, sha):
    return dict(path=str(path), sha256=sha)

STATIC = {
    'public11': rr(R/'astra_report_increment11_v1/publication_v3/CLOSED_PUBLICATION_HANDOFF_V11.json', 'c96b202a689e16c6d10f199d91ca348ad133855cd5417c6c3e67a2960cdc79de'),
    'candidate11': rr(H/'astra/report_increment11_v1/release2/CLOSED_CANDIDATE_HANDOFF_V11.json', 'ab1e64a4b160fab39a68bd10c9a5cafeab424446e0a6f0165c59f7ed27bc4e9a'),
    'gallery3': rr(R/'astra_supplement32_v7_partial_gallery_candidate_v1/PARENT_HANDOFF_V1.json', 'df66e2979a227f920532929b368114222649b5a74a9c9f21d023e418a39945c1'),
    'UR04': rr(R/'astra_supplement32_v7_partial_gallery_candidate_v1/actual03/PARENT_HANDOFF_UR04_V1.json', 'f2d3924e7cb70e6c1d383bbf85c3bd400923f518f2a52afb71b93810c3ea1724'),
    'taxonomy': rr(R/'astra_v7_ur04_otherhand_taxonomy_diagnosis_v1/actual01/PARENT_HANDOFF_V1.json', 'c5e0121b28a53d7d5b3d166fb3870f7c4e910f31bbb0f6f99eb37421b7c807d7'),
    'reader_v3': rr(R/'parent_supplement_native32_profile_intake_v1/CPU/parent_ssd_v7_reader03/PARENT_ACTUAL_READER_CLOSED_V1.json', '65de606e583d2c5a2046bf77e4dd9483e2ff4b52b8e6f200f70f73af6e01f856'),
    'V8_request': rr(R/'parent_supplement_native32_profile_intake_v1/campaign_ssd_v8/FROZEN_NATIVE_REQUEST_V1.json', '5a21d7f8ae88faa439b5ed1c0d30da3736b75588143e7d4ef36fc1f95636f258'),
    'V8_author_SOURCE': rr(H/'astra/full74_supplemental_mixed32_native_v8_ssd/release2/HANDOFF_V8_SSD.json', 'f1e6b3541ea237d344b0cc21bb3ccac00e1e710b0bd7879e013b5899cc08dbbe'),
    'V8_independent_SOURCE': rr(R/'astra_full74_supplemental_mixed32_native_v8_ssd_independent_review_v1/REVIEW_V1.json', 'b602faa3fc3798b2237d67e870a6485e716325bcac0e18f567539e4c2d2b6670'),
}

def authenticated_module(path, sha, name):
    with Path(path).open('rb') as f:
        a = os.fstat(f.fileno()); raw = f.read(2*1024**2+1); z = os.fstat(f.fileno())
    if len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('exact authenticated source buffer required')
    if (a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(z.st_dev,z.st_size,z.st_mtime_ns,z.st_ctime_ns):
        raise ValueError('changed source buffer')
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__)
    return m

def binding():
    return authenticated_module(BIND,BIND_SHA,'_report12_bound_metadata')

def exact_in_tree(tree, ref):
    return tree==ref or (isinstance(tree,dict) and any(exact_in_tree(v,ref) for v in tree.values())) or (isinstance(tree,list) and any(exact_in_tree(v,ref) for v in tree))

def closed_actual_wait(b, ref):
    w=b.read(ref)
    b.need(w.get('waitpid_observed') is True and type(w.get('child_pid')) is int and w['child_pid']==w.get('waited_pid'),'actual wait4 identity')
    b.need(type(w.get('raw_wait_status')) is int and type(w.get('actual_wait_exit')) is int,'typed actual terminal status')
    b.need(os.WIFEXITED(w['raw_wait_status']) or os.WIFSIGNALED(w['raw_wait_status']),'terminal wait4 required')
    b.need(os.waitstatus_to_exitcode(w['raw_wait_status'])==w['actual_wait_exit'],'raw wait status agrees')
    b.need(w.get('owned_identities') and all(b.fresh_absent(i) for i in w['owned_identities']),'all recorded owned identities fresh absent')
    b.need(b.sha(w['log'])==w['log_sha256'],'actual log SHA')
    return w

def closed_tool(b, ref):
    t=b.read(ref)
    b.need(t.get('session_closed') is True and type(t.get('actual_tool_exit')) is int,'exact closed parent tool schema')
    term=t.get('actual_terminal_result',t.get('observed_completion'))
    b.need(isinstance(term,dict) and type(term.get('exit_code')) is int and term['exit_code']==t['actual_tool_exit'],'real tool terminal result agrees')
    return t['actual_tool_exit']

def closed_input(b, h):
    for key in ('actual_worker_wait','actual_outer_wait'):
        b.closed_wait(h[key])
    b.need(closed_tool(b,h['actual_tool_wait'])==0,'real source/data producer CPU tool0')
    b.need(h.get('source_after_unchanged') is True and h.get('all_enrolled_owned_fresh_absent') is True,'closed source/data handoff')

def inputs(b):
    s={k:b.read(v) for k,v in STATIC.items()}
    b.need(s['public11']['status']=='CLOSED_ACTUAL_NORMAL_PUBLICATION' and s['public11']['commit']==BASE_COMMIT,'exact genuinely published11 base')
    b.need(s['public11']['candidate']==STATIC['candidate11'],'same public11 candidate')
    for key in ('gallery3','UR04','taxonomy'):
        closed_input(b,s[key])
    author=s['V8_author_SOURCE']; peer=s['V8_independent_SOURCE']
    b.need(author['actual_SOURCE_checks']==43 and peer['checks_count']==29 and peer['blocking_findings']==[] and peer['author_handoff']==STATIC['V8_author_SOURCE'],'actual engineering SOURCE43+29, not experiment samples')
    for key in ('actual_worker_wait','actual_outer_wait'):
        b.closed_wait(author[key]);b.closed_wait(peer[key])
    b.need(closed_tool(b,author['actual_tool_wait'])==closed_tool(b,peer['actual_tool_wait'])==0,'actual engineering source tools0')
    g=b.read(s['gallery3']['candidate']); notes=b.read(s['gallery3']['caption_notes'])
    u=b.read(s['UR04']['candidate']); uc=b.read(s['UR04']['caption']); tax=b.read(s['taxonomy']['diagnosis'])
    b.need(g['camera_group_receipts']==u['camera_group_receipts']==0 and g['qualification'] is False and u['qualification']=='NOT_QUALIFIED','partial gallery cannot qualify')
    b.need(g['selection_fixed_before_local_pixel_checks']==['wide_attempt_02','F_L_attempt_01','wide_attempt_01'],'fixed display input, no search/resampling')
    b.need(tax['producer_mask_bug_reproduced'] is False and tax['producer_mask_replay_exact'] is True,'actual taxonomy diagnosis scope')
    b.need(tax['mask_coordinate_agreement']['other_hand_intersection_pixels']==601 and tax['mask_coordinate_agreement']['target_intersection_pixels']==43323,'sealed bounded local pixel counts')
    b.need(s['taxonomy']['other_hand_owner']=='env0 F_L' and tax['predicate_comparison']['source_or_threshold_modified'] is False,'same-environment hand; no gate modification')
    reader=s['reader_v3']; result=b.read(reader['reader_result'])
    b.need(reader['actual_exits']==[2,2,2] and reader['actual_NATIVE_RAW_qualification'] is False and result['status']=='CLOSED_FAILED_NATIVE_RECEIPTS_ONLY','latest real reader is receipts-only')
    for key in ('actual_worker_wait','actual_outer_wait'):
        b.need(closed_actual_wait(b,reader[key])['actual_wait_exit']==2,'reader failure2 preserved')
    b.need(closed_tool(b,reader['actual_tool_wait'])==2,'reader actual tool2 retained')
    entries=[]
    for e,caption in zip(g['entries'],notes['captions']):
        png=e['display_png']; stem=Path(png['path']).stem
        b.need(stem==caption['stem'] and png['sha256']==e['original_png']['sha256'],'original PNG and exact sealed caption identity')
        entries.append(dict(stem=stem,image=png,caption_zh=caption['zh'],caption_en=caption['en'],producer_single_attempt_label=e['producer_single_attempt_status'],sidecars=dict(e['copy_sidecars'],native_paths=e['native_paths']),source_entry=e))
    entries.append(dict(stem='U_R_attempt_04',image=u['display_png'],caption_zh=uc['zh'],caption_en=uc['en'],producer_single_attempt_label=u['producer_single_attempt_status'],sidecars=dict(u['copy_sidecars'],native_paths=u['native_path_classification']),source_entry=u))
    for e in entries:
        raw=b.buffer(e['image'],4*1024**2)
        b.need(raw[:8]==b'\x89PNG\r\n\x1a\n' and int.from_bytes(raw[16:20],'big')==1280 and int.from_bytes(raw[20:24],'big')==720,'original uncropped1280x720 PNG bytes')
        b.need('PARTIAL_INITIAL_ONLY' in e['caption_en'] and 'NOT_QUALIFIED' in e['caption_en'],'sealed caption nonqualification')
    return s,dict(entries=entries,raw_gallery=g,UR04=u,taxonomy=tax,reader=result)

COVERAGE_KEYS=('committed_micro_ids','physical_prefix_advance_count','allocated_initial_env_dirs','partial_PNG_count','completed_camera_groups','ENTRY','PRODUCER','ProfileClose','held27','independent_RAW_scope')

def terminal_inputs(b, packet_ref):
    p=b.read(packet_ref)
    b.need(p.get('schema')=='astra.report12.closed_terminal_input.v1' and p.get('status')=='PARENT_SUPPLIED_CLOSED_FACTS','real closed V8 packet required before writes')
    b.need(p.get('bank_count')==0 and p.get('safety_acceptance') is False and p.get('holdout_opened') is False,'no report-driven promotion')
    epoch=dt.datetime.fromisoformat(p['snapshot_utc']); b.need(epoch.utcoffset()==dt.timedelta(0),'explicit frozen UTC')
    parent=b.read(p['parent_closed'])
    b.need(parent.get('status')=='CLOSED_ACTUAL_NATIVE_FAILED_OR_INCOMPLETE_UNQUALIFIED' and dt.datetime.fromisoformat(parent['utc'])<=epoch,'exact V8 parent utc closure before cutoff')
    b.need(exact_in_tree(parent,STATIC['V8_request']),'parent binds exact V8 request')
    b.need(p['native_wait']['path']==str(R/'astra_full74_supplemental_mixed32_native_v8_ssd/native_attempt1/native_process/ACTUAL_WAIT.json'),'exact actual V8 native root')
    for key in ('native_wait','outer_wait','tool_closure'):
        b.need(exact_in_tree(parent,p[key]),'parent binds exact actual '+key)
    nw=closed_actual_wait(b,p['native_wait']); ow=closed_actual_wait(b,p['outer_wait']); te=closed_tool(b,p['tool_closure'])
    b.need(set(p['coverage_projection'])==set(COVERAGE_KEYS),'complete literal metadata field adapter')
    coverage={}
    for key,projection in p['coverage_projection'].items():
        if projection is None:
            coverage[key]='UNKNOWN_MISSING_BOUND_FIELD'; continue
        b.need(set(projection)=={'record','keys'} and projection['record']=='parent' and isinstance(projection['keys'],list) and projection['keys'],'explicit actual parent field projection')
        coverage[key]=b.get(parent,projection['keys'])
    b.need(parent.get('bank_count')==0 and parent.get('safety_acceptance') is False and parent.get('task_acceptance') is False and parent.get('formal_holdout_opened') is False,'actual V8 closure forbids promotion')
    b.need((parent['native_actual_exit'],parent['raw_native_wait_status'],parent['outer_actual_exit'],parent['actual_parent_tool_exit'])==(nw['actual_wait_exit'],nw['raw_wait_status'],ow['actual_wait_exit'],te),'parent and actual terminal waits agree')
    b.closed_wait(parent['CPU_check_wait'])
    b.need(parent['registered_owned_count']==len(parent['fresh_owned_absence'])==13 and parent['all_recorded_owned_fresh_absent'] is True,'actual V8 thirteen owned identities')
    for i in parent['fresh_owned_absence']:
        b.need(i['fresh_same_identity_live'] is False and b.fresh_absent(dict(pid=i['pid'],startticks=i['startticks'],live=False)),'fresh exact V8 PID/starttick absence')
    return dict(snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=epoch.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),native_exit=nw['actual_wait_exit'],native_raw_status=nw['raw_wait_status'],outer_exit=ow['actual_wait_exit'],tool_exit=te,native_error=nw.get('error'),native_resource_abort=nw.get('resource_abort'),native_signals=nw.get('signals'),native_elapsed_seconds=nw['elapsed_seconds'],native_work_deadline_seconds=parent['native_work_deadline_seconds'],registered_fresh_absent_count=13,CPU_check_exit=parent['CPU_check_actual_exit'],failure_cause='UNKNOWN',timeout_not_crash_fix=True,last_partial_location_parent_diagnostic=parent['failure_location_live_partial'],coverage=coverage,packet_ref=packet_ref,bound_terminal_refs={k:p[k] for k in ('parent_closed','native_wait','outer_wait','tool_closure')},same32_related_new_IID=0,bank_count=0,all_requirements_completed=False,safety_acceptance=False,tasks='NOT_RUN',holdout='UNOPENED',Drive='UNKNOWN',wrenchframe='UNKNOWN',global_error_bound='UNKNOWN',historical_snapshot_not_live=True)

def bi(zh,en):
    return '<span class="r12-zh" lang="zh">'+html.escape(zh)+'</span><span class="r12-en" lang="en">'+html.escape(en)+'</span>'

def escaped(value):
    return html.escape(json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value))

def render(data,outcome=None):
    title=bi('报告12：部分原生画面与证据边界','REPORT12: partial native views and evidence boundaries')
    s='<section id="nativeAttemptIncrement12"><h2>'+title+'</h2><fieldset class="r12-controls"><legend>阅读模式 / Reading controls</legend><label><input type="radio" id="r12LanguageZH" name="r12lang" checked>中文</label><label><input type="radio" id="r12LanguageEN" name="r12lang">English</label><label><input type="radio" id="r12Detailed" name="r12detail" checked>详细 / Detailed</label><label><input type="radio" id="r12Summary" name="r12detail">摘要 / Summary</label></fieldset><div class="r12-content">'
    s+='<p class="r12-result">'+bi('完整要求仍未完成；BANK0。以下四图来自失败的V7初态，均未资格化，0完整相机组。','Full requirements remain incomplete; BANK0. These four views come from failed V7 initial capture; none is qualified and there are0 complete camera groups.')+'</p>'
    if outcome is None:
        s+='<p id="report12DraftBoundary">'+bi('SOURCE/UI准备稿；V8真实终态尚未绑定，不是可发布的科学候选。','SOURCE/UI preparation only; actual V8 terminal facts are not bound. This is not a publishable scientific candidate.')+'</p>'
    else:
        s+='<p>'+bi('科学截止快照（不是实时）','Frozen scientific cutoff, not live status')+'：<time datetime="'+escaped(outcome['snapshot_utc'])+'">'+escaped(outcome['snapshot_utc'])+'</time> / Asia/Shanghai '+escaped(outcome['snapshot_Asia_Shanghai'])+'</p>'
        s+='<p id="native12V8Outcome">V8 native / raw / outer / tool: '+escaped([outcome['native_exit'],outcome['native_raw_status'],outcome['outer_exit'],outcome['tool_exit']])+'. '+bi('这是独立列出的后续相关重试终态；下方图像仍属于历史V7。','This is the separately listed later related retry; the images below remain historical V7 views.')+'</p>'
    if outcome is not None:
        s+='<p id="r12Deadline">'+bi('原native工作期限1200秒；TERM15后原grace再KILL9；elapsed包含原grace。原因UNKNOWN，超时不证明V4崩溃已修复；SOURCE工程测试不代替原生验收。','Original native deadline1200s;TERM15 followed by original grace andKILL9;elapsed includes grace. Cause UNKNOWN;timeout does not prove the V4 crash was fixed. SOURCE engineering tests do not replace native acceptance.')+' elapsed '+escaped(outcome['native_elapsed_seconds'])+'s.</p>'
        s+='<p id="r12IncompleteProducts">'+bi('CPUcheck0；13个登记owned身份现已不存在。仅env000的24张partial PNG、micro[0]、0相机组；ENTRY / PRODUCER / ProfileClose均缺失。物理prefix推进数UNKNOWN_PRODUCER_ABSENT。这些目录、PNG和初态producer记录不能补足32×13或12步验收。','CPUcheck0;13 enrolled owned identities are absent. Only env000 with24 partial PNGs,micro[0],0 camera groups;ENTRY / PRODUCER / ProfileClose are missing. Physical prefix advances remain UNKNOWN_PRODUCER_ABSENT. Directories,PNGs and producer initial records cannot fill32×13 or12-step acceptance.')+'</p>'
    s+='<div class="r12-gallery">'
    for e in data['entries']:
        stem=e['stem'];name=PREFIX+'/assets/'+stem+'.png'
        s+='<figure id="r12_'+stem+'"><a href="'+name+'" download><img src="'+name+'" width="1280" height="720" loading="lazy" alt="'+html.escape(stem+' — PARTIAL_INITIAL_ONLY / NOT_QUALIFIED')+'"></a><figcaption>'+bi(e['caption_zh'],e['caption_en'])+'</figcaption></figure>'
    s+='</div><div class="r12-detail">'
    s+='<p id="r12Taxonomy">'+bi('UR04局部两流各识别目标U_R手43323像素、同环境F_L手601像素；两流是同一画面，不能相加或算作独立样本。原predicate仅拒其他环境的手；NoHide禁止可见性修改，并不要求其它手不可见。没有复现mask bug，也不新增同环境其它手的资格门或unsafe判定。','UR04 local instance and semantic streams each identify43323 target U_R pixels and601 same-environment F_L pixels. They describe the same image; counts are not additive or independent samples. The original predicate rejects hands from other environments. NoHide forbids visibility mutation; it does not require other hands to be absent. No mask bug was reproduced; no extra same-environment-hand qualification gate or unsafe label is introduced.')+'</p>'
    s+='<p>'+bi('UR04部分非手semantic原生身份仍UNKNOWN（66647像素）；不能据此断言未标注手或clean。wide01的env5=47867、env10=185是局部instance全场景范围，不能与早期仅机器人计数直接比较。','UR04 non-hand semantic native identity remains partly UNKNOWN(66647 pixels); this does not establish an unlabelled hand or a clean view. wide01 env5=47867 and env10=185 are local instance counts across the whole scene; they cannot be compared directly with earlier robot-only counts.')+'</p>'
    s+='<p>'+bi('V8作者43项与独立29项SOURCE检查真实闭合；这些是工程检查，不是原生实验样本或camera/native验收。','V8 author43 and independent29 SOURCE checks genuinely closed. These are engineering checks, not native experiment samples or camera/native acceptance.')+'</p>'
    s+='<div class="table-window" tabindex="0"><table id="native12EvidenceTable"><caption>'+bi('证据范围保持独立','Evidence scopes remain distinct')+'</caption><thead><tr><th>'+bi('项目','Item')+'</th><th>'+bi('真实范围与缺口','Actual scope and gap')+'</th></tr></thead><tbody>'
    rows=[('V7 reader v3','receipts-only; actual worker/outer/tool2/2/2; NO_DURABLE_PRODUCER_RECEIPT; not RAW numerical or camera PASS'),('Camera state','original27 MANDATORY; latest42 NEW_PROPOSED_NOT_ADOPTED; historical catalogue NOT_BOUND;42 is not an explicit user acceptance requirement'),('Strong camera','original thresholds unchanged;32 initial +32 final +all alarm groups across12 microsteps;4 hand groups ×3 views +3 wide views/group; partial PNGs cannot fill absent groups'),('MODEL_ONLY','8192(8018NEG/174predictedCLEAR)+32768(31986NEG/782predictedCLEAR)=40960; original8192 EMPTY retained; predictedCLEAR is not safety; same32 retries newIID0'),('Physical acceptance','Drive / CONTACT / wrenchframe / terminal set / global error bound UNKNOWN; tasks NOT_RUN; holdout UNOPENED; fullrequirementsfalse / BANK0')]
    for title,value in rows:s+='<tr><th scope="row">'+html.escape(title)+'</th><td>'+html.escape(value)+'</td></tr>'
    s+='</tbody></table></div>'
    if outcome is not None:s+='<pre id="r12V8Coverage">'+escaped(outcome['coverage'])+'</pre>'
    s+='<ul>'+''.join('<li><a href="'+PREFIX+'/evidence/'+k+'.json" download>'+html.escape(k)+'</a></li>' for k in STATIC)
    s+='<li><a href="'+PREFIX+'/FACTS_INCREMENT12_V1.json" download>Frozen typed facts / 冻结事实</a></li><li><a href="'+PREFIX+'/archive_candidate11/index.html" download>Frozen11 HTML / 历史原字节</a></li></ul></div></div></section>'
    return s

STYLE='''#nativeAttemptIncrement12 .r12-controls{display:flex;flex-wrap:wrap;gap:.8rem}#nativeAttemptIncrement12 .r12-controls label{white-space:nowrap}#nativeAttemptIncrement12 .r12-content{overflow-wrap:anywhere}#nativeAttemptIncrement12 .r12-en{display:none}#nativeAttemptIncrement12:has(#r12LanguageEN:checked) .r12-zh{display:none}#nativeAttemptIncrement12:has(#r12LanguageEN:checked) .r12-en{display:inline}#nativeAttemptIncrement12:has(#r12Summary:checked) .r12-detail{display:none}#nativeAttemptIncrement12 .r12-gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}#nativeAttemptIncrement12 .r12-gallery figure{margin:0;min-width:0}#nativeAttemptIncrement12 .r12-gallery img{display:block;width:100%;height:auto;object-fit:contain}#nativeAttemptIncrement12 .r12-gallery figcaption{font-size:.92rem;line-height:1.55;margin-top:.5rem}#nativeAttemptIncrement12 pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:700px){#nativeAttemptIncrement12 .r12-gallery{grid-template-columns:minmax(0,1fr)}}'''

def build(b,packet_ref,out):
    outcome=terminal_inputs(b,packet_ref)  # before any mkdir; no live outcome adoption
    s,data=inputs(b);base=s['candidate11'];plan0=b.read(base['plan']);manifest0=b.read(base['manifest'])
    graph=b.read(b.ref(D/'SOURCE_GRAPH_V1.json'))
    before={k:b.ref(v['path']) for k,v in graph['own_source_files'].items()}
    b.need(all(before[k]==q for k,q in graph['own_source_files'].items()),'exact frozen report12 source')
    out=Path(out);b.need(out.is_absolute() and not out.exists() and out.resolve().is_relative_to(O),'exclusive own candidate output')
    out.mkdir(parents=True,exist_ok=False);overlay=out/'overlay';overlay.mkdir()
    def copy_file(ref,dest):
        raw=b.buffer(ref,4*1024**2);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    for key,ref in STATIC.items():copy_file(ref,overlay/PREFIX/'evidence'/(key+'.json'))
    for key,ref in outcome['bound_terminal_refs'].items():copy_file(ref,overlay/PREFIX/'evidence'/('V8_'+key+'.json'))
    copy_file(b.read(STATIC['V8_FAILED_METADATA'])['review'],overlay/PREFIX/'evidence/V8_FAILED_METADATA_review.json')
    for name,ref in [('index.html',base['HTML']),('panel_manifest.json',base['manifest']),('summary.json',plan0['file_sources']['summary.json'])]:copy_file(ref,overlay/PREFIX/'archive_candidate11'/name)
    for name,ref in graph['own_source_files'].items():copy_file(ref,overlay/PREFIX/'source'/name)
    copy_file(b.ref(D/'SOURCE_GRAPH_V1.json'),overlay/PREFIX/'source/SOURCE_GRAPH_V1.json')
    for key,handoff_key,field in [('gallery_entries','gallery3','candidate'),('gallery_captions','gallery3','caption_notes'),('UR04_entry','UR04','candidate'),('UR04_caption','UR04','caption'),('taxonomy_diagnosis','taxonomy','diagnosis'),('reader_receipts_only','reader_v3','reader_result')]:copy_file(s[handoff_key][field],overlay/PREFIX/'evidence'/(key+'.json'))
    for e in data['entries']:
        copy_file(e['image'],overlay/PREFIX/'assets'/(e['stem']+'.png'))
        for key,ref in e['sidecars'].items():copy_file(ref,overlay/PREFIX/'assets'/(e['stem']+'.'+key+'.json'))
    outcome.update(gallery=dict(stage='PARTIAL_INITIAL_ONLY',qualification='NOT_QUALIFIED',image_count=4,complete_groups=0,original1280x720=True,edited=False,cropped=False,source_handoffs={k:STATIC[k] for k in ('gallery3','UR04','taxonomy')},entries=[dict(stem=e['stem'],original_image=e['image'],caption_zh=e['caption_zh'],caption_en=e['caption_en'],producer_single_attempt_label=e['producer_single_attempt_label']) for e in data['entries']]),taxonomy=dict(other_hand_owner='env0 F_L',same_env_other_hand_pixels=601,target_U_R_pixels=43323,semantic_nonhand_native_identity_UNKNOWN_pixels=66647,producer_mask_bug_reproduced=False,original_foreign_env_gate_unchanged=True,NoHide='NO_VISIBILITY_MUTATION',other_hand_unsafe_or_extra_gate=False),reader_v3=dict(actual_exits=[2,2,2],scope='CLOSED_FAILED_NATIVE_RECEIPTS_ONLY',actual_RAW_qualification=False,ref=STATIC['reader_v3']),held_state_scope=dict(original27='MANDATORY',latest42='NEW_PROPOSED_NOT_ADOPTED',historical_catalogue='NOT_BOUND',explicit_user42_acceptance_requirement=False),V8_SOURCE_engineering_only=dict(author_checks=43,independent_checks=29,native_acceptance=False,refs={k:STATIC[k] for k in ('V8_author_SOURCE','V8_independent_SOURCE')}),model_denominators=dict(original8192=dict(n=8192,NEG=8018,predicted_CLEAR=174,UNKNOWN=0,schedule='EMPTY'),supplement32768=dict(n=32768,NEG=31986,predicted_CLEAR=782,UNKNOWN=0),cumulative_MODEL_ONLY=40960),static_refs=STATIC,old_figures=8,original_RGB=15,new_partial_original_RGB=4,new_science_figures=0,RAW_math_recomputed=False)
    factref=b.write(overlay/PREFIX/'FACTS_INCREMENT12_V1.json',outcome)
    helper=authenticated_module(H/'astra/report_increment07_v1/build_v3.py','dce7ccdfc222a27053194b6db2f9fb4d2b8adfed6682ac1ee50eaf3fe0274026','_report12_unchanged_markup')
    replace_id,_=helper.presentation_helpers(b)
    page=b.buffer(base['HTML'],4*1024**2).decode();epoch=outcome['snapshot_utc'];local=outcome['snapshot_Asia_Shanghai']
    header='<p class="asof" id="reportAsOf">报告12冻结事实快照 / Frozen snapshot <time datetime="'+escaped(epoch)+'">'+escaped(epoch)+' UTC</time>；Asia/Shanghai '+escaped(local)+'。不是实时状态；06–11各节与原字节archive保留各自截止时点。</p>'
    page=replace_id(page,'p','reportAsOf',header)
    b.need(page.count('<section id="nativeAttemptIncrement11">')==1 and page.count('</head>')==1,'one preserved11 section/head')
    page=page.replace('</head>','<style id="report12ModesV1">'+STYLE+'</style></head>',1)
    page=page.replace('<section id="nativeAttemptIncrement11">',render(data,outcome)+'<section id="nativeAttemptIncrement11">',1)
    b.need(page.count('<div class="executive">')==1,'one inherited executive')
    notice='<p id="native12Executive"><strong>'+bi('完整验收仍未完成，BANK0。新增4张部分原生图，0相机组。','Full acceptance remains incomplete,BANK0. Four added partial native views;0 camera groups.')+'</strong> <a href="#nativeAttemptIncrement12">REPORT12 / PARTIAL_INITIAL_ONLY</a></p>'
    page=page.replace('<div class="executive">','<div class="executive">'+notice,1)
    needle='<a href="#nativeAttemptIncrement11">';b.need(page.count(needle)>=1,'historical11 TOC exists')
    # TOC only; the already frozen executive11 link remains byte-identical.
    toc='<a href="#nativeAttemptIncrement11">最新终态与验收缺口 / Terminal outcome and coverage</a>'
    b.need(page.count(toc)==1,'exact inherited11 TOC')
    page=page.replace(toc,'<a href="#nativeAttemptIncrement12">部分原生画面 / Partial native gallery</a>'+toc,1)
    with (overlay/'index.html').open('x') as f:f.write(page)
    summary=b.read(plan0['file_sources']['summary.json']);summary.update(asof_utc=epoch,asof_AsiaShanghai=local,increment12=outcome,historical_candidate11_summary=PREFIX+'/archive_candidate11/summary.json');b.write(overlay/'summary.json',summary)
    files={p.relative_to(overlay).as_posix():b.ref(p) for p in overlay.rglob('*') if p.is_file()}
    manifest=copy.deepcopy(manifest0);manifest['files'].update({k:q['sha256'] for k,q in files.items()});manifest.update(snapshot_updated_utc=epoch,increment12=dict(facts=PREFIX+'/FACTS_INCREMENT12_V1.json',bank_count=0,safety_acceptance=False))
    b.write(overlay/'panel_manifest.json',manifest);files['panel_manifest.json']=b.ref(overlay/'panel_manifest.json')
    b.need(all(manifest['files'][n]==sha for n,sha in manifest0['files'].items() if n not in ('index.html','summary.json')),'all old archives/8fig15RGB metadata SHA unchanged')
    plan=dict(schema='astra.report.increment12.overlay_plan.v1',base_commit=BASE_COMMIT,base_plan=base['plan'],base_manifest=base['manifest'],before_sha256={n:(base['manifest']['sha256'] if n=='panel_manifest.json' else manifest0['files'][n]) for n in files if n=='panel_manifest.json' or n in manifest0['files']},file_bytes={n:Path(q['path']).stat().st_size for n,q in files.items()},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},inherited_files={n:sha for n,sha in manifest0['files'].items() if n not in files},manifest=b.ref(overlay/'panel_manifest.json'),source_facts=factref,source_graph=b.ref(D/'SOURCE_GRAPH_V1.json'),old8fig_and15RGB_SHA_preserved=True,new_partial_original_RGB=4,new_science_figures=0,raw_tree_copy=False,publish_performed=False,scientific_peer='PENDING_DISTINCT_REPORT12_PEER')
    b.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    b.need(before=={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()},'build source unchanged')
    return b.write(out/'CANDIDATE_BUILD_HANDOFF_V1.json',dict(status='BUILT_LOCAL_CANDIDATE_PENDING_LOCAL4MODES_AND_DISTINCT_SCIENCE_PEER',HTML=b.ref(overlay/'index.html'),plan=b.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],facts=factref,snapshot_utc=epoch,snapshot_AsiaShanghai=local,source_before_after_unchanged=True,base_public_commit=BASE_COMMIT,publish_performed=False,bank_count=0,safety_acceptance=False))

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--packet',required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(build(binding(),rr(a.packet,a.packet_sha256),a.out),ensure_ascii=False))



# release3 changes only the metadata evidence/cutoff adapter. Original rendering,
# four-image bytes, original criteria and all model denominators stay unchanged.
STATIC['V8_FAILED_METADATA']=rr(R/'astra_supplement32_v8_timeout_failed_metadata_peer_v1/actual01/PARENT_HANDOFF_V1.json','742c62c087dddea4f216463e55886503db2641bb23f5a83b8e3512302813259f')

def metadata_scope(b,h,m,packet,epoch):
    b.need(h['schema']=='astra.V8.timeout_failed_metadata.closed_handoff.v1' and h['status']=='FAILED_ATTEMPT_METADATA_VERIFIED' and h['verdict']=='PASS_METADATA_SCOPE_ONLY' and h['blocking_findings']==[],'exact closed metadata handoff scope')
    b.need(h['parent_closure']==m['parent_closure']==packet['parent_closed'] and m['request']==STATIC['V8_request'],'same V8 request and actual parent closure')
    b.need(dt.datetime.fromisoformat(h['closed_utc'])<=epoch,'metadata closure not after frozen cutoff')
    b.need(m['scope']=='WAIT_SOURCE_JOURNAL_MISSING_RECEIPT_ENV000_FILENAME_METADATA_ONLY' and m['status']=='FAILED_ATTEMPT_METADATA_VERIFIED' and m['verdict']=='PASS_METADATA_SCOPE_ONLY' and m['blocking_findings']==[],'metadata is not pixel or math replay')
    b.need((h['native_actual_exit'],h['CPU_check_actual_exit'],h['parent_outer_actual_exit'],h['parent_tool_actual_exit'])==(-9,0,1,1),'actual failed native and parent exits retained')
    b.need((m['native_actual_exit'],m['native_raw_wait_status'],m['CPU_check_actual_exit'],m['outer_actual_exit'],m['parent_tool_actual_exit'])==(-9,9,0,1,1),'independent failure statuses agree')
    b.need(h['qualification'] is False and h['full_RAW_DATA_PASS'] is False and h['bank_count']==0 and h['related_same32_not_new_IID'] is True,'handoff cannot promote failed native')
    b.need(m['complete_full12_prefixes']==m['camera_state_receipts']==m['camera_group_receipts']==0 and m['all32_fullprefix_NOT_EVALUATED']==32 and m['partial_initial_PNG_count']==24,'actual zero complete/qualified groups and32 unresolved')
    b.need(m['committed_micro_ids']==[0] and m['missing_committed_micro_ids']==list(range(1,13)) and m['actual_uncommitted_physical_advances']=='UNKNOWN_PRODUCER_ABSENT','journal micro0 is not full physical prefix')
    b.need(m['ENTRY'] is None and m['producer'] is None and m['ProfileClose'] is None,'missing products remain absent')
    b.need(m['qualification'] is False and m['full_RAW_DATA_PASS'] is False and m['scientific_safety_PASS'] is False and m['all_requirements_satisfied'] is False and m['bank_count']==0 and m['formal_holdout_opened'] is False,'no safety promotion')
    b.need(m['raw_math_recomputed'] is False and m['camera_pixels_decoded']==0,'metadata-only independent scope')
    b.need(m['native_wait']==packet['native_wait'] and m['outer_wait']==packet['outer_wait'] and m['parent_tool_closure']==packet['tool_closure'],'actual native/outer/tool evidence chain')
    return dict(handoff=STATIC['V8_FAILED_METADATA'],review=h['review'],verdict='PASS_METADATA_SCOPE_ONLY',scope=m['scope'],native_exit=-9,native_raw_status=9,CPU_check_exit=0,parent_outer_exit=1,parent_tool_exit=1,committed_micro_ids=[0],missing_committed_micro_ids=list(range(1,13)),partial_PNG_count=24,PNG_count_scope=m['PNG_count_source'],complete_full12_prefixes=0,all32_fullprefix_NOT_EVALUATED=32,complete_camera_groups=0,qualification=False,full_RAW_DATA_PASS=False,raw_math_recomputed=False,camera_pixels_decoded=0,physical_prefix='UNKNOWN_PRODUCER_ABSENT',held27=m['held27'],journal_IO_ns=m['journal_IO_ns'],bank_count=0,closed_utc=h['closed_utc'])

_original_terminal_inputs=terminal_inputs
def terminal_inputs(b,packet_ref):
    outcome=_original_terminal_inputs(b,packet_ref);packet=b.read(packet_ref)
    b.need(packet['metadata_peer']==STATIC['V8_FAILED_METADATA'],'exact actual metadata input before writes')
    h=b.read(STATIC['V8_FAILED_METADATA']);m=b.read(h['review'])
    meta=metadata_scope(b,h,m,packet,dt.datetime.fromisoformat(outcome['snapshot_utc']))
    # live=False is adapted from the explicit closed-handoff absence field;
    # actual wait status is always read separately from original wait receipts.
    for key in ('actual_worker_wait','actual_outer_wait'):b.closed_wait(h[key])
    b.need(closed_tool(b,h['actual_tool_wait'])==closed_tool(b,h['actual_closer_tool'])==0,'real reviewer worker/outer/tool/closer0')
    b.need(h['source_after_unchanged'] is True and h['all_reviewer_enrolled_owned_fresh_absent'] is True and h['all13_native_enrolled_owned_fresh_absent'] is True,'genuinely closed reviewer and native identities')
    ids=[*h['enrolled_identities'],h['closer_identity']]
    b.need(len(ids)==h['reviewer_enrolled_identity_count']==4 and all(b.fresh_absent(dict(i,live=False)) for i in ids),'four reviewer identities fresh absent')
    outcome.update(V8_independent_FAILED_METADATA=meta,prior_frozen_candidate=packet['prior_frozen_candidate'],excluded_after_cutoff='V7 SSD-to-NAS mirror; pending V8 reader; future native or camera changes')
    return outcome

_original_render=render
def render(data,outcome=None):
    markup=_original_render(data,outcome)
    if outcome is not None:
        p='<p id="r12FailedMetadataPeer">'+bi('V8独立FAILED_METADATA审计已真实闭合，PASS_METADATA_SCOPE_ONLY：核验终态、SOURCE、journal及缺失收据与env000文件名。仅micro[0]、24个PNG文件名；完整12步prefix为0、完整相机组为0，32候选全NOT_EVALUATED。没有解码相机像素、NPY尾部或重算RAW数学；这不是camera/native/物理PASS。','The V8 independent FAILED_METADATA audit genuinely closed: PASS_METADATA_SCOPE_ONLY for waits, SOURCE, journal, missing receipts and env000 filenames. Only micro[0] and24 PNG filenames;0 complete12-step prefixes,0 complete camera groups, all32 candidates NOT_EVALUATED. No camera pixels or NPY tails were decoded and no RAW math was recomputed. This is not camera, native or physical acceptance.')+' <a href="'+PREFIX+'/evidence/V8_FAILED_METADATA_review.json" download>FAILED_METADATA review</a></p>'
        b=binding();b.need(markup.count('<div class="r12-gallery">')==1,'one unchanged gallery insertion point')
        markup=markup.replace('<div class="r12-gallery">',p+'<div class="r12-gallery">',1)
    return markup

if __name__=='__main__':main()
