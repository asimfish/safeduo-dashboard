"""Narrow REPORT11 overlay. Actual terminal inputs are mandatory before writes.

This module neither launches native/GPU nor freezes parent metadata or publishes.
SOURCE preparation tests render only a labelled fragment, never native results.
"""
import argparse
import copy
import datetime as dt
import hashlib
import html
import json
import os
import types
from pathlib import Path

H = Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D = Path(__file__).resolve().parent
O = R/'astra_report_increment11_v1'
PREFIX = 'native_attempt_increment11_v1'
BASE_COMMIT = 'f8b17692155dfa58afc6a14ceb46d9db159e2255'
BIND = H/'astra/report_increment05_v4/binding_v1.py'
BIND_SHA = 'bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'
CORE_SHA = '82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626'
LOADER = H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py'
LOADER_SHA = '8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512'
OBSERVER = H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py'
OBSERVER_SHA = '65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc'
PY = '/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BPY = '/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/bin/python'

def rr(path, sha):
    return dict(path=str(path), sha256=sha)

STATIC = {
    'public10': rr(R/'astra_report_increment10_v1/publication_v2/closeout01/PARENT_ACTUAL_PUBLICATION_CLOSED_V1.json', 'd85b0a044ce8d167f5849a93ed33ab944434f2075058b616ded778678c43e454'),
    'public10_handoff': rr(R/'astra_report_increment10_v1/publication_v2/closeout01/CLOSED_PUBLICATION_HANDOFF_V1.json', 'e41267814b1587362c4cc4b198b91a75ed501b73937b8a1b0b9932e46980a730'),
    'host_closed': rr(R/'astra_full74_native_v5_closed_host_profile_independent_analysis_v1/actual3/FINAL_HOST_ANALYSIS_HANDOFF_V1.json', '491e54dddc484f31f0cf1050031c67a7b70cd240fc0e30c9d8c2bed7a3819d53'),
    'host_compact': rr(R/'astra_full74_native_v5_closed_host_profile_independent_analysis_v1/actual3/COMPACT_FINDINGS_V1.json', '3ee7933e0adb02e195d5a9d23384b4805d8c8e764d97a99beaec709634663140'),
    'IO_closed': rr(R/'camera_snapshot_storage_benchmark_v1/io_cpu_v1/release3/CLOSED_ACTUAL_IO_CPU_HANDOFF_V1.json', 'c342ab86ac9898d7681a8462888b6ef0578cbe0442d32a8dd6b2cb924b4f915a'),
    'IO_readout': rr(R/'camera_snapshot_storage_benchmark_v1/io_cpu_v1/release3/IO_READOUT_V1.json', '2a534717b382762eebe2ef7fe49d0740ada50b9abdb4c413537ef38f0998d667'),
    'fsync_closed': rr(R/'camera_snapshot_storage_benchmark_v1/journal_fsync_cpu_v1/CLOSED_ACTUAL_FSYNC_CPU_HANDOFF_V1.json', '69145da578094e22c46a6bf0461313a1da9d9221bd44ca973e15f470a039a9e5'),
    'fsync_readout': rr(R/'camera_snapshot_storage_benchmark_v1/journal_fsync_cpu_v1/FSYNC_READOUT_V1.json', '93e79270166a05907343bf50118677b80a3e87d707bcc6525bee4e2c7bd00f6c'),
    'V6_preApp_closed': rr(R/'parent_supplement_native32_profile_intake_v1/campaign_ssd_v6_retry02/actual_native01/PARENT_ACTUAL_NATIVE_PREAPP_FAILED_CLOSED_V6.json', '16a64c4b3f85124ccd201d77e0f4aa18b6ae9b68ac48a03c35178889185a8034'),
    'V7_source': rr(H/'astra/full74_supplemental_mixed32_native_v7_ssd/release2/HANDOFF_V7_SSD.json', '4c6e17bfa558939ecd688dd80ba5ff793a41be010fe985af4a2b649842f5c7d7'),
    'V7_API_closed': rr(R/'parent_supplement_native32_profile_intake_v1/CPU/parent_ssd_v7_API02/PARENT_ACTUAL_CPU_API_CLOSED_V7.json', '923f3a62b1352894d9eb2157c90605700b5c52521d83ac5cd3f3ad26002771da'),
    'camera_protocol': rr(H/'CAMERA_SUFFICIENCY_ACCEPTANCE_SUPPLEMENT_V3.json', '1069d1ed2a5be3683821d268cb5daa3cd48a8b9bb7dc056f7350ca7d1a688d58'),
}

def authenticated_module(path, sha, name):
    with Path(path).open('rb') as f:
        a = os.fstat(f.fileno()); raw = f.read(2*1024**2+1); z = os.fstat(f.fileno())
    if len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('authenticated exact code buffer')
    if (a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(z.st_dev,z.st_size,z.st_mtime_ns,z.st_ctime_ns):
        raise ValueError('code bytes changed')
    m = types.ModuleType(name); m.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),m.__dict__)
    return m

def binding():
    return authenticated_module(BIND,BIND_SHA,'_report11_bound_metadata')

def exact_in_tree(tree, ref):
    if isinstance(tree,dict):
        return tree==ref or any(exact_in_tree(v,ref) for v in tree.values())
    return isinstance(tree,list) and any(exact_in_tree(v,ref) for v in tree)

def closed_actual_wait(b, ref):
    """Truthful failure waits are allowed; zero never implies measurement PASS."""
    w = b.read(ref)
    b.need(w.get('waitpid_observed') is True and type(w.get('child_pid')) is int and w['child_pid']==w.get('waited_pid'), 'actual owned wait4, not a snapshot')
    b.need(type(w.get('raw_wait_status')) is int and type(w.get('actual_wait_exit')) is int, 'typed raw and decoded actual exit')
    b.need(os.WIFEXITED(w['raw_wait_status']) or os.WIFSIGNALED(w['raw_wait_status']), 'terminal raw wait status')
    b.need(os.waitstatus_to_exitcode(w['raw_wait_status'])==w['actual_wait_exit'], 'raw wait and exit agree')
    b.need(w.get('owned_identities') and all(b.fresh_absent(i) for i in w['owned_identities']), 'exact enrolled identities now absent')
    b.need(b.sha(w['log'])==w['log_sha256'], 'terminal log SHA')
    return w

def tool_exit(b, ref):
    t = b.read(ref)
    b.need(t.get('session_closed') is True and type(t.get('actual_tool_exit')) is int, 'actual closed tool receipt, no default zero')
    terminal = t.get('actual_terminal_result',t.get('observed_completion'))
    b.need(isinstance(terminal,dict) and type(terminal.get('exit_code')) is int and terminal['exit_code']==t['actual_tool_exit'], 'actual terminal tool result agrees')
    return t['actual_tool_exit']

def static_inputs(b):
    s = {k:b.read(v) for k,v in STATIC.items()}
    b.need(s['public10']['commit']==s['public10_handoff']['commit']==BASE_COMMIT, 'exact already published10 base')
    b.need(s['host_closed']['all_owned_closed'] is True and s['host_compact']['speedup_claimed'] is False, 'closed host diagnostic only')
    b.need(s['host_compact']['closed_boundary_overhead_lower_bound_seconds']==332.811544538, 'actual host lower bound')
    for key in ('host_compact','IO_readout','fsync_readout'):
        c = s[key].get('actual_closure',s[key])
        b.closed_wait(c['actual_worker_wait']); b.closed_wait(c['actual_outer_wait'])
    api = s['V7_API_closed']; b.closed_wait(api['actual_worker_wait'])
    b.need(tool_exit(b,api['actual_tool_closure'])==0 and api['native_started'] is False, 'closed API CPU does not create native evidence')
    b.need(s['V6_preApp_closed']['native_actual_exit']==1 and s['V6_preApp_closed']['physical_App_Env_created'] is False, 'preserved before-App failure')
    return s

COVERAGE_KEYS = ('committed_micro_ids','full32_evaluation','complete_camera_groups','held27_status','device_identity','ProfileClose_status')

def prepared_terminal_inputs(b, packet_ref):
    """Parent supplies exact terminal refs and explicit field paths after closure.

    Projection keys are schema adapters, not a new arithmetic or camera oracle.
    A missing projection stays UNKNOWN. No live directory discovery is performed.
    """
    p = b.read(packet_ref)
    b.need(p.get('schema')=='astra.report11.closed_terminal_input.v1' and p.get('status')=='PARENT_SUPPLIED_CLOSED_FACTS', 'actual closed terminal packet required')
    b.need(p.get('bank_count')==0 and p.get('safety_acceptance') is False and p.get('holdout_opened') is False, 'no report-driven promotion')
    epoch = dt.datetime.fromisoformat(p['snapshot_utc']); b.need(epoch.utcoffset()==dt.timedelta(0), 'explicit frozen UTC')
    parent = b.read(p['parent_closed'])
    b.need('CLOSED' in parent.get('status','') and parent.get('bank_count')==0 and parent.get('all_requirements_completed') is False, 'closed parent outcome, requirements incomplete')
    b.need(dt.datetime.fromisoformat(parent['closed_utc'])<=epoch, 'parent closure before frozen cutoff')
    api = b.read(STATIC['V7_API_closed'])
    b.need(exact_in_tree(parent,api['request']) and exact_in_tree(parent,api['review']), 'exact current V7 request/review from genuine API closure')
    expected_native = R/'astra_full74_supplemental_mixed32_native_v7_ssd/native_attempt1/native_process/ACTUAL_WAIT.json'
    b.need(p['native_wait']['path']==str(expected_native), 'V7 exact actual native root; prior failures stay historical')
    for name in ('native_wait','outer_wait','tool_closure'):
        b.need(exact_in_tree(parent,p[name]), 'parent binds exact '+name)
    nw = closed_actual_wait(b,p['native_wait']); ow = closed_actual_wait(b,p['outer_wait']); te = tool_exit(b,p['tool_closure'])
    records = dict(parent=parent)
    if p.get('entry') is not None:
        b.need(exact_in_tree(parent,p['entry']), 'actual ENTRY linked to parent closure'); records['entry']=b.read(p['entry'])
    if p.get('raw_audit') is not None:
        records['raw_audit']=b.read(p['raw_audit'])
        b.need(p.get('raw_audit_wait') is not None, 'RAW review requires genuine independent CPU closure')
        b.closed_wait(p['raw_audit_wait'])
        b.need(exact_in_tree(records['raw_audit'],p['raw_audit_wait']), 'RAW audit binds its actual wait')
    b.need(set(p['coverage_projection'])==set(COVERAGE_KEYS), 'complete explicitly typed coverage adapter')
    coverage = {}
    for key,projection in p['coverage_projection'].items():
        if projection is None:
            coverage[key] = 'UNKNOWN_MISSING_BOUND_FIELD'; continue
        b.need(set(projection)=={'record','keys'} and projection['record'] in records and isinstance(projection['keys'],list) and projection['keys'], 'explicit actual field path')
        value = records[projection['record']]
        for component in projection['keys']:
            b.need(isinstance(component,str) and isinstance(value,dict) and component in value, 'actual projection field exists')
            value = value[component]
        coverage[key] = value
    ids = coverage['committed_micro_ids']
    if isinstance(ids,list):
        b.need(all(type(v) is int and 0<=v<=12 for v in ids) and sorted(set(ids))==ids, 'actual journal micro IDs; not allocated array tails')
    return dict(snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=epoch.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),native_exit=nw['actual_wait_exit'],native_raw_status=nw['raw_wait_status'],native_resource_abort=nw.get('resource_abort'),native_error=nw.get('error'),native_signals=nw.get('signals'),outer_exit=ow['actual_wait_exit'],tool_exit=te,entry_status=records.get('entry',{}).get('status','UNKNOWN_MISSING_ENTRY'),coverage=coverage,independent_RAW_scope='BOUND_CLOSED_REPORT_ONLY' if 'raw_audit' in records else 'UNKNOWN_NOT_AUDITED',bank_count=0,all_requirements_completed=False,safety_acceptance=False,tasks='NOT_RUN',holdout='UNOPENED',production_drive_peak='UNKNOWN',CONTACT_wrench_global_bound='UNKNOWN',same32_related_new_IID=0,packet_ref=packet_ref,bound_terminal_refs={k:v for k,v in p.items() if isinstance(v,dict) and set(v)=={'path','sha256'}})

def text(value):
    return html.escape(json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else str(value))

def bilingual(zh, en):
    return '<span class="r11-zh" lang="zh">'+html.escape(zh)+'</span><span class="r11-en" lang="en">'+html.escape(en)+'</span>'

def table(ident, title, heads, rows):
    return '<div class="table-window" tabindex="0"><table id="'+ident+'"><caption>'+title+'</caption><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def prepared_render(s, outcome=None):
    """Small additive section with native radio controls: JS is unnecessary."""
    draft = outcome is None
    title = bilingual('报告11准备稿：等待真实V7终态，未冻结科学快照','REPORT11 preparation: waiting for actual V7 terminal facts; no frozen scientific snapshot') if draft else bilingual('报告11：实际终态与验收缺口','REPORT11: actual terminal outcome and acceptance gaps')
    host = s['host_compact']; io = s['IO_readout']; fs = s['fsync_readout']
    io_rows = { (x['root'],x['method']):x for x in io['stats'] }
    stages = host['host_stage_inclusive_seconds']
    fmt = lambda v:format(v,'.6f')
    section = '<section id="nativeAttemptIncrement11"><h2>'+title+'</h2><fieldset class="r11-controls"><legend>阅读模式 / Reading controls</legend><label><input type="radio" id="r11LanguageZH" name="r11lang" checked>中文</label><label><input type="radio" id="r11LanguageEN" name="r11lang">English</label><label><input type="radio" id="r11Detailed" name="r11detail" checked>详细 / Detailed</label><label><input type="radio" id="r11Summary" name="r11detail">摘要 / Summary</label></fieldset><div class="r11-content">'
    section += '<p class="r11-result">'+bilingual('完整安全需求未完成；BANK0。模型预测、源代码检查、实际数据与物理验收各有独立范围。','Full safety requirements remain incomplete; BANK0. Model predictions, source checks, actual data and physical acceptance have distinct scopes.')+'</p>'
    if draft:
        section += '<p id="report11DraftBoundary">'+bilingual('这是SOURCE/UI准备组件，V7结果未接入；不表示当前运行结果，也不是公开候选。','This SOURCE/UI preparation component contains no V7 outcome; it is not a public candidate or a live status report.')+'</p>'
    else:
        section += '<p>'+bilingual('科学快照截止（不是实时）','Frozen scientific cutoff, not live status')+'：<time datetime="'+text(outcome['snapshot_utc'])+'">'+text(outcome['snapshot_utc'])+'</time> / Asia/Shanghai '+text(outcome['snapshot_Asia_Shanghai'])+'</p>'
        section += table('native11ActualTerminalTable',bilingual('真实关闭不等于测量通过','Actual closure does not establish valid measurement'),[bilingual('原生 / 父级 / 工具','Native / outer / tool'),bilingual('ENTRY与数据审计','ENTRY and data audit')],[[text({'native_exit':outcome['native_exit'],'native_raw_status':outcome['native_raw_status'],'outer_exit':outcome['outer_exit'],'tool_exit':outcome['tool_exit']}),text({'ENTRY':outcome['entry_status'],'RAW':outcome['independent_RAW_scope']})]])
    section += '<div class="r11-detail">'
    requirement_rows = [
        [bilingual('SOURCE / API','SOURCE / API'),bilingual('V7 SOURCE 4c6e、实际API348闭合；工程检查','V7 SOURCE 4c6e and actual API348 closed; engineering checks'),bilingual('不能代替真实原生或像素验收','Cannot replace native measurement or pixel acceptance')],
        [bilingual('模型分母','Model denominators'),'8192: 8018 NEG / 174 predicted CLEAR; 32768: 31986 NEG / 782 predicted CLEAR; cumulative40960 MODEL_ONLY',bilingual('原8192 EMPTY保留；预测CLEAR不是安全状态库','Original8192 EMPTY retained; predicted CLEAR is not a safety bank')],
        [bilingual('原生全74状态','Full74 native state'),'q74 / qd74 / target74; 32 related lanes × 13 journal points; 12 microsteps',text(outcome['coverage']['full32_evaluation']) if outcome else 'WAITING_ACTUAL_CLOSED_FACTS'],
        [bilingual('数值与backend','Numerics and backend'),'hard74 / speed74 / soft26 / raw9021 / scalar82; actual clock and source/readback',bilingual('本报告不重解码RAW；依赖单独已闭合审计','No RAW redecoding here; a separate closed audit is required')],
        [bilingual('原始相机','Raw cameras'),'32 initial + 32 final + all alarm groups across12 micro; 4 hand groups ×3 +3 wide views/group',text(outcome['coverage']['complete_camera_groups']) if outcome else 'WAITING_ACTUAL_CLOSED_FACTS'],
        [bilingual('held与设备身份','Held state and device identity'),'native27 bitwise before/after; simulation time unchanged; actual CUDA/Kit identity',text({k:outcome['coverage'][k] for k in ('held27_status','device_identity')}) if outcome else 'WAITING_ACTUAL_CLOSED_FACTS'],
        [bilingual('接触与drive','Contact and drive'),'production DrivePeak / CONTACT / wrench frame / terminal set / global error bound','UNKNOWN; '+bilingual('不存在全局物理误差界','No global physical error bound')],
        [bilingual('任务 / holdout','Tasks / holdout'),'tasks NOT_RUN; holdout UNOPENED; BANK0; complete=false',bilingual('同32重试相关；新增IID=0','Related retries of the same32; new IID=0')],
    ]
    section += table('native11RequirementsTable',bilingual('需求覆盖与风险','Requirements, evidence coverage and risks'),[bilingual('层级','Layer'),bilingual('所需证据或已知范围','Required evidence or known scope'),bilingual('实际状态 / 缺口','Actual status / gap')],requirement_rows)
    criteria = s['camera_protocol']['criteria']
    section += table('native11StrongCameraTable',bilingual('原强相机门槛全部保留','Original strong camera thresholds retained'),[bilingual('范围','Scope'),bilingual('量化门槛','Quantitative threshold')],[['4 hands ×3 views', 'pairwise ≥30°; palm ≥'+str(criteria['explicit_palm_peak_pixels_min'])+' px; each5 finger family ≥'+str(criteria['each_of_five_finger_families_peak_pixels_min'])+' px'],['all native hand bodies','each peak ≥64 px; zero gaps'],['3 wide views / 4 arms / 2 tables','each arm ≥1024 px; each table ≥1024 px'],['critical forearm + wrist2','each native body ≥128 px; zero gaps'],['native27 / time / inventory','ALL registered before-after bitwise equal; time unchanged; raw RGB/masks/path/SHA; independent pixels']])
    section += '<p>'+bilingual('组内各视角峰值不是完整表面、接触点可见性、无遮挡或物理安全证明。初末态与全部alarm组不能靠部分目录或PNG数量补齐。','Per-view group maxima do not prove full-surface or contact-patch visibility, absence of occlusion, or physical safety. Directory or PNG counts cannot fill missing initial/final or alarm groups.')+'</p>'
    section += table('native11HostTable',bilingual('已关闭V5 host诊断：inclusive时间不可相加','Closed V5 host diagnostic: inclusive times are nonadditive'),[bilingual('边界','Boundary'),'host seconds',bilingual('解释','Interpretation')],[['NPZ file / nested codec',fmt(stages['original_NPZ_file'])+' / '+fmt(stages['original_NPZ_codec']),'5076 begun / 5075 closed; file median43.752434ms; nested, NONADDITIVE'],['PNG file / nested codec',fmt(stages['original_PNG_file'])+' / '+fmt(stages['original_PNG_codec']),'NONADDITIVE'],['JSON file / nested codec',fmt(stages['original_JSON_file'])+' / '+fmt(stages['original_JSON_codec']),'NONADDITIVE'],['existing render',fmt(stages['existing_render']),'host boundary, not GPU kernel time'],['completed boundary overhead LOWERBOUND',format(host['closed_boundary_overhead_lower_bound_seconds'],'.3f'),'includes logger/setup/scheduling; NOT pure IO; journal_IO_ns UNKNOWN_PROFILE_CLOSE_ABSENT']])
    section += table('native11IOTable',bilingual('真实CPU文件基准，不是native加速结论','Actual CPU file benchmark, not native speedup'),['root / method','median ms',bilingual('证据边界','Scope')],[[root+' / '+method,fmt(io_rows[(root,method)]['combined_related15_statistics']['median_ms']),'3 related original samples ×5 repeats; typed bytes checked'] for root,method in [('NAS_CIFS','compressed_original'),('SSD_NVME_EXT4','compressed_original'),('NAS_CIFS','hardlink_verified_current')]])
    section += '<p>'+bilingual('NAS hardlink inode身份UNKNOWN_NOT_ADOPTABLE；105.011537ms包含实际验证路径，不能宣传link加速。fsync基准保留每event fsync与完全相同bytes；本工作负载没有显示SSD收益。不能外推已改善V7时限、崩溃原因或物理结果。','NAS hardlink inode identity is UNKNOWN_NOT_ADOPTABLE;105.011537ms includes the actual verification path and does not support a link speedup claim. The fsync benchmark preserves every-event fsync and identical bytes; this workload demonstrates no SSD benefit. Neither benchmark establishes a V7 speedup, crash cause, or physical outcome.')+'</p>'
    section += table('native11FsyncTable',bilingual('固定fsync基准：5次相关重复','Fixed fsync benchmark:5 related repetitions'),['root','total close median ms'],[[v['root'],fmt(v['rep_total_close_statistics']['median_ms'])] for v in fs['statistics']])
    section += '<p>'+bilingual('V6 SSD路由尝试真实before-App失败native1、outer1/tool1，0物理步与0render；旧失败永久保留。V7 SOURCE/API0仅是修复工程证据，与原生终态及DATA验收分别列明。','The V6 SSD routing attempt failed before App with native1,outer1/tool1,0 physics steps and0 renders; this failure is retained. V7 SOURCE/API0 are engineering evidence, distinct from native closure and DATA acceptance.')+'</p></div></div></section>'
    return section

STYLE = '''#nativeAttemptIncrement11 .r11-controls{display:flex;flex-wrap:wrap;gap:.8rem}#nativeAttemptIncrement11 .r11-controls label{white-space:nowrap}#nativeAttemptIncrement11 .r11-content{overflow-wrap:anywhere}#nativeAttemptIncrement11 .r11-en{display:none}#nativeAttemptIncrement11:has(#r11LanguageEN:checked) .r11-zh{display:none}#nativeAttemptIncrement11:has(#r11LanguageEN:checked) .r11-en{display:inline}#nativeAttemptIncrement11:has(#r11Summary:checked) .r11-detail{display:none}'''

def prepared_fragment(b):
    s = static_inputs(b)
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REPORT11 SOURCE/UI preparation only</title><style>'+STYLE+'</style><main>'+render(s)+'</main></html>'

def build(b, packet_ref, out):
    """Only after actual closure: produce a small inherited overlay for peer review."""
    outcome = terminal_inputs(b,packet_ref)  # reject live/missing before any mkdir
    s = static_inputs(b); base = s['public10_handoff']; oldplan=b.read(base['plan']); oldmanifest=b.read(base['manifest'])
    graph = b.read(b.ref(D/'SOURCE_GRAPH_V1.json'))
    for ref in graph['own_source_files'].values(): b.buffer(ref,1024**2)
    out = Path(out); b.need(out.is_absolute() and out.resolve().is_relative_to(O) and not out.exists(), 'exclusive REPORT11 candidate namespace')
    sources_before = {k:b.ref(v['path']) for k,v in graph['own_source_files'].items()}
    out.mkdir(parents=True,exist_ok=False); overlay=out/'overlay'; overlay.mkdir()
    def copy_file(ref, dest):
        raw=b.buffer(ref,4*1024**2);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())
    for name,ref in STATIC.items():copy_file(ref,overlay/PREFIX/'evidence'/(name+'.json'))
    for name,ref in outcome['bound_terminal_refs'].items():copy_file(ref,overlay/PREFIX/'evidence'/(name+'.json'))
    for name,ref in [('index.html',base['HTML']),('panel_manifest.json',base['manifest']),('summary.json',oldplan['file_sources']['summary.json'])]:copy_file(ref,overlay/PREFIX/'archive_candidate10'/name)
    for name,ref in graph['own_source_files'].items():copy_file(ref,overlay/PREFIX/'source'/name)
    copy_file(b.ref(D/'SOURCE_GRAPH_V1.json'),overlay/PREFIX/'source'/'SOURCE_GRAPH_V1.json')
    outcome.update(model_denominators=dict(original8192=dict(n=8192,NEG=8018,predicted_CLEAR=174,UNKNOWN=0,schedule='EMPTY'),supplement32768=dict(n=32768,NEG=31986,predicted_CLEAR=782,UNKNOWN=0),cumulative_MODEL_ONLY=40960),static_refs=STATIC)
    factref=b.write(overlay/PREFIX/'FACTS_INCREMENT11_V1.json',outcome)
    helper=authenticated_module(H/'astra/report_increment07_v1/build_v3.py','dce7ccdfc222a27053194b6db2f9fb4d2b8adfed6682ac1ee50eaf3fe0274026','_report11_presentation');replace_id,_=helper.presentation_helpers(b)
    page=b.buffer(base['HTML'],4*1024**2).decode(); epoch=outcome['snapshot_utc'];local=outcome['snapshot_Asia_Shanghai']
    header='<p class="asof" id="reportAsOf">报告11冻结事实快照 / Frozen snapshot <time datetime="'+text(epoch)+'">'+text(epoch)+' UTC</time>；Asia/Shanghai '+text(local)+'。不是实时状态。06–10各节及原字节archive保留各自截止时点。</p>'
    page=replace_id(page,'p','reportAsOf',header)
    b.need(page.count('<section id="nativeAttemptIncrement10">')==1 and page.count('</head>')==1,'one preserved10 section/head')
    page=page.replace('</head>','<style id="report11ModesV1">'+STYLE+'</style></head>',1)
    notice='<p id="native11Executive"><strong>'+bilingual('完整安全验收尚未通过，BANK0。','Full safety acceptance remains incomplete; BANK0.')+'</strong> '+bilingual('实际终态、相机和数值审计的范围分别列明；SOURCE/API0不升级物理验收。','Actual terminal, camera and numeric audit scopes are listed separately; SOURCE/API0 cannot promote physical acceptance.')+' <a href="#nativeAttemptIncrement11">Report11: native'+str(outcome['native_exit'])+' / outer'+str(outcome['outer_exit'])+' / tool'+str(outcome['tool_exit'])+'</a></p>'
    b.need(page.count('<div class="executive">')==1,'one inherited executive')
    page=page.replace('<div class="executive">','<div class="executive">'+notice,1)
    page=page.replace('<section id="nativeAttemptIncrement10">',render(s,outcome)+'<section id="nativeAttemptIncrement10">',1)
    needle='<a href="#nativeAttemptIncrement10">V5 timeout / CPU codec / UNKNOWN边界</a>';b.need(page.count(needle)==1,'one historical10 TOC')
    page=page.replace(needle,'<a href="#nativeAttemptIncrement11">最新终态与验收缺口 / Terminal outcome and coverage</a>'+needle,1)
    with (overlay/'index.html').open('x') as f:f.write(page)
    summary=b.read(oldplan['file_sources']['summary.json']);summary.update(asof_utc=epoch,asof_AsiaShanghai=local,increment11=outcome,historical_candidate10_summary=PREFIX+'/archive_candidate10/summary.json');b.write(overlay/'summary.json',summary)
    files={p.relative_to(overlay).as_posix():b.ref(p) for p in overlay.rglob('*') if p.is_file()}
    updated=copy.deepcopy(oldmanifest);updated['files'].update({k:v['sha256'] for k,v in files.items()});updated.update(snapshot_updated_utc=epoch,increment11=dict(facts=PREFIX+'/FACTS_INCREMENT11_V1.json',bank_count=0,safety_acceptance=False))
    b.write(overlay/'panel_manifest.json',updated);files['panel_manifest.json']=b.ref(overlay/'panel_manifest.json')
    b.need(all(updated['files'][n]==sha for n,sha in oldmanifest['files'].items() if n not in ('index.html','summary.json')), 'all old snapshots/8fig15RGB hashes preserved')
    plan=dict(schema='astra.report.increment11.overlay_plan.v1',base_commit=BASE_COMMIT,base_plan=base['plan'],base_manifest=base['manifest'],before_sha256={n:(base['manifest']['sha256'] if n=='panel_manifest.json' else oldmanifest['files'][n]) for n in files if n=='panel_manifest.json' or n in oldmanifest['files']},file_bytes={n:Path(q['path']).stat().st_size for n,q in files.items()},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=files,changed_files={n:v['sha256'] for n,v in files.items()},inherited_files={n:v for n,v in oldmanifest['files'].items() if n not in files},manifest=b.ref(overlay/'panel_manifest.json'),source_facts=factref,source_graph=b.ref(D/'SOURCE_GRAPH_V1.json'),old8fig_and15RGB_SHA_preserved=True,new_science_figures=0,raw_tree_copy=False,publish_performed=False,scientific_peer='PENDING_DISTINCT_REPORT11_PEER')
    b.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    b.need(sources_before=={k:b.ref(v['path']) for k,v in graph['own_source_files'].items()}, 'source unchanged during build')
    return b.write(out/'CANDIDATE_BUILD_HANDOFF_V1.json',dict(status='BUILT_LOCAL_CANDIDATE_PENDING_REAL_LOCAL_QA_AND_DISTINCT_SCIENCE_PEER',HTML=b.ref(overlay/'index.html'),plan=b.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],facts=factref,snapshot_utc=epoch,source_before_after_unchanged=True,publish_performed=False,bank_count=0,safety_acceptance=False))

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--packet',required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(build(binding(),rr(a.packet,a.packet_sha256),a.out),ensure_ascii=False))

def terminal_inputs(b, packet_ref):
    value=prepared_terminal_inputs(b,packet_ref)
    parent=b.read(value['bound_terminal_refs']['parent_closed'])
    b.need(parent['native_actual_exit']==-9 and parent['raw_native_wait_status']==9 and parent['outer_actual_exit']==parent['actual_parent_tool_exit']==1,'actual failed V7 snapshot')
    b.need(parent['native_work_deadline_seconds']==1200 and parent['committed_numeric_micro_ids']==[0] and parent['completed_camera_group_receipts']==0,'no prefix or camera qualification')
    b.need(parent['registered_owned_count']==len(parent['fresh_owned_absence'])==121 and all(i['fresh_same_identity_live'] is False for i in parent['fresh_owned_absence']),'actual closed121 catalogue')
    for i in parent['fresh_owned_absence']:
        b.need(b.fresh_absent(dict(pid=i['pid'],startticks=i['startticks'],live=False)),'fresh V7 PID/starttick absence')
    committed=b.read(STATIC['producer_micro0'])['data']
    admission=b.read(STATIC['producer_admission'])['data']
    b.need(committed['micro']==0 and len(committed['flags'])==32 and all(len(row)==7 and all(flag is False for flag in row) for row in committed['flags']),'producer initial flags only')
    b.need(admission['exact']==admission['root']==[True]*32,'producer initial restore observations only')
    value.update(historical_snapshot_not_live=True,native_work_deadline_seconds=1200,native_elapsed_seconds_including_grace=parent['native_elapsed_seconds'],supervisor_signal_sequence=[v['signal'] for v in parent['native_supervisor_signals']],registered_fresh_absent_count=121,allocated_initial_env_dirs_NOT_completed=parent['allocated_camera_env_dirs_not_completed']['initial'],partial_PNG_NOT_qualified=parent['partial_PNG_count_not_qualified'],committed_micro_ids=[0],missing_micro_ids=list(range(1,13)),full32_independent_prefix='NOT_EVALUATED',ENTRY='MISSING',PRODUCER='MISSING',PROFILE_CLOSE='MISSING',physical_prefix_advance_count=parent['actual_physical_prefix_advance_count'],journal_IO_ns=parent['journal_IO_ns'],RAW_pixels='UNKNOWN_NOT_AUDITED',held27='UNKNOWN_INCOMPLETE_RECEIPTS',latest42='UNKNOWN_CATALOGUE_UNBOUND',producer_INITIAL_observations=dict(micro=0,flags_shape=[32,7],all_flags_false=True,q_qd_target_restore_exact_count=32,root_admission_count=32,scope='PRODUCER_INITIAL_ONLY_NOT_INDEPENDENT_RAW_OR_FULLPREFIX_PASS',refs={k:STATIC[k] for k in ('producer_micro0','producer_admission')}),native_speedup_demonstrated=False,crash_cause='UNKNOWN_RETAINED',new_science_figures=0,old_figures=8,original_RGB=15,excluded_later_work='ongoing Franklin benchmark and camera pose source candidate excluded')
    return value

def render(s,outcome=None):
    body=prepared_render(s,outcome)
    if outcome is None:return body
    detail=table('native11FailedScopeTable',bilingual('V7原期限失败与未完成项','V7 original-deadline failure and incomplete products'),[bilingual('范围','Scope'),bilingual('真实观测','Actual observation'),bilingual('验收边界','Acceptance limit')],[
        ['native / outer / tool','−9 / raw9; outer1 / tool1; original1200s; TERM15 →10s grace→KILL9; elapsed'+text(outcome['native_elapsed_seconds_including_grace'])+'s; CPUcheck0;121 exact owned fresh absent','failure preserved, no physical acceptance'],
        ['allocated versus completed','14 allocated initial env dirs;548 partial PNG;0 cameraGroups','not14 completed groups; pixels/held27/latest42 UNKNOWN'],
        ['numeric journal','committed micro[0] only; missing1..12; full32 independent prefix NOT_EVALUATED','ENTRY / PRODUCER / ProfileClose MISSING; physical prefix advances UNKNOWN_PRODUCER_ABSENT'],
        ['producer INITIAL observations','flags32×7 allfalse; exact q/qd/target restore32; root admission32','PRODUCER_INITIAL_ONLY; not independent RAW or full-prefix PASS']])
    detail+='<p>'+bilingual('V5的15个与V7的14个allocated目录来自相关重试，并有不同启动与SOURCE条件；不得用目录数比较native收益。SSD版本保留原fresh SDK getters、codec与NAS phase/journal；真实CPU文件38.060401/3.076201ms不是原生端到端加速证明。','V5’s15 and V7’s14 allocated directories are related retries with different startup and source conditions; directory counts cannot measure native gain. Fresh SDK getters,codec and NAS phase/journal were retained;CPU file38.060401/3.076201ms is not a native end-to-end speedup.')+'</p>'
    detail+='<ul>'+''.join('<li><a href="'+PREFIX+'/evidence/'+k+'.json" download>'+html.escape(k)+'</a></li>' for k in STATIC)
    detail+=''.join('<li><a href="'+PREFIX+'/evidence/'+k+'.json" download>'+html.escape(k)+'</a></li>' for k in outcome['bound_terminal_refs'])
    detail+='<li><a href="'+PREFIX+'/FACTS_INCREMENT11_V1.json" download>Frozen typed facts / 冻结事实</a></li><li><a href="'+PREFIX+'/archive_candidate10/index.html">PUBLIC10 frozen archive / 历史原字节</a></li></ul>'
    return body.replace('<div class="r11-detail">','<div class="r11-detail">'+detail,1)

STATIC.update({'producer_micro0': {'path': '/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352/astra/full74_supplemental_mixed32_native_v7_ssd/evidence/native_attempt1/native/journal/00001.json', 'sha256': '80e34319a60fa060c2af2bbdb3d35f01dc6f92a78ad2997b725cd2fdc5cfc8de'}, 'producer_admission': {'path': '/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352/astra/full74_supplemental_mixed32_native_v7_ssd/evidence/native_attempt1/native/journal/00002.json', 'sha256': 'd67037bb127dce7f487e1d76c4107f09acd3af1545c391413cf0ee8613bd3f78'}})

if __name__=='__main__':main()
