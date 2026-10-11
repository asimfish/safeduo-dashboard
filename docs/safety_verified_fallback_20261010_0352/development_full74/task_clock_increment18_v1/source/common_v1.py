"""Bounded Report18 metadata; inherit science without repeating a scientific audit."""
import hashlib,types
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent;O=R/'astra_report_increment18_v1'
_p=H/'astra/report_increment16_v1/actual_data_release3/common_v1.py';_raw=_p.read_bytes()
if hashlib.sha256(_raw).hexdigest()!='1f897053b0cb833c6172409fecbbee26865f2b7d4572cd044a76e44447f131f6':raise ValueError('authenticated original metadata buffer')
_m=types.ModuleType('_r18_exact_metadata');_m.__file__=str(_p);exec(compile(_raw,str(_p),'exec'),_m.__dict__)
for _k in ('need','sha','ref','buffer','read','copy','module','absent','wait','tool','PY','BPY','W','REL','THREADS','LOADER','LOADER_SHA','CORE_SHA','OBSERVER','OBSERVER_SHA'):globals()[_k]=getattr(_m,_k)
_m.D=D;_m.O=O;write=_m.write
INPUTS={'path':str(D/'FROZEN_INPUTS_V1.json'),'sha256':'00830e567d2fed40acd53be9310546e2ea247d72a69c3d5004bc053ab41f21b9'}
PREFIX='task_clock_increment18_v1'
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'
LAUNCH_ARGS=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox']
def sources():
    g=read(ref(D/'SOURCE_GRAPH_V1.json'))
    need(all(ref(q['path'])==q for q in g['own_source_files'].values()),'own exact immutable sources')
    need(all(ref(q['path'])==q for q in g['reused_source_files'].values()),'exact reused metadata and guardian')
    return g
def inherited():
    frozen=read(INPUTS);inputs=frozen['inputs'];v={k:read(q) for k,q in inputs.items()}
    p=v['report17_parent'];tool(inputs['report17_parent_tool']);need(p['public_commit']==frozen['base_public_commit'] and p['all_current_actual_worker_outer_tool_metadata0'] and p['all_current_registered_owned_fresh_absent'],'actual accepted public17 current scope')
    need(p['bank_count']==0 and not p['full12'] and not p['safety_acceptance'] and p['historical71948']=='UNKNOWN_PERMANENT','old acceptance and unknown preserved')
    t=v['task_parent'];tool(inputs['task_parent_tool'])
    need(t['author_new_checks']==20 and t['independent_new_checks']==5 and t['all_current_actual_worker_outer_tool_metadata_final_seal0'] and t['all_registered_owned_exact_identities_fresh_absent'] and t['current_SOURCE_unchanged'],'real closed SOURCE20 plus independent5')
    author=v['task_author'];peer=v['task_independent_peer'];review=v['task_independent_review'];gap=v['task_production_schema_gap']
    need(t['author']==inputs['task_author'] and t['independent_peer']==inputs['task_independent_peer'] and t['independent_review']==inputs['task_independent_review'],'same actual author/peer subject')
    need(peer['verdict']=='CONDITIONAL_PASS' and not peer['blocking_findings'] and peer['author_handoff']==inputs['task_author'] and peer['independent_CPU_oracles']==5,'focused distinct review, not physical acceptance')
    wait(author['actual_worker_wait']);wait(author['actual_outer_wait']);tool(author['actual_tool_closure']);tool(author['actual_metadata_tool_closure'])
    wait(peer['actual_worker_wait']);wait(peer['actual_outer_wait']);tool(peer['actual_tool_wait']);tool(peer['actual_metadata_tool_wait']);tool(inputs['task_independent_actual_final_seal'])
    need(all(absent(i) for i in author['actual_owner_identities']) and all(absent(i) for i in peer['registered_owned_identities']) and absent(p['main_identity']) and absent(t['main_identity']),'closed exact identities remain absent')
    need(review['independent_check_count']==5 and review['original_counterexample']==inputs['post_live_clock_counterexample'] and review['repaired_evidence']==inputs['post_live_clock_repair_evidence'],'preserved original counterexample and real repair evidence')
    need(gap['status']=='REAL_BANK_SCHEMA_AND_NATIVE_PROVIDER_MISSING' and gap['actual_bank_count']==0 and not gap['native_dispatch_supported'],'production schema/provider explicitly missing')
    need(t['actual_bank_count']==t['native_runs']==t['task_runs']==t['SDK_calls']==0 and not t['safety_acceptance'] and not t['task_acceptance'] and not t['holdout_opened'] and t['backend_attestation']=='NOT_ATTESTED','SOURCE-only and backend scope')
    cap=v['capacity'];need(cap['SSD_available_bytes']==10168233984 and cap['SSD_original_minimum_bytes']==20401094656 and cap['SSD_deficit_bytes']==10232860672 and not cap['SSD_original_threshold_met'],'exact informational historical SSD values')
    need(cap['NAS_available_bytes']==5921033027584 and cap['admission_permission'] is False and cap['fresh_SOURCE_READY_App_inventory'] is False and cap['thresholds_unchanged'],'capacity is not admission or fresh inventory')
    h=v['report17_candidate'];need(h['snapshot_utc']==frozen['scientific_snapshot_utc'] and h['HTML']['sha256']=='ec12a6750b7c7a41cc0aecde54e479d9ac775c51448a0a04872e453d5b4c8885','exact prior frozen scientific snapshot')
    return frozen,v,h,read(h['plan']),read(h['manifest']),read(h['facts'])
def authenticate(q):
    h=read(q);frozen=read(INPUTS)
    need(h['schema']=='astra.report18.LOCAL_candidate.v1' and h['engineering_snapshot_utc']==frozen['engineering_snapshot_utc'] and h['scientific_snapshot_utc']==frozen['scientific_snapshot_utc'] and h['base_public_commit']==frozen['base_public_commit'],'actual current LOCAL candidate schema')
    need(h['publish_performed'] is False and h['science_review_pending'] is True and h['bank_count']==0 and h['backend_attestation']=='NOT_ATTESTED','pre-science candidate, no promotion')
    g=read(h['source_graph']);need(g==sources(),'current candidate SOURCE binding')
    p=read(h['plan']);f=read(h['facts']);m=read(h['manifest'])
    need(p['schema']=='astra.report18.partial_overlay.v1' and p['source_facts']==h['facts'] and p['source_graph']==h['source_graph'] and p['manifest']==h['manifest'],'actual plan/facts/SOURCE refs')
    need(h['HTML']==p['file_sources']['index.html'] and m['files']['index.html']==h['HTML']['sha256'],'exact HTML manifest')
    need(f['full12'] is False and f['safety_acceptance'] is False and f['bank_count']==0 and f['backend_attestation']=='NOT_ATTESTED' and f['scientific_snapshot_utc']==frozen['scientific_snapshot_utc'],'science boundary remains unchanged')
    return h,p,f,m
def route_leaf(path):
    need(path.startswith('/'+REL),'scoped report route');leaf=path[len('/'+REL):] or 'index.html'
    need(not Path(leaf).is_absolute() and '..' not in Path(leaf).parts,'safe declared route');return leaf
