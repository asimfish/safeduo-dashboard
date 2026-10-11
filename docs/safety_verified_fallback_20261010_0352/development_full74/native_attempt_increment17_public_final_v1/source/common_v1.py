"""Report17 public-final presentation; unchanged scientific snapshot and original guardian."""
from pathlib import Path
import hashlib,types
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent;O=R/'astra_report_increment17_v1'
_p=H/'astra/report_increment16_v1/actual_data_release3/common_v1.py'
_raw=_p.read_bytes()
if hashlib.sha256(_raw).hexdigest()!='1f897053b0cb833c6172409fecbbee26865f2b7d4572cd044a76e44447f131f6':raise ValueError('original authenticated metadata')
_m=types.ModuleType('_r17_final_metadata');_m.__file__=str(_p);exec(compile(_raw,str(_p),'exec'),_m.__dict__)
for _k in ('need','sha','ref','buffer','read','copy','module','absent','wait','tool','PY','BPY','W','REL','THREADS','LOADER','LOADER_SHA','CORE_SHA','OBSERVER','OBSERVER_SHA'):globals()[_k]=getattr(_m,_k)
_m.D=D;_m.O=O;write=_m.write
CUTOFF='2026-10-11T17:29:19.248617+00:00'
BASE='23fd54124598ed394c226cdc22b9b5d50f26d6a7'
def q(p,s):return {'path':str(p),'sha256':s}
DRAFT=q(O/'build/source_draft05/LOCAL_SOURCE_DRAFT_HANDOFF_V17.json','13edc35819980553bdcc791b8a8f499020061169d30b8db5d3d91bb03124c03b')
DRAFT_LOCAL=q(O/'draft5/LOCAL_actual_closed_v1/CLOSED_CURRENT_SOURCE_LOCAL_HANDOFF_V17_DRAFT5.json','f5222ffaca37f8878baacd8f323e0f046998ab1200b0b590bddb934df80d9610')
DRAFT_TERMINAL=q(O/'draft5/LOCAL_actual_closed_v1/ACTUAL_FINAL_LOCAL_METADATA_TOOL_CLOSED_V17_DRAFT5.json','b7c62fa9e7228750d5b3669ebf0136a5e05ce26fd38a1eb4a07cfe632d6afbfa')
PEER=q(R/'astra_report17_draft5_independent_scientific_peer_v1/actual01/FINAL_CLOSED_SCIENTIFIC_HANDOFF_V1.json','5811bb1a5512f0dfcd91aabd1ff2a347cdf2bac2972c0607edcf1e1c93d575c9')
PEER_TERMINAL=q(R/'astra_report17_draft5_independent_scientific_peer_v1/actual01/FINAL_METADATA_ACTUAL_TOOL_TERMINAL_V1.json','70acc22a2b85c6fb53b28c71cfa4bf9c1a093c1a36c48806e7036a51cda6c8a5')
IMAGE_PREFIX='native_partial_increment16_v1/assets/'
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'
LAUNCH_ARGS=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox']
def sources():
    g=read(ref(D/'SOURCE_GRAPH_V1.json'))
    need(all(ref(v['path'])==v for v in g['own_source_files'].values()),'final presentation source unchanged')
    need(all(ref(v['path'])==v for v in g['reused_source_files'].values()),'reused closed sources unchanged')
    return g
def inherited():
    h=read(DRAFT);a=read(DRAFT_LOCAL);t=tool(DRAFT_TERMINAL)
    need(a['candidate']==DRAFT and t['handoff']==DRAFT_LOCAL and a['source_unchanged'],'actual draft5 LOCAL binding')
    for stage in a['actual_stages'].values():wait(stage['actual_worker_wait']);wait(stage['actual_outer_wait']);tool(stage['actual_tool_wait'])
    need(all(absent(i) for i in a['owned_identities']),'draft5 exact identities absent')
    peer=read(PEER);terminal=tool(PEER_TERMINAL)
    need(peer['schema']=='astra.report17.distinct.science.final.closed.v1' and peer['status']=='PASS' and not peer['blocking_findings'] and peer['candidate']==DRAFT and peer['subject_handoff']==DRAFT_LOCAL and peer['snapshot_utc']==CUTOFF,'actual distinct draft5 scientific PASS')
    need(peer['actual_exit_codes']==[0,0,0,0] and not peer['metadata_pending'] and peer['source_unchanged'] and terminal['final_handoff']==PEER,'science metadata and terminal bound')
    wait(peer['actual_worker_wait']);wait(peer['actual_outer_wait']);tool(peer['actual_tool_wait']);tool(peer['actual_metadata_tool_wait'])
    review=read(peer['review']);need(review['status']=='PASS' and not review['blocking_findings'] and review['candidate']==DRAFT,'actual review exact subject')
    need(all(absent(i) for i in terminal['fresh_absence']),'science registered exact identities absent')
    return h,a,peer
def authenticate(candidate_ref):
    h=read(candidate_ref);need(h['schema']=='astra.report17.public_final_candidate.v1' and h['snapshot_utc']==CUTOFF and h['base_public_commit']==BASE,'explicit final subject schema')
    need(h['inherited_draft']==DRAFT and h['inherited_science']==PEER and h['publish_performed'] is False and h['bank_count']==0 and h['safety_acceptance'] is False,'final presentation, no promotion')
    g=read(h['source_graph']);need(all(ref(v['path'])==v for v in g['own_source_files'].values()),'actual final subject source')
    p=read(h['plan']);f=read(h['facts']);m=read(h['manifest'])
    need(p['schema']=='astra.report17.public_final_partial_overlay.v1' and p['source_graph']==h['source_graph'] and p['source_facts']==h['facts'] and p['manifest']==h['manifest'],'final plan exact refs')
    need(h['HTML']==p['file_sources']['index.html'] and m['files']['index.html']==h['HTML']['sha256'] and f['snapshot_utc']==CUTOFF,'actual HTML/facts/manifest')
    need(f['bank_count']==0 and f['full12'] is False and f['V13_initial_acceptance_inherited']['selected_original_images']==480 and p['old16_and_earlier_tail_byte_identical'],'unchanged scope/history')
    return h,p,f,m
def route_leaf(path):
    need(path.startswith('/'+REL),'scoped report route');leaf=path[len('/'+REL):] or 'index.html'
    need(not Path(leaf).is_absolute() and '..' not in Path(leaf).parts,'safe declared route');return leaf
