"""Seal actual finished report10 build/browser tools, without launching work."""
import argparse, datetime, hashlib, json, os, resource, types
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent;O=R/'astra_report_increment10_v1/repair1'

def module(path,sha,name):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('authenticated exact buffer')
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def main():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--source-sha256',required=True);ap.add_argument('--phase',choices=['build','browser'],required=True);ap.add_argument('--initial-json',required=True);ap.add_argument('--terminal-json',required=True);a=ap.parse_args()
 b=module(H/'astra/report_increment05_v4/binding_v1.py','bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25','_closed_binding')
 b.need(b.sha(__file__)==a.source_sha256 and os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in b.THREADS),'exact bounded CPU seal')
 initial=json.loads(a.initial_json);terminal=json.loads(a.terminal_json);b.need(terminal['exit_code']==0 and terminal.get('session_id') is None,'real terminal tool0, no launch-only closure')
 root=O/('build_attempt1' if a.phase=='build' else 'local_browser_attempt1');worker=b.ref(root/'worker/ACTUAL_WAIT.json');outer=b.ref(root/'external/ACTUAL_WAIT.json');b.closed_wait(worker);b.closed_wait(outer)
 actual=b.write(root/'ACTUAL_TOOL_CLOSURE_V1.json',dict(schema='astra.report10.actual_tool_closure.v1',initial_exec_result=initial,actual_terminal_result=terminal,tool_session_id=initial.get('session_id'),session_closed=True,actual_exit=0,actual_worker_wait=worker,actual_wrapper_wait=outer,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 if a.phase=='build':
  h=b.read(b.ref(root/'CLOSED_BUILD_HANDOFF_V1.json'));h['actual_tool_wait']=actual;q=b.write(root/'CLOSED_BUILD_TOOL_HANDOFF_V1.json',h);print(json.dumps(dict(handoff=q)),flush=True);return 0
 local=b.read(b.ref(root/'CLOSED_LOCAL_BROWSER_V1.json'));br=b.read(local['browser_report'])
 b.need(br['status']=='PASS_REAL_BROWSER' and len(br['modes'])==4 and not br['errors'] and br['browser_closed'] and br['playwright_closed'] and br['preview_closed'] and br['summary_body_facts_datetime_consistent'],'four real browser modes, exact time and complete process cleanup')
 h=b.read(local['candidate']);h.update(schema='astra.report.increment10.closed_candidate.v1',status='CLOSED_REPORT10_CANDIDATE_LOCAL4MODES_PENDING_DISTINCT_HOOKE_SCIENTIFIC_PEER',actual_local_browser=local['browser_report'],actual_local_worker_wait=worker,actual_local_wrapper_wait=outer,actual_local_tool_wait=actual,local_browser_modes=4,old8fig_and15RGB_SHA_preserved=True,own_GPU_native_Git_actions=0,publish_performed=False,report10_scientific_peer=None,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 graph=b.read(h['source_graph']);b.need(all(b.sha(path)==sha for path,sha in graph['own_source_sha256'].items()),'all author/QA/seal bytes unchanged')
 q=b.write(D/'CLOSED_CANDIDATE_HANDOFF_V10.json',h)
 request=b.write(D/'REQUEST_DISTINCT_SCIENTIFIC_PEER_V10.json',dict(candidate=q,requested_distinct_reviewer='Hooke or Descartes (parent relay)',parent_relay_required=True,scope='Report10 frozen00:11:44UTC: V5 original1200 timeout native-9/raw9 TERM15/KILL9 outer/tool1; micro0only 15dirs0groups/32NOT_EVAL; missingENTRY/ProfileClose/held27/latest42/RAWpixels/journalIO; V4SIGSEGVUNKNOWN; q2API0/SOURCE9d/processmapfe389 scoped; CPUBytesIO three related samples5each and10pairs27typed equal, noNAS/nativegain; inherited09/8fig15RGB/models preserved',snapshot_utc=h['snapshot_utc'],actual_local_browser=local['browser_report'],publication_authority='NO_PUBLISH_AUTHORITY_THIS_TASK: distinct peer then later explicit parent publication instruction required',automatic_peer_or_publish=False))
 print(json.dumps(dict(candidate=q,peer_request=request,worker_wait=worker,wrapper_wait=outer)),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
