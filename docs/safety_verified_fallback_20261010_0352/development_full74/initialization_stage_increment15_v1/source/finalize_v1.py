"""Record genuine observed tool terminals and close only this local candidate scope."""
import argparse,datetime as dt,json,os
from pathlib import Path
import common_v1 as c

def identity():
    f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),startticks=int(f[19]),ppid=os.getppid())

def stage(root,tool):
    root=Path(root);w=c.wait(c.ref(root/'worker/ACTUAL_WAIT.json'));outer=c.wait(c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json'));c.tool(tool)
    d=c.read(c.ref(root/'CPU_DELIVERY_V1.json'));c.need(d['source_unchanged'] and d['source_before_after']==c.sources(),'real stage frozen source')
    return dict(worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),outer_wait=c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json'),tool_closure=tool,delivery=c.ref(root/'CPU_DELIVERY_V1.json'))

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--mode',choices=['record-stage','tools','handoff','final-object'],required=True)
    for k in ['stage-root','terminal-json','build-tool','qa-tool','seal-tool','seal-tool-sha256','handoff-tool']:p.add_argument('--'+k)
    a=p.parse_args();before=c.sources();now=dt.datetime.now(dt.timezone.utc).isoformat()
    if a.mode=='record-stage':
        root=Path(a.stage_root);c.need(root.resolve().is_relative_to(c.O),'own executed stage')
        terminal=json.loads(a.terminal_json);c.need(terminal['exit_code']==0 and terminal.get('session_id') is None and terminal['chunk_id'],'real completed executor terminal')
        c.wait(c.ref(root/'worker/ACTUAL_WAIT.json'));c.wait(c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json'))
        q=c.write(root/'ACTUAL_TOOL_CLOSURE_V1.json',dict(schema='astra.report15.actual_tool_closure.v1',status='ACTUAL_TOOL_REAPED0',actual_tool_exit=terminal['exit_code'],session_closed=True,actual_terminal_result=terminal,worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),outer_wait=c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json'),source_before_after=before,source_unchanged=True,recorded_utc=now,metadata_recorder_identity=identity()))
        print(json.dumps({'tool_closure':q}),flush=True)
    elif a.mode=='tools':
        br=c.ref(Path(a.build_tool));qr=c.ref(Path(a.qa_tool));build=stage(Path(a.build_tool).parent,br);qa=stage(Path(a.qa_tool).parent,qr)
        browser=c.ref(c.O/'local_browser01/BROWSER_REPORT_V1.json')
        q=c.write(c.O/'ACTUAL_BUILD_AND_LOCAL_TOOLS_CLOSED_V1.json',dict(schema='astra.report15.actual.local_tools.v1',stages={'build':build,'local-qa':qa},browser_report=browser,source_before_after=before,source_unchanged=True,all_stage_worker_outer_tool_actual0=True,old_media_QA_reused_not_reexecuted=True,metadata_writer_identity=identity()));print(json.dumps({'tools':q}),flush=True)
    elif a.mode=='handoff':
        sq={'path':a.seal_tool,'sha256':a.seal_tool_sha256};seal=stage(Path(a.seal_tool).parent,sq);candidate=c.ref(c.O/'LOCAL_CLOSED_CANDIDATE_V15.json');j=c.read(candidate)
        c.need(j['status']=='CLOSED_LOCAL_REPORT_CANDIDATE_PENDING_DISTINCT_SCIENTIFIC_REVIEW' and not j['publish_performed'],'local candidate not scientific/public pass')
        ids=list(j['registered_CPU_identities'])
        for executed in j['actual_CPU_stages'].values():ids.append(c.read(executed['tool_closure'])['metadata_recorder_identity'])
        ids.append(c.read(c.ref(c.O/'ACTUAL_BUILD_AND_LOCAL_TOOLS_CLOSED_V1.json'))['metadata_writer_identity'])
        ids.append(c.read(sq)['metadata_recorder_identity'])
        for key in ['worker_wait','outer_wait']:ids.extend(c.wait(seal[key])['owned_identities'])
        c.need(all(c.absent(i) for i in ids),'actual prior own stage identities fresh absent')
        hand=dict(schema='astra.report15.parent_handoff.v1',status='LOCAL_CANDIDATE_CLOSED_PENDING_DISTINCT_SCIENTIFIC_REVIEW',candidate=candidate,HTML=j['HTML'],plan=j['plan'],facts=j['facts'],manifest=j['manifest'],source_graph=j['source_graph'],snapshot_utc=j['snapshot_utc'],snapshot_Asia_Shanghai=j['snapshot_Asia_Shanghai'],base_public_commit=c.HEAD,actual_local_modes=4,actual_local_language_detail_combinations=16,local_qa_scope='NEW_TEXT_INCREMENT_ONLY_IMAGES_BLOCKED',old_media_QA_reused=j['old_media_QA_reused'],actual_build_local_stages=j['actual_CPU_stages'],actual_seal_stage=seal,all_scoped_prior_owned_fresh_absent=True,registered_prior_CPU_identities=ids,metadata_writer_identity=identity(),current_handoff_stage_terminal='REQUIRES_EXTERNAL_TRUE_WAIT_AND_TOOL_FINAL_SUPPLEMENT',final_metadata_closure_path=str(c.O/'FINAL_METADATA_ACTUAL_TOOL_CLOSED_V15.json'),new_original_images=0,new_figures=0,inherited_original_selected_RGB=480,inherited_scientific_figures=2,old8fig15RGB_preserved=True,original_PNG_NPY_NPZ_reads=0,V12_reader_exits=[2,2,2],V12_all13_committed_rows_missing=True,V12_physical_advance='UNKNOWN_PRODUCER_ABSENT',bank_count=0,all_requirements_completed=False,safety_acceptance=False,scientific_peer=None,reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,reviewer_model_declaration=None,publish_performed=False,publication_authorized_by_this_artifact=False,historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False,source_before_after=before,source_unchanged=True,created_utc=now)
        q=c.write(c.O/'PARENT_HANDOFF_V15.json',hand);print(json.dumps({'handoff':q,'candidate':candidate}),flush=True)
    else:
        hq=c.ref(c.O/'PARENT_HANDOFF_V15.json');h=c.read(hq);tq=c.ref(Path(a.handoff_tool));hs=stage(Path(a.handoff_tool).parent,tq)
        ids=list(h['registered_prior_CPU_identities'])+[h['metadata_writer_identity'],c.read(tq)['metadata_recorder_identity']]
        for k in ['worker_wait','outer_wait']:ids.extend(c.wait(hs[k])['owned_identities'])
        c.need(all(c.absent(i) for i in ids),'actual all scoped worker/outer/browser/handoff identities fresh absent')
        c.need(c.sources()==before and h['source_before_after']==before,'actual final source unchanged')
        obj=dict(schema='astra.report15.final.actual_metadata_tool_closed.v1',status='CURRENT_LOCAL_CANDIDATE_REQUIREMENTS_CLOSED_PENDING_DISTINCT_SCIENTIFIC_REVIEW',parent_handoff=hq,candidate=h['candidate'],HTML=h['HTML'],facts=h['facts'],plan=h['plan'],source_graph=h['source_graph'],current_handoff_stage=hs,current_metadata_writer_fresh_absent=True,all_scoped_registered_owned_fresh_absent=True,registered_scoped_identities=ids,source_before_after=before,source_unchanged=True,local_actual_modes=4,local_EN_ZH_detail_summary_combinations=16,old_media_QA_reused=h['old_media_QA_reused'],old_media_not_redecoded=True,current_build_local_seal_handoff_worker_outer_tools_actual0=True,historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False,scientific_peer=None,publication_performed=False,publication_authorized_by_this_artifact=False,bank_count=0,physical_acceptance=False,native_GPU_SDK_Git_W_actions=0,final_confirmation_identity=identity(),final_confirmation_true_tool_terminal='TO_BE_INSERTED_AFTER_THIS_EXECUTOR_ACTUALLY_RETURNS',fresh_absence_checked_utc=now)
        print(json.dumps(obj,ensure_ascii=False),flush=True)
    c.need(c.sources()==before,'metadata source unchanged')

if __name__=='__main__':main()
