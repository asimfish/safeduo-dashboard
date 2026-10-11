"""A small partial overlay; inherit every original gallery/asset without opening it."""
import argparse,copy,datetime,json,re
from html.parser import HTMLParser
from pathlib import Path
import common_v1 as c
from render_v1 import render,STYLE
from information_architecture_v1 import compose
from current_facts_v1 import add_current_facts

def build(out):
    sources=c.sources();packet=c.read(c.ref(c.D/'FROZEN_INPUTS_V1.json'))
    refs=packet['refs'];docs={k:c.read(q) for k,q in refs.items() if k!='base_HTML'}
    n=docs['V14_parent'];host=docs['V14_HOST_parent'];cp=docs['checkpoint_author'];peer=docs['checkpoint_peer'];sa=docs['sigaction_author'];sp=docs['sigaction_peer']
    c.need([n[k] for k in ('native_actual_exit','raw_native_wait_status','CPU_check_actual_exit','outer_actual_exit','actual_parent_tool_exit')]==[-11,139,0,1,1],'actual failed V14 termination')
    c.need(n['registered_owned_count']==82 and n['owned_counts']==dict(native=80,CPU=1,outer=1,campaign_raw_rows=82),'82 campaign, not82 native')
    c.need(n['all_owned_fresh_absent'] and n['native_error'] is n['native_resource_abort'] is None and n['native_supervisor_signals']==[],'closed recorded natural failure')
    c.need(n['partial_original_png_files']==399 and n['partial_allocated_environments']==11 and n['durable_camera_state_files']==0,'partial image/state scope')
    c.need(len(n['committed_numeric_rows'])==1 and n['committed_numeric_rows'][0]['file_name']=='row_000_complete.json' and n['entry'] is n['producer'] is n['profile_close'] is None,'one initial marker, no full producer')
    c.need(host['native_exit']==-11 and host['root_cause']=='UNKNOWN' and host['writer_cost_and_queue_metrics']=='UNKNOWN_UNREACHED' and host['durable_milestone_elapsed']=='UNKNOWN_NO_MONOTONIC_ORIGIN' and host['NPZ_overlap_causal_gain']=='UNKNOWN','HOST unknown does not prove speedup/cause')
    c.need(cp['checks_count']==23 and cp['actual_worker_outer_tool_exits']==[0,0,0] and peer['checks_count']==45 and peer['status']=='COND_PASS' and peer['scope']=='SOURCE_ONLY' and peer['author_handoff']==refs['checkpoint_author'] and peer['blocking_findings']==[],'checkpoint SOURCE-only23+45')
    c.need(sa['checks_count']==3 and sa['native_adoption'] is False and sp['status']=='COND_PASS' and sp['scope']=='SOURCE_ONLY' and sp['blocking_findings']==[] and sp['author_handoff']==refs['sigaction_author'] and sp['independent_focused_delta_checks']==1 and sp['inherited_five_PASS_checks']==5,'sigaction3+1 inherits5 SOURCE-only')
    c.need(docs['public16']['public_commit']==packet['base_public_commit'] and docs['public16']['historical71948_tool_terminal']=='UNKNOWN_PERMANENT' and docs['public16_external']['confirmation_writer_actual_tool_exit']==0,'closed public16 plus real final writer terminal')
    adopted=docs['public16_distinct_MAIN'];review=docs['V15_full_SOURCE_review'];whole=docs['V15_full_SOURCE_peer'];author=docs['V15_author_SOURCE'];commands=docs['V15_commands']
    c.need(adopted['status']=='PARENT_CURRENT_REPORT16_PUBLICATION_ADOPTED' and adopted['public_commit']==packet['base_public_commit'] and adopted['all_current_worker_outer_tool_metadata_exits_zero'] and adopted['history_71948']=='UNKNOWN_PERMANENT' and adopted['all_history_tools_exit_zero'] is False,'MAIN distinct public16 adoption, historical unknown preserved')
    c.need(review['status']==whole['status']=='NEEDS_FIX' and review['scope']==whole['scope']=='SOURCE_ONLY' and whole['review']==refs['V15_full_SOURCE_review'] and whole['author_handoff']==review['author_handoff']==refs['V15_author_SOURCE'],'actual whole V15 SOURCE review, not native acceptance')
    findings=review['blocking_findings'];nonblocking=review['nonblocking_findings']
    c.need(len(findings)==2 and {x['id'] for x in findings}=={'F1','F2'} and all(x['severity']=='HIGH' for x in findings) and whole['blocking_findings']==findings,'exact two HIGH full-source blockers')
    c.need(len(nonblocking)==1 and nonblocking[0]['id']=='F3' and nonblocking[0]['field']=='future_complete_static_registration' and whole['nonblocking_findings']==nonblocking,'old static commands are a separate nonblocking review finding')
    c.need(author['commands']==refs['V15_commands'] and author['source_graph']==whole['source_graph']==review['source_graph']==commands['source_graph'],'exact commands/full-source binding')
    contract=review['parent_serial_refresh_contract']
    c.need(contract['minimum_required_wait_refs']==79 and contract['complete_static_raw_identity_rows']==1665 and commands['future_complete_static_registration']!=contract['actual_complete_static_registration'],'reviewed stale 76-ref commands differ from current79/1665 static; neither is a fresh permit')
    c.need(commands['source_only'] is True and commands['native_launch_authorized'] is False and commands['native_executions']==0 and whole['native_runs']==0 and whole['SDK_calls']==0 and whole['native_allowed'] is False and whole['metadata_pending_external_terminal'] is False,'SOURCE closed without native permission or results')
    c.need(whole['source_unchanged'] and author['source_unchanged'] and author['input_unchanged'] and author['production_SDK_native_executions']==0 and author['all_requirements'] is False and author['bank_count']==0,'closed author engineering checks never science pass')
    for name in ('actual_worker_wait','actual_outer_wait'):c.wait(whole[name])
    for name in ('actual_tool_wait','actual_metadata_tool_wait'):c.tool(whole[name])
    oldfacts=docs['base_facts'];baseplan=docs['base_plan'];oldmanifest=docs['base_manifest'];old=c.buffer(refs['base_HTML'],4*1024**2).decode('utf-8')
    c.need(oldfacts['V13']['camera_INITIAL32_PASS'] and oldfacts['V13']['selected_original_images']==480 and oldfacts['V13']['reason_counts_micro1']==dict(nonfinite=0,hard74=18,velocity74=22,soft26=3,raw9021=2,point_scalar82=30,held_target74=0),'inherited V13 current INITIAL/data counts')
    c.need(oldfacts['model_denominators']['cumulative_MODEL_ONLY']==40960 and oldfacts['bank_count']==0,'unchanged model denominators/bank')
    epoch=datetime.datetime.fromisoformat(packet['snapshot_utc']);local=epoch.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    f=dict(schema='astra.report17.closed_metadata_LOCAL_draft_facts.v1',status='SOURCE_ONLY_LOCAL_DRAFT_PENDING_PARENT_COMPLETE_FACTS_AND_DISTINCT_REPORT_SCIENCE',snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=local,historical_snapshot_not_live=True,prefix=c.PREFIX,source_refs=refs,base_public_commit=packet['base_public_commit'],model_denominators=copy.deepcopy(oldfacts['model_denominators']),new_IID_samples=0,V13_initial_acceptance_inherited=copy.deepcopy(oldfacts['V13']),V14=dict(native_exit=-11,raw_wait_status=139,CPU_exit=0,outer_tool=[1,1],elapsed_seconds=n['native_elapsed_seconds'],original_deadline_seconds=1200,partial_original_PNGs=399,allocated_environments=11,completed_camera_states=0,camera_qualification='UNKNOWN_NOT_AUDITED',committed_rows=[0],missing_rows=list(range(1,13)),final_camera='MISSING_UNAUDITED',physical_advance='UNKNOWN_PRODUCER_ABSENT',campaign_owned_counts=n['owned_counts'],failure_cause='UNKNOWN',ENTRY=None,PRODUCER=None,ProfileClose=None,full12=False,related_same32_not_new_IID=True),HOST=dict(root_cause='UNKNOWN',writer_cost_and_queue='UNKNOWN_UNREACHED',durable_milestone_elapsed='UNKNOWN_NO_MONOTONIC_ORIGIN',native_gain='UNKNOWN',GPU_time='NOT_COLLECTED',pre_render_not_complete_capture=True),engineering_only=dict(checkpoint_author_checks=23,checkpoint_independent_SOURCE_checks=45,sigaction_author_checks=3,sigaction_independent_delta_checks=1,sigaction_inherited_PASS_checks=5,not_native_samples=True,crashfix_proven=False,physical_acceptance=False),V15=dict(status='NOT_RUN_IN_THIS_SCOPED_SCIENTIFIC_SNAPSHOT',results=None,excluded_later_or_live_outcomes=True),full12=False,all_mesh=False,CONTACT='UNKNOWN_SDK_REFRESH_EPOCH',actual_drive_peak='UNKNOWN',within_micro_peak='UNKNOWN',wrench='UNKNOWN',world_error_bound='UNKNOWN',bank_count=0,safety_acceptance=False,all_requirements_completed=False,tasks='NOT_RUN',holdout='UNOPENED',historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False,report_RAW_math_or_camera_audit=False,PNG_NPZ_NPY_payload_reads=0,new_media=0,new_figures=0,publish_performed=False,publish_allowed=False,report_science_peer='PENDING',parent_complete_Report17_fact_binding='PENDING',browser_QA='NOT_RUN_SOURCE_DRAFT_SCOPE')
    f['draft_version']=5;f['draft_input_cutoff_UTC']=f['snapshot_utc'];f['final_scientific_cutoff']='PENDING_MAIN_COMPLETE_FACT_BINDING'
    f['information_architecture']='LATEST_FIRST_SEVEN_SECTIONS_WITH_NATIVE_COLLAPSED_HISTORY_NAV'
    f['layout_revision_UTC']=packet['layout_revision_UTC']
    f['public16_distinct_MAIN_adopted']=True
    f['V15'].update(SOURCE_review_status='NEEDS_FIX',SOURCE_scope='SOURCE_ONLY',blocking_HIGH_findings=copy.deepcopy(findings),nonblocking_commands_finding=copy.deepcopy(nonblocking[0]),commands_status='OLD_STATIC_REFERENCE_NOT_CURRENT_EXECUTION_PLAN',commands_static_reference=copy.deepcopy(commands['future_complete_static_registration']),reviewed_current_static_reference=copy.deepcopy(contract['actual_complete_static_registration']),reviewed_current_static_wait_refs=79,reviewed_current_static_raw_rows=1665,static_catalog_is_not_fresh_permit=True,whole_SOURCE_peer_worker_outer_tool_metadata_exits=[0,0,0,0],native_runs=0,native_permission=False,scientific_acceptance=False)
    current_checks=add_current_facts(f,docs,refs)
    out=Path(out);c.need(out.resolve().is_relative_to(c.O) and not out.exists(),'exclusive Report17 draft only');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir();prefix=ov/c.PREFIX
    c.write(prefix/'FACTS_DRAFT_INCREMENT17_V1.json',f)
    for k,q in refs.items():
        if not k.startswith('base_'):
            p=prefix/'evidence'/(k+'.json');p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as w:w.write(c.buffer(q))
            c.need(c.ref(p)['sha256']==q['sha256'],'exact closed evidence bytes')
    for name,q in sources['own_source_files'].items():
        p=prefix/'source'/Path(name).name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as w:w.write(c.buffer(q,2*1024**2))
    p=prefix/'source/SOURCE_GRAPH_V1.json'
    with p.open('xb') as w:w.write(c.buffer(c.ref(c.D/'SOURCE_GRAPH_V1.json')))
    marker='<section id="nativeAttemptIncrement16"';c.need(old.count(marker)==1,'one immutable report16 section')
    page,IA_checks=compose(old,render(f),STYLE,f)
    c.need(page[page.index(marker):]==old[old.index(marker):],'Report16 complete tail and older sections byte exact')
    (ov/'index.html').write_text(page,encoding='utf-8')
    summary=docs['base_summary'];saved=copy.deepcopy(summary['increment16']);summary.update(asof_utc=epoch.isoformat(),asof_AsiaShanghai=local,increment17=f,latest_local_anchor='nativeAttemptIncrement17');c.need(summary['increment16']==saved,'Report16 facts retained');c.write(ov/'summary.json',summary)
    files={p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()}
    manifest=copy.deepcopy(oldmanifest);manifest['files'].update({k:q['sha256'] for k,q in files.items()});manifest.update(snapshot_updated_utc=epoch.isoformat(),increment17=dict(facts=c.PREFIX+'/FACTS_DRAFT_INCREMENT17_V1.json',DRAFT_ONLY=True,bank_count=0,publish_allowed=False));c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    c.need(all(manifest['files'][k]==v for k,v in oldmanifest['files'].items() if k not in ('index.html','summary.json')),'all inherited assets/archives SHA entries retained without payload reads')
    plan=dict(schema='astra.report17.SOURCE_LOCAL_draft_partial_overlay.v1',base_commit=packet['base_public_commit'],base_plan=refs['base_plan'],overlay=str(ov),overlay_kind='PARTIAL_DELTA_RESOLVE_ALL_FILE_SOURCES',file_sources=files,inherited_file_sources=baseplan['file_sources'],changed_files={k:q['sha256'] for k,q in files.items()},manifest=c.ref(ov/'panel_manifest.json'),source_facts=c.ref(prefix/'FACTS_DRAFT_INCREMENT17_V1.json'),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),all_old_manifest_entries_preserved=True,old16_and_earlier_tail_byte_identical=True,new_images=0,new_figures=0,PNG_payload_reads=0,publish_performed=False,publish_allowed=False,parent_complete_fact_binding='PENDING',distinct_report_science_peer='PENDING',actual_browser_QA='NOT_RUN')
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    own=page[page.index('<section id="nativeAttemptIncrement17"'):page.index('<div id="r17HistoricalReports"')]
    class Parse(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.inputs=[];self.labels=[];self.images=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if tag=='input':self.inputs.append(a)
            if tag=='label':self.labels.append(a)
            if tag=='img':self.images.append(a)
    parsed=Parse();parsed.feed(own)
    c.need(len(parsed.ids)==len(set(parsed.ids)) and len(parsed.inputs)==4 and {x['id'] for x in parsed.inputs}=={x['for'] for x in parsed.labels} and not parsed.images,'scoped navigation/control labels, no new images')
    c.need(':focus-visible' in STYLE and 'clip-path:inset(50%)' in STYLE and own.count('class="r17-title-zh"')==own.count('class="r17-title-en"')==1,'accessible hidden radios and separate language titles')
    c.need(all(text in own for text in ('NEEDS_FIX','F1','F2','F3','2 HIGH','79 /1665','76','release11','77','110','198')) and 'V15全包' in own and 'NO_ADMISSION' in own,'historical source blockers, current engineering closure and classified attempt units visible without qualification promotion')
    c.need('r17Criteria' in own and 'r17Metrics' in own and 'r17Failures' in own and 'r17Reproduction' in own,'criteria metrics failures and reproducibility have current direct anchors')
    c.need(c.sources()==sources and all(c.ref(q['path'])==q for q in refs.values()),'own source and exact closed inputs unchanged')
    checks=dict(status='PASS_ACTUAL_READONLY_METADATA_PROJECTION_AND_TEXT_STRUCTURE_ONLY',V14_campaign_82_not_native82=True,V14_camera_qualification_UNKNOWN=True,V13_INITIAL32_not_borrowed=True,SOURCE_checks_not_native_or_crashfix=True,MODEL40960_only=True,old16_tail_byte_exact=True,all_old_asset_manifest_SHA_entries_retained=True,hidden_radio_labels_focus_text_contract=True,language_title_separation=True,new_images=0,media_payload_reads=0,actual_browser_modes_run=0,physical_acceptance=False,publish_performed=False)
    checks.update(V15_release10_historical_SOURCE_NEEDS_FIX_exact_two_HIGH=True,V15_release10_historical_commands_stale_static_reference=True,V15_no_native_or_science_PASS=True,public16_distinct_MAIN_adoption_consumed=True,final_scientific_cutoff_PENDING=True,current_closed_facts=current_checks)
    c.write(out/'CURRENT_CLOSED_FACTS_CHECKS_V1.json',current_checks)
    checks['information_architecture']=IA_checks
    c.write(out/'INFORMATION_ARCHITECTURE_CHECKS_V1.json',IA_checks)
    c.write(out/'PROJECTION_CHECKS_V1.json',checks)
    candidate=dict(schema='astra.report17.SOURCE_LOCAL_draft_handoff.v1',status='REVIEWABLE_SOURCE_ONLY_LOCAL_DRAFT_NOT_PUBLICATION_READY',snapshot_utc=f['snapshot_utc'],base_public_commit=packet['base_public_commit'],HTML=c.ref(ov/'index.html'),facts=plan['source_facts'],plan=c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],checks=c.ref(out/'PROJECTION_CHECKS_V1.json'),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),closed_source_inputs=refs,old16_tail_byte_exact=True,new_media=0,new_figures=0,media_payload_reads=0,browser_QA='NOT_RUN_SOURCE_DRAFT_SCOPE',parent_complete_Report17_facts='PENDING',report_scientific_peer='PENDING',publish_performed=False,publish_allowed=False,historical71948_tool_terminal='UNKNOWN_PERMANENT',bank_count=0,safety_acceptance=False)
    candidate.update(draft_version=5,layout_revision_UTC=f['layout_revision_UTC'],final_scientific_cutoff='PENDING_MAIN_COMPLETE_FACT_BINDING',information_architecture_checks=c.ref(out/'INFORMATION_ARCHITECTURE_CHECKS_V1.json'),current_closed_facts_checks=c.ref(out/'CURRENT_CLOSED_FACTS_CHECKS_V1.json'),entire_historical_main_payload_byte_exact=True)
    c.write(out/'LOCAL_SOURCE_DRAFT_HANDOFF_V17.json',candidate)
    print(json.dumps({'draft':c.ref(out/'LOCAL_SOURCE_DRAFT_HANDOFF_V17.json'),'HTML':candidate['HTML'],'snapshot_utc':f['snapshot_utc'],'changed_paths':len(files),'new_media':0,'browser_QA':'NOT_RUN','publish_performed':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);a=p.parse_args();build(a.out)
