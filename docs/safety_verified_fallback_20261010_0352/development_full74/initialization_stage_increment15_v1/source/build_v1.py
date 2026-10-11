"""Project authenticated closed receipts only; never open old image/numeric buffers."""
import argparse,copy,datetime as dt,json,re
from pathlib import Path
import common_v1 as c
from render_v1 import STYLE,render

def validate_scope(f):
    c.need(f['model_denominators']['cumulative_MODEL_ONLY']==40960,'separate unchanged model denominator')
    c.need(f['new_IID_native_samples']==0 and f['bank_count']==0 and f['safety_acceptance'] is False,'no new IID or safety promotion')
    v=f['attempts']['V12']
    c.need(v['committed_rows']==[] and v['missing_rows']==list(range(13)),'V12 all committed rows missing')
    c.need(v['physical_advance']=='UNKNOWN_PRODUCER_ABSENT' and v['failure_cause']=='UNKNOWN','missing records do not prove physical zero or driver fault')
    c.need(v['camera_assessment']=='NOT_REACHED' and v['camera_all32_failed'] is None,'camera stage not reached is not all32 failure')
    c.need(v['PNG_count']==0 and v['state_receipt_count']==0 and v['full12_complete'] is False,'before env return zero recorded data')
    c.need(f['attempts']['V11']['camera_assessment']=='UNKNOWN_UNASSESSED','V11 receipts-only does not qualify pixels')
    c.need(f['contact_SDK_epoch']=='UNKNOWN' and f['actual_drive_peak']=='UNKNOWN' and f['global_error_bound']=='UNKNOWN','physical unknowns preserved')

def native(n):
    c.need([n[k] for k in ['native_actual_exit','raw_native_wait_status','CPU_check_actual_exit','outer_actual_exit','actual_parent_tool_exit']]==[-9,9,0,1,1],'genuine failed native status chain')
    c.need(n['native_work_deadline_seconds']==1200 and n['entry'] is None and n['producer'] is None and n['profile_close'] is None,'original deadline and absent durable producer')
    c.need(n['source_unchanged'] is True and n['all_owned_fresh_absent'] is True,'closed original source/ownership scope')
    for k,ex,clean in [('native_wait',-9,False),('CPU_check_wait',0,True),('outer_wait',1,False)]:c.wait(n[k],ex,clean)
    c.tool(n['tool_closure'],1)

def consumer(h):
    c.need(h['status']=='CLOSED_FAILED_NATIVE_RECEIPTS_ONLY' and h['actual_exit_codes']=={'worker':2,'outer':2,'tool':2},'expected failure reader true 2/2/2')
    c.need(h['all_owned_closed'] is True and h['source_unchanged'] is True and h['actual_RAW_read'] is False,'receipt-only consumer scope')
    c.wait(h['actual_worker_wait'],2);c.wait(h['actual_outer_wait'],2);c.tool(h['actual_tool_wait'],2)

def load():
    packet=c.read(c.ref(c.D/'FROZEN_INPUTS_V1.json'));s={k:c.read(q) for k,q in packet['refs'].items()}
    base=s['candidate14'];plan=s['base_plan'];manifest=s['base_manifest'];b=s['base_facts'];ad=s['public14_parent_adoption']
    c.need(base['plan']==packet['refs']['base_plan'] and base['manifest']==packet['refs']['base_manifest'] and base['facts']==packet['refs']['base_facts'],'exact old candidate plan/facts')
    c.need(ad['current_public_delivery_complete'] is True and ad['commit']==c.HEAD and ad['historical71948_tool_terminal']=='UNKNOWN_PERMANENT','current public base and permanent historical unknown')
    c.need(ad['actual_worker_outer_tool']==[0,0,0] and ad['all_historical_tools_zero'] is False,'current closure is not all historical tools0')
    qa=s['base_local_qa'];c.need(qa['status']=='PASS_LOCAL_PRESENTATION_ONLY' and len(qa['modes'])==4 and not qa['errors'],'reuse exact completed old-media local QA')
    c.need(b['camera_INITIAL32_PASS'] is True and b['selected_original_images']==480 and b['committed_rows']==[0,1] and b['numerical_DENY_micro1_lanes']==32,'inherited V10 partial DATA scope')
    for k in ['V11_parent','V12_parent']:native(s[k])
    v11=s['V11_parent'];v12=s['V12_parent'];a11=s['V11_reader_adoption'];h11=s['V11_reader_handoff'];h12=s['V12_reader_handoff'];a12=s['V12_reader_adoption'];sem=s['V12_reader_semantic'];final=s['V12_reader_final']
    c.need(a11['consumer_handoff']==packet['refs']['V11_reader_handoff'] and a11['actual_reader_exits']=={'worker':2,'outer':2,'tool':2},'actual V11 parent consumption')
    c.need(a11['actual_RAW_read'] is False and a11['actual_independent_pixel_qualification'] is False and a11['source_and_input_unchanged'] is True,'V11 parent scopes')
    consumer(h11);consumer(h12)
    c.need(h12['actual_result']==packet['refs']['V12_reader_result'] and h12['semantic_result']==packet['refs']['V12_reader_semantic'],'V12 actual result and semantic exact chain')
    c.need(a12['handoff']==packet['refs']['V12_reader_handoff'] and a12['final_tools']==packet['refs']['V12_reader_final'] and a12['semantic_result']==packet['refs']['V12_reader_semantic'],'V12 exact parent adoption')
    c.need(a12['actual_exit_codes']=={'worker':2,'outer':2,'tool':2} and a12['source_input_unchanged'] and a12['all_owned_fresh_absent'],'V12 adoption source and true expected failures')
    c.need(final['ACTUAL_CLOSED_HANDOFF']==packet['refs']['V12_reader_handoff'] and final['actual_reader_exit_codes']=={'worker':2,'outer':2,'tool':2},'final reader tools agree')
    c.need(final['actual_metadata_sealer_terminal_result']['exit_code']==0 and final['metadata_sealer_fresh_absent'] is True and c.absent(final['metadata_sealer_identity']),'real metadata0 plus fresh absence')
    c.need(sem['errors']==[] and sem['original_reader_result']==packet['refs']['V12_reader_result'] and sem['parent_native_closure']==packet['refs']['V12_parent'],'exact semantic source, errors empty')
    c.need(sem['missing_required_committed_rows']==list(range(13)) and sem['durable_committed_numeric_rows']==[] and sem['full12_complete'] is False and sem['physical_advance']=='UNKNOWN_PRODUCER_ABSENT','missing13 rows preserve unknown physics')
    c.need(sem['actual_RAW_read'] is False and sem['original_PNG_read'] is False and sem['SDK_calls']==0 and sem['native_runs']==0 and sem['bank_count']==0,'no raw/native/pixel promotion')
    c.need(v11['partial_original_png_files']==1112 and v11['durable_camera_state_files']==0 and [r['file_name'] for r in v11['committed_numeric_rows']]==['row_000_complete.json'],'V11 actual partial metadata')
    c.need(v12['partial_original_png_files']==0 and v12['durable_camera_state_files']==0 and v12['committed_numeric_rows']==[],'V12 actual zero recorded data')
    init=s['V12_init_refresh'];refresh=s['V12_refresh_receipt']
    c.need(init['actual_native_closure']==packet['refs']['V12_parent'] and init['actual_refresh_receipt']==packet['refs']['V12_refresh_receipt'] and init['camera_stage_reached'] is False,'V12 before environment returned')
    age=refresh['timing']['renewed_age_after_full_verification_s'];c.need(age==2.724541187286377 and age<=refresh['freshness_seconds']==30 and refresh['receipt_is_not_a_native_permit'] is True,'actual refreshed age, original gate, not permit')
    c.need(refresh['SDK_calls']==0 and refresh['native_launched_by_refresh'] is False,'refresh alone launches no SDK/native')
    return packet,s,base,plan,manifest,b,age

def build(out):
    before=c.sources();packet,s,base,plan0,manifest0,b,age=load();out=Path(out)
    c.need(out.resolve().is_relative_to(c.O) and not out.exists(),'new exclusive overlay');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir();prefix=ov/c.PREFIX
    epoch=dt.datetime.fromisoformat(packet['snapshot_utc']);local=epoch.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()
    facts=dict(schema='astra.report.increment15.frozen_facts.v1',snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=local,historical_snapshot_not_live=True,source_refs=packet['refs'],packet=c.ref(c.D/'FROZEN_INPUTS_V1.json'),base_public_commit=c.HEAD,model_denominators=b['model_denominators'],new_IID_native_samples=0,attempts={},contact_SDK_epoch='UNKNOWN',actual_drive_peak='UNKNOWN',global_error_bound='UNKNOWN',bank_count=0,safety_acceptance=False,all_requirements_completed=False,tasks='NOT_RUN',holdout='UNOPENED',full12_accepted_lanes=0,old_original_RGB=15,selected_original_RGB=480,old_figures=8,inherited_new_figures=2,new_figures=0,new_images=0,RAW_PNG_NPY_NPZ_reads=0,old_media_QA=packet['refs']['base_local_qa'],old_media_QA_reused_not_reexecuted=True,physical_acceptance=False,historical71948_tool_terminal='UNKNOWN_PERMANENT',publish_performed=False,distinct_report_science_peer='PENDING',reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,reviewer_model_declaration=None)
    facts['attempts']['V10']=dict(committed_rows=[0,1],missing_rows=list(range(2,13)),PNG_count=1312,initial_attempts_pixel_audited=1201,other_PNG_not_pixel_accepted=111,state_receipt_count=32,selected_originals=480,initial_qualified_groups=32,camera_assessment='INITIAL_ONLY_PASS',camera_final='MISSING_UNAUDITED',recorded_micro1_DENY=32,overlapping_reason_counts=b['recorded_reason_counts'],contact_SDK_epoch='UNKNOWN',full12_complete=False,physical_advance='UNKNOWN_PRODUCER_ABSENT',source=packet['refs']['base_facts'])
    for name in ['V11','V12']:
        n=s[name+'_parent'];facts['attempts'][name]=dict(committed_rows=[0] if name=='V11' else [],missing_rows=list(range(1 if name=='V11' else 0,13)),PNG_count=n['partial_original_png_files'],state_receipt_count=n['durable_camera_state_files'],allocated_environment_count=n['partial_allocated_environments'],camera_assessment='UNKNOWN_UNASSESSED' if name=='V11' else 'NOT_REACHED',camera_all32_failed=None,full12_complete=False,physical_advance='UNKNOWN_PRODUCER_ABSENT',failure_cause='UNKNOWN',native_exit=-9,raw_wait_status=9,CPU_exit=0,outer_exit=1,tool_exit=1,elapsed_s=n['native_elapsed_seconds'],original_native_deadline_s=1200,original_signals=n['native_supervisor_signals'],error=n['native_error'],resource_abort=n['native_resource_abort'],registered_identity_count=n['registered_owned_count'],registered_identity_scope='CAMPAIGN_NATIVE_CPU_OUTER',ENTRY=None,PRODUCER=None,ProfileClose=None,reader_worker_outer_tool=[2,2,2],reader_scope='CLOSED_FAILED_NATIVE_RECEIPTS_ONLY',raw_and_pixel_qualification=False)
    facts['attempts']['V12'].update(environment_initialization_returned=False,fresh_parent_registration_age_s=age,original_freshness_gate_s=30,near_deadline_NVML_query_remaining_s=0.008113730000331998,driver_cause_proven=False,native_pixel_gain='NOT_MEASURED_CAMERA_STAGE_NOT_REACHED',parent_reader_adoption=packet['refs']['V12_reader_adoption'])
    validate_scope(facts)
    checks={}
    for key,value in [('camera_all32_failed',True),('physical_advance','ZERO'),('failure_cause','DRIVER_FAULT')]:
        bad=copy.deepcopy(facts);bad['attempts']['V12'][key]=value
        try:validate_scope(bad)
        except ValueError:checks[key]='REJECTED'
        else:raise ValueError('scope promotion accepted '+key)
    # Copy only the pinned small evidence packets. The full refresh history is referenced,
    # not recursively copied or reopened; original media/numerical buffers remain untouched.
    for name in ['V11_parent','V11_reader_adoption','V12_parent','V12_init_refresh','V12_reader_semantic','V12_reader_handoff','V12_reader_final','V12_reader_adoption','public14_parent_adoption']:
        c.copy(packet['refs'][name],prefix/'evidence'/(name+'.json'))
    c.write(prefix/'evidence/V12_REFRESH_COMPACT_V1.json',dict(schema='astra.report15.refresh.compact_projection.v1',original_receipt=packet['refs']['V12_refresh_receipt'],timing_field='timing.renewed_age_after_full_verification_s',fresh_age_s=age,original_freshness_gate_s=30,original_closed_native=packet['refs']['V12_parent'],native_measurement_or_safety_permit=False,SDK_calls=0,history_redecoded=False))
    for k,q in before.items():c.copy(q,prefix/'source'/k)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),prefix/'source/SOURCE_GRAPH_V1.json')
    for name,q in [('index.html',base['HTML']),('summary.json',plan0['file_sources']['summary.json']),('panel_manifest.json',base['manifest'])]:c.copy(q,prefix/'archive_report14'/name)
    c.write(prefix/'FACTS_INCREMENT15_V1.json',facts)
    page=c.buffer(base['HTML'],4*1024**2).decode();old=page;marker='<section id="nativeAttemptIncrement14">';c.need(page.count(marker)==1,'one old14 marker')
    page=re.sub(r'<p class="asof" id="reportAsOf">.*?</p>','<p class="asof" id="reportAsOf">REPORT15 · '+epoch.isoformat()+' UTC / Asia/Shanghai '+local+' · 固定科学快照，非实时状态 / Fixed scientific snapshot, not live status.</p>',page,count=1,flags=re.S)
    page=page.replace('</head>','<style id="report15ModesV1">'+STYLE+'</style></head>',1).replace(marker,render(facts,c.PREFIX)+marker,1)
    page=page.replace('<nav class="sitebar" aria-label="站点导航">','<nav class="sitebar" aria-label="站点导航"><a href="#nativeAttemptIncrement15">最新报告15 / Latest Report15</a>',1)
    page=page.replace('<a href="#nativeAttemptIncrement14">初始覆盖与首步验收 / Initial coverage and first-step checks</a>','<a href="#nativeAttemptIncrement15">最新初始化阶段 / Latest initialization stage</a><a href="#nativeAttemptIncrement14">历史报告14：初始覆盖与首步验收 / Historical initial coverage and first-step checks</a>',1)
    page=page.replace('<div class="executive">','<div class="executive"><p id="native15Executive"><strong>最新重试未走出环境初始化，完整12步仍未完成；原初始相机验收与首步数值拒绝分别保留。Latest retry did not complete environment initialization; initial camera coverage and first-step numerical denials keep their separate scopes.</strong> <a href="#nativeAttemptIncrement15">REPORT15</a></p>',1)
    c.need(page[page.index(marker):]==old[old.index(marker):],'old14 and entire earlier tail byte-identical')
    (ov/'index.html').write_text(page,encoding='utf-8')
    summary=c.read(plan0['file_sources']['summary.json']);old_increment=copy.deepcopy(summary['increment14']);summary.update(asof_utc=epoch.isoformat(),asof_AsiaShanghai=local,increment15=facts,historical_candidate14_summary=c.PREFIX+'/archive_report14/summary.json')
    c.need(summary['increment14']==old_increment,'old14 summary values unchanged');c.write(ov/'summary.json',summary)
    files={p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()};manifest=copy.deepcopy(manifest0);manifest['files'].update({n:q['sha256'] for n,q in files.items()});manifest.update(snapshot_updated_utc=epoch.isoformat(),increment15=dict(facts=c.PREFIX+'/FACTS_INCREMENT15_V1.json',bank_count=0,safety_acceptance=False));c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    preserved={n:d for n,d in manifest0['files'].items() if n not in ('index.html','summary.json')};c.need(all(manifest['files'][n]==d for n,d in preserved.items()),'all prior media/archives manifest entries unchanged')
    plan=dict(schema='astra.report.increment15.overlay_plan.v1',base_commit=c.HEAD,base_plan=packet['refs']['base_plan'],base_manifest=packet['refs']['base_manifest'],overlay=str(ov),overlay_kind='PARTIAL_DELTA_RESOLVE_FILE_SOURCES',file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},file_bytes={n:Path(q['path']).stat().st_size for n,q in files.items()},inherited_files=preserved,before_sha256={n:(base['manifest']['sha256'] if n=='panel_manifest.json' else manifest0['files'][n]) for n in files if n=='panel_manifest.json' or n in manifest0['files']},manifest=c.ref(ov/'panel_manifest.json'),source_facts=c.ref(prefix/'FACTS_INCREMENT15_V1.json'),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),old8fig_and15RGB_SHA_preserved=True,inherited_original_INITIAL_RGB=480,inherited_scientific_figures=2,new_science_figures=0,new_original_RGB=0,media_buffers_read_or_copied=False,old_media_QA=packet['refs']['base_local_qa'],RAW_reaudit=False,publish_performed=False,scientific_peer='PENDING_DISTINCT_REPORT15_PEER')
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    c.write(out/'PROJECTION_CHECKS_V1.json',dict(status='PASS_METADATA_PROJECTION_ONLY',scope_negative_cases=checks,all_prior_manifest_entries_preserved=True,old14_section_tail_byte_identical=True,old14_summary_unchanged=True,RAW_PNG_NPY_NPZ_reads=0,new_images_and_figures=0,old_media_QA_reused=packet['refs']['base_local_qa']))
    c.need(c.sources()==before,'source unchanged after build');print(json.dumps({'built_plan':c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),'new_metadata_paths':len(files),'media_buffers_read':0}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);build(p.parse_args().out)
