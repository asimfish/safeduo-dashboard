"""Immutable LOCAL V13 DATA projection. Reuses copied current480 without another copy."""
import argparse,copy,datetime as dt,html,json,re
from pathlib import Path
import common_v1 as c
from render_v1 import STYLE,render,gallery_page

def build(out,inventory_ref,gallery_tool_ref):
    before=c.sources();packet=c.read(c.ref(c.D/'FROZEN_INPUTS_V1.json'));s={k:c.read(q) for k,q in packet['refs'].items() if k!='base_HTML'}
    inv=c.read(inventory_ref);tool=c.tool(gallery_tool_ref);c.wait(tool['worker_wait']);c.wait(tool['outer_wait'])
    c.need(inv['status']=='CLOSED_BYTE_EXACT_SELECTED_INITIAL_ORIGINALS_ONLY' and inv['source_unchanged'] is True and inv['data_report']==packet['refs']['V13_DATA_report'] and inv['actual_selected_metadata']==packet['refs']['V13_selected_ledger'],'new actual V13 copied inventory; no oldV10 gallery reuse')
    c.need(inv['copied_original_images']==480 and inv['copied_original_bytes']==184366361 and inv['selected_initial_groups']==32 and len(inv['copies'])==480,'current480 actual bytes copied/readback')
    d=s['V13_DATA_report'];h=s['V13_DATA_handoff'];n=s['V13_parent'];ph=s['V13_DATA_peer_handoff'];pr=s['V13_DATA_peer_review'];pt=s['V13_DATA_peer_tools'];ad=s['V13_DATA_final_adoption']
    c.need(h['data_report']==packet['refs']['V13_DATA_report'] and h['selected_original_PNG_ledger']==packet['refs']['V13_selected_ledger'],'author closed exact current DATA/selected ledger')
    c.need(ph['status']=='PASS_DATA_ONLY_PARTIAL_SCOPE' and ph['blocking_findings']==[] and ph['review']==packet['refs']['V13_DATA_peer_review'] and ph['actual_DATA_result']==packet['refs']['V13_DATA_report'] and ph['current480_selected_ledger']==packet['refs']['V13_selected_ledger'],'actual distinct DATA10 PASS, not report science')
    c.need(pr['status']=='PASS_DATA_ONLY_PARTIAL_SCOPE' and pr['blocking_findings']==[] and len(pr['checks'])==10 and pt['actual_worker_exit']==pt['actual_outer_exit']==pt['actual_tool_exit']==0,'actual semantic10 and true tool closure')
    c.wait(ph['actual_worker_wait']);c.wait(ph['actual_outer_wait']);c.tool(ph['actual_tool_wait'])
    c.need(ad['bank']==0 and ad['safety_acceptance'] is False and ad['independent_handoff']==packet['refs']['V13_DATA_peer_handoff'] and ad['independent_review']==packet['refs']['V13_DATA_peer_review'] and ad['result']==packet['refs']['V13_DATA_report'],'parent final DATA adoption preserves no safety promotion')
    c.need(d['initial_camera_qualified_groups']==32 and d['selected_initial_images_verified']==480 and d['micro1_original_numeric_denied_lanes']==32,'actual INITIAL32 and first-step32DENY')
    c.need(d['original_native_PNGs']==1294 and d['initial_attempt_PNGs_audited']==1201 and d['other_partial_or_orphan_PNGs_preserved_not_pixel_audited']==93,'three current image denominators')
    reasons=d['numeric']['reason_counts_by_micro'][1];c.need(reasons==dict(nonfinite=0,hard74=18,velocity74=22,soft26=3,raw9021=2,point_scalar82=30,held_target74=0),'hard74 jointlimits; overlapping recorded predicates')
    c.need(d['committed_rows']==[0,1] and d['missing_micro_rows']==list(range(2,13)) and d['final_camera']=='MISSING_UNAUDITED' and d['full12_acceptance'] is d['safety_acceptance'] is False and d['bank_count']==0,'incomplete fullprefix/physical safety retained')
    c.need([n[k] for k in ('native_actual_exit','raw_native_wait_status','CPU_check_actual_exit','outer_actual_exit','actual_parent_tool_exit')]==[-9,9,0,1,1] and n['entry'] is n['producer'] is n['profile_close'] is None,'original failed native termination retained')
    base=s['base_candidate'];p0=s['base_plan'];m0=s['base_manifest'];oldfacts=s['base_facts']
    c.need(base['HTML']==packet['refs']['base_HTML'] and base['plan']==packet['refs']['base_plan'] and base['manifest']==packet['refs']['base_manifest'],'exact public15 basis')
    epoch=dt.datetime.fromisoformat(packet['snapshot_utc']);local=epoch.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat();groups=inv['groups']
    facts=dict(schema='astra.report16.actual_partial_facts.v1',status='LOCAL_ACTUAL_PARTIAL_DATA_CANDIDATE_PENDING_DISTINCT_REPORT_SCIENCE',snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=local,historical_snapshot_not_live=True,source_refs=packet['refs'],base_public_commit=c.HEAD,model_denominators=copy.deepcopy(oldfacts['model_denominators']),new_IID_samples=0,V13=dict(camera_INITIAL32_PASS=True,camera_qualification='INITIAL32_ONLY_PASS',independent_camera_DATA='CLOSED_INITIAL32_PIXEL_COVERAGE_ONLY',independent_numeric_DATA='CLOSED_COMMITTED_ROWS01_ONLY',partial_DATA_ref=packet['refs']['V13_DATA_report'],selected_original_images=480,selected_original_bytes=184366361,PNG_count=1294,initial_attempt_PNGs_pixel_audited=1201,unpixel_tail_PNGs=93,state_files=32,committed_rows=[0,1],related_numeric_observations=64,micro1_numeric_denied_lanes=32,reason_counts_micro1=reasons,reason_counts_overlap=True,hard74_meaning='HARD_JOINT_LIMIT_VIOLATIONS_NOT_GEOMETRIC_OVERLAPS',missing_rows=list(range(2,13)),final_camera='MISSING_UNAUDITED',full12_complete=False,full_mesh_coverage_certified=False,physical_advance='UNKNOWN_PRODUCER_ABSENT',ENTRY=None,PRODUCER=None,ProfileClose=None,native_exit=-9,raw_wait_status=9,CPU_exit=0,outer_tool=[1,1],original_deadline_s=1200,elapsed_s=n['native_elapsed_seconds'],registered_counts=n['owned_counts'],failure_cause='UNKNOWN',original27_held='INDEPENDENT_INITIAL_DATA_AUDITED',latest42='UNKNOWN_UNBOUND_NOT_ADOPTED',raw9021_proxy_not_mesh=True),gallery_groups=groups,selected_inventory=inventory_ref,current_gallery_actual_tool=gallery_tool_ref,distinct_DATA_semantic_peer=packet['refs']['V13_DATA_peer_review'],distinct_DATA_semantic_scope='10_CHECKS_PARTIAL_DATA_ONLY_TWO_FIXED_PNG_SPOTCHECKS_NOT_ALL480_REDECODE',parent_complete_DATA_adoption=packet['refs']['V13_DATA_final_adoption'],HOST=dict(matched_calls=6,constructor_returned=True,constructor_s=99.664621304,nested_intervals_not_additive=True,GPU_time='NOT_COLLECTED',pure_IO_time='UNKNOWN',root_cause_fix_proven=False),V10_historical_only=copy.deepcopy(oldfacts['attempts']['V10']),CONTACT='UNKNOWN_SDK_REFRESH_EPOCH',actual_drive_peak='UNKNOWN',within_micro_peak='UNKNOWN',wrench='UNKNOWN',global_error_bound='UNKNOWN',tasks='NOT_RUN',holdout='UNOPENED',bank_count=0,eligible=0,safety_acceptance=False,all_requirements_completed=False,publish_performed=False,publish_allowed=False,distinct_report_science_peer=None,reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,report_new_mask_or_numeric_audit=False,report_NPY_NPZ_reads=0,current_PNG_copies_480_from_original_current_ledger=True,historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False)
    c.need(facts['model_denominators']['cumulative_MODEL_ONLY']==40960,'model denominator unchanged; not native40960')
    out=Path(out);c.need(out.resolve().is_relative_to(c.O) and not out.exists(),'new immutable LOCAL build');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir();prefix=ov/c.PREFIX
    for k,q in packet['refs'].items():
        if k.startswith('V13_'):c.copy(q,prefix/'evidence'/(k+'.json'))
    for name,q in before.items():c.copy(q,prefix/'source'/name)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),prefix/'source/SOURCE_GRAPH_V1.json');c.copy(base['HTML'],prefix/'archive_report15/index.html')
    c.copy(inventory_ref,prefix/'SELECTED_CURRENT_V13_ORIGINAL_INVENTORY_V1.json');c.copy(h['numeric_CSV'],prefix/'COMMITTED_2X32_NUMERIC_V1.csv')
    files={}
    copied={r['original']['path']:r for r in inv['copies']}
    for g in groups:
        for im in g['images']:
            r=copied[im['original']['path']];c.need(r['original']==im['original'] and r['copied']['sha256']==im['original']['sha256'],'actual current copied PNG mapping')
            name=c.PREFIX+'/'+im['relative'];c.need(name not in files,'unique current gallery destination');files[name]=r['copied']
        pp=prefix/'gallery'/f'env_{g["lane"]:03d}.html';pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text(gallery_page(g,c.PREFIX),encoding='utf-8')
    c.need(len(files)==480,'480 partial overlay refs reuse already-verified copied bytes; no duplicate image copy')
    c.write(prefix/'FACTS_INCREMENT16_V1.json',facts)
    old=c.buffer(base['HTML'],4*1024**2).decode();marker='<section id="nativeAttemptIncrement15">';c.need(old.count(marker)==1,'one historical15 marker')
    page=re.sub(r'<p class="asof" id="reportAsOf">.*?</p>','<p class="asof" id="reportAsOf">REPORT16 · '+epoch.isoformat()+' UTC / Asia/Shanghai '+local+' · LOCAL固定科学快照 / LOCAL fixed scientific snapshot.</p>',old,count=1,flags=re.S)
    page=page.replace('</head>','<style id="report16ModesV1">'+STYLE+'</style></head>',1).replace(marker,render(facts,c.PREFIX)+marker,1)
    page=page.replace('<a href="#nativeAttemptIncrement15">最新报告15 / Latest Report15</a>','<a href="#nativeAttemptIncrement16">最新报告16 / Latest Report16</a><a href="#nativeAttemptIncrement15">历史报告15 / Historical Report15</a>',1)
    page=page.replace('<a href="#nativeAttemptIncrement15">最新初始化阶段 / Latest initialization stage</a>','<a href="#nativeAttemptIncrement16">最新覆盖与首步结果 / Latest coverage and first-step results</a><a href="#nativeAttemptIncrement15">历史报告15 / Historical Report15</a>',1)
    page=page.replace('<div class="executive">','<div class="executive"><p id="native16Executive"><strong>V13初始32组覆盖已核验；首步32条数值拒绝，完整安全未通过。V13 initial32 coverage checked;32 first-step numerical denials, full safety unaccepted.</strong> <a href="#nativeAttemptIncrement16">REPORT16 LOCAL</a></p>',1)
    nav_counts={}
    for kind in ('sitebar','toc'):
        blocks=re.findall(r'<nav[^>]*class="'+kind+r'"[^>]*>.*?</nav>',page,re.S)
        c.need(len(blocks)==1,'one scoped '+kind+' nav')
        nav_counts[kind]=blocks[0].count('href="#nativeAttemptIncrement16"')
        c.need(nav_counts[kind]==1,'one latest16 link in '+kind)
    c.need(page[page.index(marker):]==old[old.index(marker):],'all original report15 and earlier section bytes retained')
    (ov/'index.html').write_text(page,encoding='utf-8');summary=c.read(p0['file_sources']['summary.json']);saved=copy.deepcopy(summary['increment15']);summary.update(asof_utc=epoch.isoformat(),asof_AsiaShanghai=local,increment16=facts,latest_local_anchor='nativeAttemptIncrement16');c.need(summary['increment15']==saved,'historical15 facts unchanged');c.write(ov/'summary.json',summary)
    for p in ov.rglob('*'):
        if p.is_file():files[p.relative_to(ov).as_posix()]=c.ref(p)
    manifest=copy.deepcopy(m0);manifest['files'].update({k:q['sha256'] for k,q in files.items()});manifest.update(snapshot_updated_utc=epoch.isoformat(),increment16=dict(facts=c.PREFIX+'/FACTS_INCREMENT16_V1.json',INITIAL32_PASS=True,full12=False,bank_count=0,publish_allowed=False));c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    c.need(all(manifest['files'][k]==v for k,v in m0['files'].items() if k not in ('index.html','summary.json')),'all inherited media/archive SHA entries unchanged')
    plan=dict(schema='astra.report.increment16.actual_partial_overlay.v1',base_commit=c.HEAD,base_plan=packet['refs']['base_plan'],overlay=str(ov),overlay_kind='PARTIAL_DELTA_RESOLVE_ALL_FILE_SOURCES',file_sources=files,inherited_file_sources=p0['file_sources'],changed_files={k:q['sha256'] for k,q in files.items()},manifest=c.ref(ov/'panel_manifest.json'),source_facts=c.ref(prefix/'FACTS_INCREMENT16_V1.json'),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),selected_image_inventory=c.ref(prefix/'SELECTED_CURRENT_V13_ORIGINAL_INVENTORY_V1.json'),current_gallery_copy_tool=gallery_tool_ref,old_media_QA=oldfacts['old_media_QA'],all_old_manifest_entries_preserved=True,old15_and_earlier_tail_byte_identical=True,new_images=480,new_image_bytes=184366361,new_figures=0,report_raw_pixel_or_numeric_reaudit=False,publish_performed=False,publish_allowed=False,distinct_report_science_peer='PENDING')
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan);c.write(out/'PROJECTION_CHECKS_V1.json',dict(status='PASS_ACTUAL_CLOSED_DATA_PROJECTION_AND_ORIGINAL_BYTE_REUSE_ONLY',current_INITIAL32_PASS=True,related_rows=64,denied_micro1=32,reason_counts_overlap=True,hard74_means_joint_limits=True,unpixel_tail93=True,full12=False,bank0=True,CONTACT_epoch_UNKNOWN=True,all_old15_tail_byte_identical=True,all_old_manifest_entries_preserved=True,duplicate_current_PNG_copy_bytes=0,no_old_media_opened=True,no_new_RAW_or_SDK=True,publish_performed=False));c.need(c.sources()==before,'candidate source unchanged');print(json.dumps(dict(plan=c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),current_images=480,actual_image_bytes=184366361,snapshot_utc=facts['snapshot_utc'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('out','gallery-inventory','gallery-inventory-sha256','gallery-tools','gallery-tools-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();build(a.out,dict(path=a.gallery_inventory,sha256=a.gallery_inventory_sha256),dict(path=a.gallery_tools,sha256=a.gallery_tools_sha256))
