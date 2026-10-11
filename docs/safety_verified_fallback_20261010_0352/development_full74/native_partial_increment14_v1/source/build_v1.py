"""Report14 immutable metadata overlay and byte-exact selected initial gallery."""
import argparse,copy,csv,datetime as dt,html,io,json,math,re
from pathlib import Path
import common_v1 as c
from render_v1 import STYLE,render,gallery_page
from figures_v1 import figures
REASONS=('nonfinite','hard74','velocity74','soft26','raw9021','point_scalar82','held_target74')
def validate_scope(current):
    c.need(current['initial_camera_qualified_groups']==32 and current['selected_original_images']==480,'initial32/480 distinct from fullprefix')
    c.need(current['raw_initial_original_PNGs_audited']==1201 and current['all_native_original_PNGs']==1312 and current['other_PNGs_preserved_without_pixel_acceptance']==111,'pixel-audit denominators')
    c.need(current['micro1_numeric_denied_lanes']==32 and current['reason_counts_micro1']==dict(zip(REASONS,[0,18,22,3,2,30,0])),'recorded predicate counts')
    c.need(current['micro1_contact_refresh_freshness']=='UNKNOWN_NO_INDEPENDENT_SDK_REFRESH_EPOCH_EVIDENCE','producer fresh label is not SDK epoch')
    c.need(current['bank_count']==0 and current['safety_acceptance'] is False and current['full12'] is False,'no physical/safety/fullprefix promotion')
def load_inputs():
    q=c.ref(c.D/'FROZEN_INPUTS_V1.json');p=c.read(q);s={k:c.read(v) for k,v in p['refs'].items()}
    h=s['V10_DATA_handoff'];validate_scope(h['current']);c.need(h['all_original_input_SHA_unchanged'] and h['all_selected_source_SHA_unchanged'],'closed audit original/source integrity')
    for item in [h,*h['part_actual_closures']]:
        c.wait(item['worker_wait']);c.wait(item['outer_wait']);c.tool(item['actual_tool_closure'] if 'actual_tool_closure' in item else item['tool_closure'])
    peer=s['V10_DATA_peer'];ph=s['V10_DATA_peer_closed'];pf=s['V10_DATA_peer_final']
    c.need(peer['verdict']=='PASS_DISTINCT_DATA_SEMANTIC_SCOPE_ONLY' and not peer['blocking_findings'] and peer['author_handoff']==p['refs']['V10_DATA_handoff'],'actual distinct DATA peer, not report science')
    c.need(ph['review']==p['refs']['V10_DATA_peer'] and pf['closed_handoff']==p['refs']['V10_DATA_peer_closed'] and pf['all_owned_closed'] and pf['SDK_contact_epoch']=='UNKNOWN','actual peer final closure and epoch scope')
    c.wait(peer['actual_worker_wait']);c.wait(peer['actual_outer_wait']);c.tool(peer['actual_tool_wait'])
    n=s['V10_parent'];c.need((n['native_actual_exit'],n['raw_native_wait_status'],n['CPU_check_actual_exit'],n['outer_actual_exit'],n['actual_parent_tool_exit'])==(-9,9,0,1,1),'actual failed native statuses')
    c.wait(n['native_wait'],-9,False);c.wait(n['CPU_check_wait']);c.wait(n['outer_wait'],1,False)
    nt=c.read(n['tool_closure']);c.need((nt['actual_tool_exit'],nt['actual_exit'],nt['actual_native_exit'],nt['actual_outer_exit'])==(1,1,-9,1) and bool(nt['actual_tool_chunk']) and bool(nt['tool_start_chunk']) and nt['all_owned_fresh_absent'] is True,'actual parent native tool schema with terminal chunk; no synthetic terminal aliases')
    c.need([x['file_name'] for x in n['committed_numeric_rows']]==['row_000_complete.json','row_001_complete.json'] and n['partial_original_png_files']==1312 and n['registered_owned_count']==253,'same native two-row partial scope')
    c.need(n['native_work_deadline_seconds']==1200 and n['entry'] is None and n['producer'] is None and n['profile_close'] is None,'original finite deadline, missing closure products')
    c.need(s['parent_DATA_adoption']['handoff']==p['refs']['V10_DATA_handoff'] and s['parent_DATA_adoption']['bank_count']==0,'parent consumed exact closed partial DATA')
    pub=s['public13'];c.need(pub['status']=='CURRENT_PUBLIC_REQUIREMENTS_CLOSED_WITH_HISTORICAL_TOOL_TERMINAL_UNKNOWN' and pub['commit']==c.HEAD,'correct current published base with retained historical unknown')
    c.need(s['public13_peer']['verdict']=='PASS_CURRENT_PUBLICATION_REQUIREMENTS_ONLY' and not s['public13_peer']['blocking_findings'],'distinct current publication peer')
    f=s['dynamic_frontier'];c.wait(f['actual_worker_wait']);c.wait(f['actual_outer_wait']);ft=c.read(f['actual_tool_wait'])['actual_terminal'];c.need(type(ft['exit_code']) is int and ft['exit_code']==0 and ft['session_id'] is None,'frontier direct observed terminal schema');c.need(f['qualification'] is False and f['bank_count']==0 and f['adopted'] is False,'frontier is observation/proposal only')
    data=c.read(h['data_report']);base=s['candidate13'];plan0=c.read(base['plan']);manifest0=c.read(base['manifest']);summary=c.read(plan0['file_sources']['summary.json'])
    models=summary['increment13']['model_denominators'];c.need(models['cumulative_MODEL_ONLY']==40960 and models['original8192']['n']==8192 and models['supplement32768']['n']==32768,'unchanged separate model denominators')
    rows=list(csv.DictReader(io.StringIO(c.buffer(h['numeric_CSV'],1024**2).decode())));c.need(len(rows)==64,'only closed2x32 CSV')
    keys={(int(x['micro']),int(x['lane'])) for x in rows};c.need(keys=={(m,l) for m in (0,1) for l in range(32)},'all64 related observation identities')
    for x in rows:
        y=next(y for y in data['numeric']['rows'] if (y['micro'],y['lane'])==(int(x['micro']),int(x['lane'])))
        for k,v in y.items():
            parsed=(x[k] in ('True','true','1')) if isinstance(v,bool) else type(v)(x[k])
            c.need(parsed==v,'CSV matches closed DATA field '+k)
    for k,v in h['current']['reason_counts_micro1'].items():c.need(sum(x[k] in ('True','true','1') for x in rows if x['micro']=='1')==v,'CSV recorded reason count '+k)
    c.need(data['numeric']['all_original_flags_exact'] and data['numeric']['original_before_parameters_bitexact'] and data['numeric']['original_before_geometry_exact'],'actual before-only numerical/readback scope')
    c.need(data['numeric']['measurement_after']=='MISSING_UNKNOWN' and data['numeric']['physical_force_certificate'] is False,'missing after readback and force authority remain unknown')
    c.need(data['numeric']['clock']['actual_dt_s']==1/120 and data['missing_micro_rows']==list(range(2,13)),'same original clock/missing micro scope')
    return p,s,h,data,base,plan0,manifest0,summary,models,rows
def build(out):
    before=c.sources();p,s,h,data,base,plan0,manifest0,summary,models,rows=load_inputs()
    out=Path(out);c.need(out.resolve().is_relative_to(c.O) and not out.exists(),'exclusive own build output');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir();prefix=ov/c.PREFIX
    for k,q in p['refs'].items():c.copy(q,prefix/'evidence'/f'{k}.json')
    for k in ['data_report','compact_findings','source_graph','closure_source_graph']:c.copy(h[k],prefix/'evidence'/f'V10_{k}.json')
    for k in ['native_wait','CPU_check_wait','outer_wait','tool_closure']:c.copy(s['V10_parent'][k],prefix/'evidence'/f'V10_{k}.json')
    for n,q in before.items():c.copy(q,prefix/'source'/n)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),prefix/'source/SOURCE_GRAPH_V1.json')
    for n,q in [('index.html',base['HTML']),('summary.json',plan0['file_sources']['summary.json']),('panel_manifest.json',base['manifest'])]:c.copy(q,prefix/'archive_report13'/n)
    c.copy(h['numeric_CSV'],prefix/'COMMITTED_2X32_NUMERIC_V1.csv')
    groups=[];seen=set();totalbytes=0
    for g in data['initial_camera_groups']:
        lane=g['lane'];state=c.read(g['state']);c.need(state['env_id']==lane and state['capture_kind']=='initial' and state['status']=='qualified' and g['audit']['qualified'] is True,'bound initial state/audit identity')
        ni=state['native_invariant'];c.need(ni['all_registered_bitwise_equal'] is True and ni['environment_count']==32 and len(ni['fields'])==27,'original27 held for all32, not all64 alias')
        root=Path(g['state']['path']).parents[4];c.need(root.name=='native','explicit producer native root')
        gallery=[]
        roles={k:state['arms'][k]['selected_group_images'] for k in ['F_L','F_R','U_L','U_R']};roles['wide']=state['wide_context']['selected_images']
        for role,images in roles.items():
            c.need(len(images)==3,'three actual selected views')
            for v,e in enumerate(images,1):
                rel=Path(e['path']);c.need(not rel.is_absolute() and '..' not in rel.parts,'safe native relative original image')
                q={'path':str(root/rel),'sha256':e['sha256']};raw=c.buffer(q,4*1024**2)
                c.need(raw[:8]==b'\x89PNG\r\n\x1a\n' and int.from_bytes(raw[16:20],'big')==1280 and int.from_bytes(raw[20:24],'big')==720,'actual1280x720 original, no edit')
                name='assets/env_'+str(lane).zfill(3)+'/'+rel.name;c.need(q['path'] not in seen,'480 unique selected image identities');seen.add(q['path'])
                c.copy(q,prefix/name,4*1024**2);totalbytes+=len(raw)
                gallery.append(dict(lane=lane,role=role,view=v,relative=name,sha256=q['sha256'],original=q,scope='INITIAL_GROUP_PIXEL_COVERAGE_ONLY_NOT_PHYSICAL_SAFETY'))
        groups.append(dict(lane=lane,candidate=g['candidate'],state=g['state'],coverage=g['audit'],images=gallery))
        dest=prefix/'gallery'/('env_'+str(lane).zfill(3)+'.html');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(gallery_page(groups[-1]),encoding='utf-8')
        print(json.dumps({'copied_initial_group':lane,'selected_originals':len(seen),'bytes':totalbytes}),flush=True)
    c.need(len(groups)==32 and {g['lane'] for g in groups}==set(range(32)) and len(seen)==480,'all32 full selected480 gallery')
    galleryindex='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V10 all32 initial groups</title><style>body{max-width:900px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui}li{margin:.6rem 0}</style><a href="../../index.html#nativeAttemptIncrement14">报告14 / Report14</a><h1>全部32组初始原图 / All32 initial groups</h1><p>每组15原图，共480。初始像素覆盖已独立验收，完整挑战与物理安全未通过。15 originals per group,480 total. Initial pixel coverage is independently checked; full challenge and physical safety are not accepted.</p><ol>'
    for g in groups:galleryindex+='<li><a href="env_'+str(g['lane']).zfill(3)+'.html">lane'+str(g['lane'])+' · '+html.escape(str(g['candidate']))+' · 15 INITIAL originals</a></li>'
    (prefix/'gallery/index.html').write_text(galleryindex+'</ol></html>',encoding='utf-8')
    c.write(prefix/'SELECTED_ORIGINAL_IMAGE_INVENTORY_V1.json',dict(schema='astra.report14.selected.original.initial_images.v1',groups=groups,images=480,total_original_PNGs=1312,independently_pixel_audited_initial_attempts=1201,other_PNGs_without_pixel_acceptance=111,original_image_bytes=totalbytes,PNG_edits=0,raw_mask_redecode_by_report=0,NPY_NPZ_reads_by_report=0,caption_scope='INITIAL_COVERAGE_ONLY_NOT_FULLPREFIX_OR_PHYSICAL_SAFETY'))
    previews=[e for lane in [17,27] for role in ['wide','F_L'] for e in next(g for g in groups if g['lane']==lane)['images'] if e['role']==role and e['view']==1]
    n=s['V10_parent'];epoch=dt.datetime.fromisoformat(p['snapshot_utc']);local=epoch.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()
    facts=dict(schema='astra.report.increment14.frozen_facts.v1',snapshot_utc=epoch.isoformat(),snapshot_Asia_Shanghai=local,historical_snapshot_not_live=True,source_refs=p['refs'],packet_ref=c.ref(c.D/'FROZEN_INPUTS_V1.json'),base_public_commit=c.HEAD,model_denominators=models,new_IID_native_samples=0,native_terminal=dict(native_exit=-9,raw_status=9,CPU_exit=0,outer_exit=1,tool_exit=1,work_deadline_seconds=1200,elapsed_seconds=n['native_elapsed_seconds'],signals=n['native_supervisor_signals'],resource_abort=n['native_resource_abort'],error=n['native_error'],failure_cause='UNKNOWN',registered_identity_count=253,registered_identity_scope='251_NATIVE_PLUS_1_CPU_PLUS_1_OUTER',all_registered_fresh_absent=True,ENTRY=None,PRODUCER=None,ProfileClose=None,physical_advance='UNKNOWN_PRODUCER_ABSENT'),independent_data_scope=h['current'],before_parameters_bitexact=True,before_geometry_exact=True,after_parameters_and_geometry='MISSING_UNKNOWN',independent_DATA_semantic_verdict=s['V10_DATA_peer']['verdict'],camera_INITIAL32_PASS=True,camera_final='MISSING_UNAUDITED',selected_original_images=480,total_PNGs=1312,initial_attempts_audited=1201,other_PNGs_not_pixel_accepted=111,committed_rows=[0,1],related_numeric_observations=64,missing_micro_rows=list(range(2,13)),numerical_DENY_micro1_lanes=32,numerical_DENY_not_full_physical_verdict=True,recorded_reason_counts=h['current']['reason_counts_micro1'],valid_owned_point_counts=[3166,2546],proxy_max_reconstruction_error_m=data['max_raw9021_float64_reconstruction_error_m'],proxy_sign_disagreements=[0,0],proxy_not_mesh_or_global_certificate=True,clock_dt_s=1/120,physical_unknowns=dict(contact_SDK_refresh_epoch='UNKNOWN',actual_drive_peak='UNKNOWN',within_micro_force_peak='UNKNOWN',wrench_frame='UNKNOWN',terminal_set='UNKNOWN',global_error_bound='UNKNOWN'),held_state_scope=dict(original27='MANDATORY_AUDITED_INITIAL',latest42='UNKNOWN_UNBOUND_NOT_ADOPTED'),previews=previews,new_science_figures=2,old_science_figures=8,old_original_RGB=15,all_historical_assets_SHA_preserved=True,bank_count=0,eligible_count=0,safety_acceptance=False,all_requirements_completed=False,tasks='NOT_RUN',holdout='UNOPENED',native_GPU_SDK_Git_W_actions=0,publish_performed=False,distinct_report_scientific_peer='PENDING',reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,reviewer_model_declaration=None,public13_historical_terminal_UNKNOWN_preserved=True,report_scope='CLOSED_METADATA_CSV_PROJECTION_AND_ORIGINAL_BYTE_COPY_NO_RAW_REAUDIT')
    c.write(prefix/'FACTS_INCREMENT14_V1.json',facts);figures(rows,prefix/'figures')
    page=c.buffer(base['HTML'],4*1024**2).decode();old=page;marker='<section id="nativeAttemptIncrement13">';c.need(page.count(marker)==1,'one historical13 section')
    page=re.sub(r'<p class="asof" id="reportAsOf">.*?</p>','<p class="asof" id="reportAsOf">REPORT14 · '+epoch.isoformat()+' UTC / Asia/Shanghai '+local+' · 固定科学快照，非实时状态 / Fixed scientific snapshot, not live status.</p>',page,count=1,flags=re.S)
    page=page.replace('</head>','<style id="report14ModesV1">'+STYLE+'</style></head>',1).replace(marker,render(facts)+marker,1)
    page=page.replace('<div class="executive">','<div class="executive"><p id="native14Executive"><strong>初始32组相机已独立验收；首步数值拒绝32/32，完整安全尚未通过。Initial32 camera groups independently verified; first-step numerical denial32/32, full safety remains unaccepted.</strong> <a href="#nativeAttemptIncrement14">REPORT14</a></p>',1)
    page=page.replace('完整安全验收尚未完成。最新重试保留更多初始画面，但没有完整相机组；任务与留出验证尚未运行。','历史报告13：完整安全验收尚未完成，彼时只有部分初始画面；任务与留出验证尚未运行。',1)
    page=page.replace('Full safety acceptance remains incomplete. The latest retry retained more initial views but no complete camera group;tasks and holdout validation remain unrun.','Historical Report13: full safety remained incomplete with partial initial views; tasks and holdout validation remained unrun.',1)
    page=page.replace('<a href="#nativeAttemptIncrement13">部分覆盖与验收缺口 / Partial coverage and acceptance gaps</a>','<a href="#nativeAttemptIncrement14">初始覆盖与首步验收 / Initial coverage and first-step checks</a><a href="#nativeAttemptIncrement13">历史报告13 / Historical Report13</a>',1)
    c.need(page[page.index(marker):]==old[old.index(marker):],'entire frozen Report13 and earlier section tail unchanged')
    (ov/'index.html').write_text(page,encoding='utf-8');summary.update(asof_utc=epoch.isoformat(),asof_AsiaShanghai=local,increment14=facts,historical_candidate13_summary=c.PREFIX+'/archive_report13/summary.json');c.write(ov/'summary.json',summary)
    files={p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()};manifest=copy.deepcopy(manifest0);manifest['files'].update({n:q['sha256'] for n,q in files.items()});manifest.update(snapshot_updated_utc=epoch.isoformat(),increment14=dict(facts=c.PREFIX+'/FACTS_INCREMENT14_V1.json',bank_count=0,safety_acceptance=False));c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    c.need(all(manifest['files'][n]==s for n,s in manifest0['files'].items() if n not in ('index.html','summary.json')),'all historical manifests assets/archives unchanged')
    plan=dict(schema='astra.report.increment14.overlay_plan.v1',base_commit=c.HEAD,base_plan=base['plan'],base_manifest=base['manifest'],overlay=str(ov),overlay_kind='PARTIAL_DELTA_RESOLVE_FILE_SOURCES',file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},file_bytes={n:Path(q['path']).stat().st_size for n,q in files.items()},inherited_files={n:s for n,s in manifest0['files'].items() if n not in files},before_sha256={n:(base['manifest']['sha256'] if n=='panel_manifest.json' else manifest0['files'][n]) for n in files if n=='panel_manifest.json' or n in manifest0['files']},manifest=c.ref(ov/'panel_manifest.json'),source_facts=c.ref(prefix/'FACTS_INCREMENT14_V1.json'),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),old8fig_and15RGB_SHA_preserved=True,new_science_figures=2,new_original_INITIAL_RGB=480,selected_image_inventory=c.ref(prefix/'SELECTED_ORIGINAL_IMAGE_INVENTORY_V1.json'),raw_tree_copy=False,raw_pixel_or_numeric_reaudit=False,publish_performed=False,scientific_peer='PENDING_DISTINCT_REPORT14_PEER')
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    checks={}
    for k,value in [('safety_acceptance',True),('full12',True),('micro1_contact_refresh_freshness','VERIFIED')]:
        bad=copy.deepcopy(h['current']);bad[k]=value
        try:validate_scope(bad)
        except ValueError:checks[k]='REJECTED'
        else:raise ValueError('promotion regression did not reject '+k)
    c.write(out/'PROJECTION_CHECKS_V1.json',dict(status='PASS_METADATA_PROJECTION_ONLY',CSV_rows=64,all_original_counts_match=True,negative_promotion_regressions=checks,all480_original_bytes_and_PNG_headers_verified=True,no_source_kernel_changes=True,no_RAW_redecode=True))
    c.need(c.sources()==before,'source unchanged after build');print(json.dumps({'built_plan':c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),'originals':480,'new_figures':2}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);build(p.parse_args().out)
