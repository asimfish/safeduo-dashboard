"""Seal local candidate after genuine CPU/tool closure; science verdict stays pending."""
import argparse,datetime,json
from pathlib import Path
import report_v1 as r
def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('tools','tools-sha256','plan','plan-sha256'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();b=r.binding();tools=b.read(r.rr(a.tools,a.tools_sha256));plan=b.read(r.rr(a.plan,a.plan_sha256));graph=b.read(plan['source_graph'])
    b.need(plan['schema']=='astra.report.increment13.overlay_plan.v1' and plan['publish_performed'] is False,'local13 candidate, not publication')
    before={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()};b.need(before==graph['own_source_files'],'unchanged exact local source')
    identities=[];waits=[]
    b.need(set(tools['stages'])=={'source-tests','build','local-qa'},'all three actual stages')
    for mode,stage in tools['stages'].items():
        for k in ('worker_wait','outer_wait'):
            w=b.closed_wait(stage[k]);waits.append(stage[k]);identities.extend(w['owned_identities'])
        r.closed_tool(b,stage['tool_closure'],0);delivery=b.read(stage['delivery'])
        b.need(delivery['source_unchanged'] is True and delivery['source_graph']==plan['source_graph'],'real delivery source graph')
    facts=b.read(plan['source_facts']);r.inputs(b,facts['packet_ref'])
    manifest=b.read(plan['manifest']);oldmanifest=b.read(plan['base_manifest'])
    for n,q in plan['file_sources'].items():b.need(b.sha(q['path'])==q['sha256']==plan['changed_files'][n],'exact changed source bytes')
    b.need(all(manifest['files'][n]==sha for n,sha in oldmanifest['files'].items() if n not in ('index.html','summary.json')),'all historical metadata SHA preserved')
    qa=b.read(tools['browser_report']);b.need(qa['status']=='PASS_REAL_LOCAL_BROWSER' and len(qa['modes'])==4 and not qa['errors'] and all(qa['closed'].values()),'four actual local browser modes clean closed')
    for mode in qa['modes']:b.need(mode['new_partial_RGB']==6 and mode['figures']==8 and mode['original_RGB']==15 and mode['page_overflow'] is False,'six original views and old assets visible')
    visual=b.read(tools['visual_inspection']);b.need(visual['status']=='VISUALLY_INSPECTED_LOCAL_DESKTOP_MOBILE' and visual['scope']=='PRESENTATION_ONLY_NOT_CAMERA_QUALIFICATION','actual presentation visual scope')
    unique={(i['pid'],i['startticks']):i for i in identities};b.need(all(b.fresh_absent(i) for i in unique.values()),'all locally enrolled fresh absent')
    out=r.O/r.D.name;out.mkdir(parents=True,exist_ok=False)
    inventory=b.write(out/'WHOLE_LOCAL_DELTA_INVENTORY_V1.json',dict(files=[dict(name=n,ref=q,bytes=Path(q['path']).stat().st_size) for n,q in plan['file_sources'].items()],inherited_metadata_only=plan['inherited_files'],old8fig15RGB_metadata_SHA_preserved=True,whole_old_payload_rescan=False,RAW_tree_scan=False))
    result=dict(schema='astra.report13.local_closed_candidate.v1',status='CLOSED_CANDIDATE_REPORT13_CPU_LOCAL4MODES_PENDING_DISTINCT_SCIENCE_PEER',verdict='PASS_LOCAL_PRESENTATION_ONLY',candidate=Path(plan['overlay']).parent.name,sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_utc=facts['snapshot_utc'],snapshot_AsiaShanghai=facts['snapshot_Asia_Shanghai'],HTML=plan['file_sources']['index.html'],plan=r.rr(a.plan,a.plan_sha256),manifest=plan['manifest'],facts=plan['source_facts'],source_graph=plan['source_graph'],source_before_after_unchanged=True,actual_stage_tools=r.rr(a.tools,a.tools_sha256),true_stage_waits=waits,local_browser=tools['browser_report'],visual_inspection=tools['visual_inspection'],whole_local_delta_inventory=inventory,local_modes=4,EN_ZH_detailed_summary=True,all_scoped_stage_owned_fresh_absent=True,scoped_stage_owned_ID_count=len(unique),scoped_stage_owned_identities=list(unique.values()),base_public_commit=r.BASE_COMMIT,preserved_figures=8,preserved_original_native_RGB=15,new_partial_native_RGB=6,new_science_figures=0,complete_camera_groups=0,original_PNGs_uncropped_unedited=True,bank_count=0,physical_acceptance=False,all_requirements_completed=False,Tasks='NOT_RUN',holdout='UNOPENED',Drive='UNKNOWN',CONTACT='UNKNOWN',global_error_bound='UNKNOWN',native_GPU_SDK_Git_W_actions=0,publish_performed=False,scientific_peer=None,scientific_review_status='PENDING_DISTINCT_PEER_SELECTED_BY_PARENT',reviewer_selected_by_parent=True,reviewer_agent_id=None,reviewer_model_declaration=None,reviewer_declared_by_user=False,provisional_early_unfrozen_draft=dict(source=str(r.D.parent/'draft_v1.py'),executed=False,terminal_pending_outline_superseded=True))
    ref=b.write(out/'LOCAL_CLOSED_CANDIDATE_V13.json',result)
    b.need(before=={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()},'source unchanged after seal')
    print(json.dumps(ref),flush=True)
if __name__=='__main__':main()
