"""Seal real local presentation, input and source scope; no scientific self-review."""
import argparse,datetime as dt,json
from pathlib import Path
import common_v1 as c
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ['plan','plan-sha256','tools','tools-sha256']:p.add_argument('--'+k,required=True)
    a=p.parse_args();before=c.sources();pq={'path':a.plan,'sha256':a.plan_sha256};plan=c.read(pq);facts=c.read(plan['source_facts']);tools=c.read({'path':a.tools,'sha256':a.tools_sha256});ids=[]
    for name,stage in tools['stages'].items():
        for k in ['worker_wait','outer_wait']:
            w=c.wait(stage[k]);ids.extend(w['owned_identities'])
        c.tool(stage['tool_closure']);d=c.read(stage['delivery']);c.need(d['source_before_after']==before and d['source_unchanged'],'each executed stage source bound')
    c.need(set(tools['stages'])=={'presentation_repair','local-qa'},'real presentation repair and fourmode browser stages')
    for k in ['worker_wait','outer_wait']:c.wait(tools['original_closed_build'][k])
    c.tool(tools['original_closed_build']['tool_closure'])
    qa=c.read(tools['browser_report']);c.need(qa['status']=='PASS_LOCAL_PRESENTATION_ONLY' and len(qa['modes'])==4 and not qa['errors'],'real4 local browser modes')
    manifest=c.read(plan['manifest']);inventory=c.read(plan['selected_image_inventory']);c.need(inventory['images']==480 and len(inventory['groups'])==32,'complete selected original gallery')
    files=[]
    for name,q in plan['file_sources'].items():
        c.need(c.sha(q['path'])==q['sha256'],'whole delta stable SHA '+name);c.need(name=='panel_manifest.json' or manifest['files'][name]==q['sha256'],'manifest/file_sources agree');files.append(dict(path=name,source=q,bytes=Path(q['path']).stat().st_size))
    c.write(c.O/'WHOLE_LOCAL_DELTA_INVENTORY_V1.json',dict(schema='astra.report14.local.delta.inventory.v1',files=files,paths=len(files),bytes=sum(x['bytes'] for x in files),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),RAW_tree_copied=False))
    candidate=dict(schema='astra.report14.local_closed_candidate.v1',status='CLOSED_LOCAL_REPORT_CANDIDATE_PENDING_DISTINCT_SCIENTIFIC_REVIEW',verdict='PASS_LOCAL_PRESENTATION_AND_EXACT_SOURCE_ONLY',snapshot_utc=facts['snapshot_utc'],snapshot_Asia_Shanghai=facts['snapshot_Asia_Shanghai'],HTML=plan['file_sources']['index.html'],plan=pq,manifest=plan['manifest'],facts=plan['source_facts'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),base_public_commit=c.HEAD,local_browser=tools['browser_report'],local_modes=4,local_EN_ZH_detail_summary=True,actual_CPU_stages=tools['stages'],inherited_actual_original_build=tools['original_closed_build'],presentation_only_delta=tools['presentation_delta'],CPU_threads=1,build_AS_bytes=1073741824,browser_policy=qa['resource_policy'],source_before_after_unchanged=True,whole_local_delta_inventory=c.ref(c.O/'WHOLE_LOCAL_DELTA_INVENTORY_V1.json'),selected_image_inventory=plan['selected_image_inventory'],initial_camera_groups=32,selected_initial_original_images=480,all480_original_byte_copies=True,initial_attempt_images_independently_pixel_audited_by_upstream=1201,other_PNGs_without_pixel_acceptance=111,committed_rows=[0,1],related_numeric_observations=64,new_IID_samples=0,micro1_recorded_numeric_DENY=32,SDK_contact_refresh_epoch='UNKNOWN',full12_complete=False,missing_micro_rows=list(range(2,13)),old_figures=8,new_scientific_figures=2,old_original_RGB=15,all_historical_assets_SHA_preserved=True,physical_acceptance=False,all_requirements_completed=False,bank_count=0,Tasks='NOT_RUN',holdout='UNOPENED',Drive='UNKNOWN',global_error_bound='UNKNOWN',publish_performed=False,native_GPU_SDK_Git_W_actions=0,scientific_peer=None,reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,reviewer_model_declaration=None,registered_CPU_identities=ids,all_scoped_prior_CPU_fresh_absent=all(c.absent(i) for i in ids),historical_Report13_terminal_UNKNOWN_preserved=True,sealed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),final_seal_stage_true_terminal_required=True)
    c.need(c.sources()==before,'source unchanged during seal');q=c.write(c.O/'LOCAL_CLOSED_CANDIDATE_V14.json',candidate);print(json.dumps({'candidate':q,'HTML':candidate['HTML']}),flush=True)
if __name__=='__main__':main()
