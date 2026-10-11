"""Seal the actual local text-only candidate; independent science and publication pending."""
import argparse,datetime as dt,json
from pathlib import Path
import common_v1 as c

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ['plan','plan-sha256','tools','tools-sha256']:p.add_argument('--'+k,required=True)
    a=p.parse_args();before=c.sources();pq={'path':a.plan,'sha256':a.plan_sha256};plan=c.read(pq);facts=c.read(plan['source_facts']);tools=c.read({'path':a.tools,'sha256':a.tools_sha256});ids=[]
    c.need(set(tools['stages'])=={'build','local-qa'},'actual minimal build plus new increment fourmode QA')
    for stage in tools['stages'].values():
        for k in ['worker_wait','outer_wait']:ids.extend(c.wait(stage[k])['owned_identities'])
        c.tool(stage['tool_closure']);d=c.read(stage['delivery']);c.need(d['source_before_after']==before and d['source_unchanged'],'real source-bound CPU stage')
    qa=c.read(tools['browser_report']);c.need(qa['status']=='PASS_LOCAL_INCREMENT_PRESENTATION_ONLY' and len(qa['modes'])==4 and not qa['errors'] and all(qa['closure'].values()),'actual fourmode new text UI gate closed')
    c.need(qa['original_media_buffer_reads']==0 and qa['blocked_old_media_UNTESTED_THIS_RUN'] is True,'no repeated old raw/image browser decoding')
    inherited=c.read(plan['old_media_QA']);c.need(inherited['status']=='PASS_LOCAL_PRESENTATION_ONLY' and len(inherited['modes'])==4 and not inherited['errors'],'exact inherited media gate')
    manifest=c.read(plan['manifest']);files=[]
    for name,q in plan['file_sources'].items():
        c.need(c.sha(q['path'])==q['sha256'],'whole small delta SHA '+name);c.need(name=='panel_manifest.json' or manifest['files'][name]==q['sha256'],'manifest/source agreement');files.append(dict(path=name,source=q,bytes=Path(q['path']).stat().st_size))
    c.need(all(manifest['files'][n]==d for n,d in plan['inherited_files'].items()),'all inherited media/history manifest entries unchanged without asset reread')
    inventory=c.write(c.O/'WHOLE_LOCAL_DELTA_INVENTORY_V1.json',dict(schema='astra.report15.local.delta.inventory.v1',files=files,paths=len(files),bytes=sum(x['bytes'] for x in files),source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),original_media_buffer_reads=0,RAW_tree_copied=False,inherited_artifact_checks='MANIFEST_REFERENCE_EQUALITY_PLUS_COMPLETED_REPORT14_QA_NOT_REHASHED_MEDIA'))
    candidate=dict(schema='astra.report15.local_closed_candidate.v1',status='CLOSED_LOCAL_REPORT_CANDIDATE_PENDING_DISTINCT_SCIENTIFIC_REVIEW',verdict='PASS_NEW_LOCAL_PRESENTATION_AND_CLOSED_METADATA_SCOPE_ONLY',snapshot_utc=facts['snapshot_utc'],snapshot_Asia_Shanghai=facts['snapshot_Asia_Shanghai'],HTML=plan['file_sources']['index.html'],plan=pq,manifest=plan['manifest'],facts=plan['source_facts'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),base_public_commit=c.HEAD,local_browser=tools['browser_report'],local_modes=4,local_EN_ZH_detail_summary=True,actual_CPU_stages=tools['stages'],original_media_buffer_reads=0,old_media_QA_reused=plan['old_media_QA'],CPU_threads=1,build_AS_bytes=1073741824,browser_policy=qa['resource_policy'],source_before_after_unchanged=True,whole_local_delta_inventory=inventory,old_figures=8,inherited_scientific_figures=2,new_scientific_figures=0,old_original_RGB=15,inherited_selected_RGB=480,new_original_RGB=0,all_historical_manifest_SHA_preserved=True,V10_initial32_PASS_retained=True,V11_V12_raw_pixel_qualification=False,V12_rows_missing=list(range(13)),V12_physical_advance='UNKNOWN_PRODUCER_ABSENT',related_retries_new_IID=0,physical_acceptance=False,all_requirements_completed=False,bank_count=0,tasks='NOT_RUN',holdout='UNOPENED',Drive='UNKNOWN',global_error_bound='UNKNOWN',publish_performed=False,native_GPU_SDK_Git_W_actions=0,scientific_peer=None,reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,reviewer_model_declaration=None,registered_CPU_identities=ids,all_scoped_prior_CPU_fresh_absent=all(c.absent(i) for i in ids),historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False,sealed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),seal_stage_true_terminal_required=True)
    c.need(c.sources()==before and candidate['all_scoped_prior_CPU_fresh_absent'],'source unchanged and own prior CPUs absent');q=c.write(c.O/'LOCAL_CLOSED_CANDIDATE_V15.json',candidate);print(json.dumps({'candidate':q,'HTML':candidate['HTML']}),flush=True)

if __name__=='__main__':main()
