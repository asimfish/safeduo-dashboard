"""LOCAL candidate closure for distinct report science; no self-granted publication."""
import argparse,json
from pathlib import Path
import common_v1 as c

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--tools',required=True);p.add_argument('--tools-sha256',required=True);a=p.parse_args();before=c.sources();ledger=c.read(dict(path=a.tools,sha256=a.tools_sha256));owned=[]
    c.need(set(ledger['stages'])=={'adapter-contract','current-gallery','build','local-qa'},'exact closed adapter/gallery/build/LOCAL stages')
    for name,x in ledger['stages'].items():
        for k in ('worker_wait','outer_wait'):owned+=c.wait(x[k])['owned_identities']
        c.tool(x['tool_closure'])
    planq=ledger['plan'];plan=c.read(planq);facts=c.read(plan['source_facts']);qa=c.read(ledger['browser_report']);inventory=c.read(plan['selected_image_inventory'])
    c.need(qa['status']=='PASS_LOCAL_ACTUAL_PARTIAL_DATA_PRESENTATION_ONLY' and len(qa['modes'])==4 and sum(len(m['EN_ZH_detailed_summary']) for m in qa['modes'])==16 and qa['errors']==[] and all(qa['closure'].values()),'four actual LOCAL modes16 language/detail combinations and real browser/server cleanup')
    c.need(qa['plan']==planq and qa['old_media_buffer_reads']==0,'exact current plan and no old media decode')
    c.need(facts['V13']['camera_INITIAL32_PASS'] is True and facts['V13']['micro1_numeric_denied_lanes']==32 and facts['V13']['missing_rows']==list(range(2,13)) and facts['V13']['full12_complete'] is False,'INITIAL DATA/recorded numeric scope separate from missing fullprefix')
    c.need(inventory['copied_original_images']==480 and inventory['copied_original_bytes']==184366361 and len(inventory['groups'])==32,'actual current480 byte-exact original gallery')
    c.need(facts['publish_allowed'] is facts['publish_performed'] is facts['safety_acceptance'] is False and facts['bank_count']==0,'no self-granted science, publication or safety')
    c.need(c.sources()==before and all(c.absent(i) for i in owned),'frozen source and actual prior owned identity absence')
    result=dict(schema='astra.report16.actual_LOCAL_candidate_handoff.v1',status='LOCAL_ACTUAL_DATA_AND_CURRENT480_GALLERY_CLOSED_PENDING_DISTINCT_REPORT_SCIENCE',HTML=plan['file_sources']['index.html'],facts=plan['source_facts'],plan=planq,manifest=plan['manifest'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),snapshot_utc=facts['snapshot_utc'],snapshot_Asia_Shanghai=facts['snapshot_Asia_Shanghai'],base_public_commit=c.HEAD,actual_LOCAL_modes=4,actual_EN_ZH_detail_summary_combinations=16,browser_report=ledger['browser_report'],actual_current_stages=ledger['stages'],prior_registered_CPU_identities=owned,prior_registered_CPU_fresh_absent=True,source_before_after=before,source_unchanged=True,current480_selection=inventory['actual_selected_metadata'],current480_original_inventory=plan['selected_image_inventory'],current480_original_bytes=184366361,new_original_INITIAL_images=480,new_figures=0,all_original1280x720=True,image_crop_resize_generation=0,old_V10_gallery_not_reused_for_V13=True,old_media_QA_reused_without_redecode=plan['old_media_QA'],all_old_manifest_entries_preserved=True,old15_and_earlier_sections_byte_identical=True,independent_partial_DATA=facts['V13']['partial_DATA_ref'],independent_DATA_semantic_peer=facts['distinct_DATA_semantic_peer'],parent_complete_DATA_adoption=facts['parent_complete_DATA_adoption'],camera_scope='INITIAL32_ONLY_PASS_NOT_FINAL_OR_FULL12',numeric_scope='64_RELATED_ROWS01_FIRSTSTEP32_DENY_NOT_CERTIFIED_PHYSICAL_ADVANCE',CONTACT_SDK_epoch='UNKNOWN',Drive='UNKNOWN',global_error_bound='UNKNOWN',full12=False,bank_count=0,tasks='NOT_RUN',holdout='UNOPENED',safety_acceptance=False,all_requirements_completed=False,distinct_report_science_peer=None,reviewer_selected_by_parent=True,reviewer_declared_by_user=False,reviewer_agent_id=None,publish_performed=False,publish_allowed=False,historical71948_tool_terminal='UNKNOWN_PERMANENT',all_historical_tools_zero=False,current_handoff_worker_outer_tool_proof='EXTERNAL_ACTUAL_WAIT_AND_TOOL_CLOSURE_REQUIRED_AFTER_RETURN')
    q=c.write(c.O/'LOCAL_ACTUAL_DATA_CANDIDATE_HANDOFF_V16.json',result);print(json.dumps(dict(handoff=q,publication=False,report_science='PENDING')),flush=True)

if __name__=='__main__':main()
