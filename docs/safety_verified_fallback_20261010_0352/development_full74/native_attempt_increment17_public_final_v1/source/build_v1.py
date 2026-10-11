"""Only immutable status wording and genuine review metadata; no new science/media."""
import argparse,copy,json,re
from pathlib import Path
import common_v1 as c
PREFIX='native_attempt_increment17_public_final_v1'
def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--out',required=True);a=ap.parse_args();before=c.sources();h,local,peer=c.inherited()
    oldplan=c.read(h['plan']);oldmanifest=c.read(h['manifest']);oldfacts=c.read(h['facts']);old=c.buffer(h['HTML'],4*1024**2).decode('utf-8')
    out=Path(a.out);c.need(not out.exists() and out.resolve().is_relative_to(c.O),'exclusive final candidate');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay'
    facts=copy.deepcopy(oldfacts)
    metadata=dict(status='FROZEN_EXPERIMENT_SNAPSHOT_DISTINCT_SCOPED_REPORT_REVIEW_PASS',final_scientific_cutoff=c.CUTOFF,report_science_peer='PASS_DRAFT5_SCOPED_MAPPING_FINAL_PRESENTATION_DELTA_REVIEW_PENDING',parent_complete_Report17_fact_binding='FROZEN_DRAFT5_SCIENTIFIC_INPUTS',browser_QA='CURRENT_FINAL_LOCAL_PENDING',publish_performed=False,publish_allowed=False)
    facts.update(metadata);facts['public_final_metadata']=dict(inherited_science=c.PEER,inherited_science_actual_terminal=c.PEER_TERMINAL,inherited_candidate=c.DRAFT,cutoff_preserved=True,presentation_delta_review='PENDING_DISTINCT_REVIEW',reviewer_selected_by_parent=True,physical_acceptance=False)
    c.need({k:v for k,v in facts.items() if k not in metadata and k!='public_final_metadata'}=={k:v for k,v in oldfacts.items() if k not in metadata},'all scientific fields byte-value equivalent')
    source_prefix=c.read(h['facts'])['prefix'];old_local=oldfacts['snapshot_Asia_Shanghai']
    replacements=[
      ('REPORT17 LOCAL DRAFT','REPORT17 · FROZEN EXPERIMENT SNAPSHOT'),
      ('data-publication="LOCAL_DRAFT_ONLY"','data-publication="FROZEN_SCOPED_REVIEWED_SNAPSHOT"'),
      ('报告17草稿：部分审计闭合，完整挑战仍未验收','报告17：部分审计闭合，完整挑战仍未验收'),
      ('Report17 draft: partial audit closed; full challenge remains unaccepted','Report17: partial audit closed; full challenge remains unaccepted'),
      (f'本版草稿输入截止 {c.CUTOFF}（北京时间 {old_local}）。最终科学截止尚待主级收束；本地草稿等待完整事实绑定与独立报告科学评审，不是当前运行状态。',f'本版实验快照冻结于 {c.CUTOFF}（北京时间 {old_local}）。范围限定的独立报告科学评审已完成，核对本版事实与证据的对应关系；这不代表完整物理安全验收，也不描述截止之后的实时运行。'),
      (f'Draft input cutoff {c.CUTOFF} (Asia/Shanghai {old_local}). The final scientific cutoff awaits parent consolidation. This local draft awaits complete fact binding and a distinct report science review; it is not live status.',f'This experiment snapshot is frozen at {c.CUTOFF} (Asia/Shanghai {old_local}). A distinct scoped scientific report review has verified the mapping of facts to evidence. It does not establish complete physical safety acceptance or describe live events after the cutoff.'),
      ('本报告新事实映射仍须独立报告科学复核。','本报告冻结事实映射已完成独立范围限定科学复核；物理验收范围未扩大。'),
      ('This report’s new fact mapping still requires a distinct report science review.','The frozen fact mapping received a distinct scoped scientific report review; physical acceptance has not expanded.'),
      ('固定草稿事实 / Frozen draft facts','冻结快照事实 / Frozen snapshot facts'),
      ('正在运行的审计与截止之后的修复，不进入本草稿结论。','截止之后的审计与修复不进入本版结论。'),
      ('This draft changes reading order while preserving historical text and anchors; running audits and later repairs do not enter its conclusions.','This report prioritizes the latest frozen snapshot while preserving historical text and anchors; audits and repairs after the cutoff do not enter its conclusions.')]
    page=old
    for x,y in replacements:c.need(page.count(x)==1,'exact single status text: '+x[:80]);page=page.replace(x,y)
    boundary=re.search(r'<p id="r17PendingBoundary">.*?</p>',page)
    c.need(boundary is not None,'one status evidence note')
    note=f'<p id="r17PendingBoundary">范围限定科学评审已完成；完整挑战、安全、任务与留出验收仍未完成。 / Scoped scientific report review completed; full challenge, safety, tasks and holdout remain unaccepted. 历史缺失工具终端仍为UNKNOWN_PERMANENT。 <a href="{PREFIX}/evidence/DRAFT5_SCIENTIFIC_HANDOFF.json">独立评审 / Distinct review</a> · <a href="{PREFIX}/source/SOURCE_GRAPH_V1.json">最终呈现源码 / Final presentation source</a></p>'
    page=page[:boundary.start()]+note+page[boundary.end():]
    marker='<div id="r17HistoricalReports"';c.need(page[page.index(marker):]==old[old.index(marker):],'entire historical payload bytes unchanged')
    current=page[page.index('<section id="nativeAttemptIncrement17"'):page.index(marker)]
    c.need(all(x not in current for x in ('LOCAL_DRAFT','报告17草稿','Report17 draft:','最终科学截止尚待','science peer PENDING','browser QA NOT_RUN')),'final current prose has no draft/review-pending notices')
    ov.mkdir();(ov/'index.html').write_text(page,encoding='utf-8');c.write(ov/source_prefix/'FACTS_DRAFT_INCREMENT17_V1.json',facts)
    summary=c.read(oldplan['file_sources']['summary.json']);summary['increment17']=facts;c.write(ov/'summary.json',summary)
    c.copy(c.PEER,ov/PREFIX/'evidence/DRAFT5_SCIENTIFIC_HANDOFF.json');c.copy(c.PEER_TERMINAL,ov/PREFIX/'evidence/DRAFT5_SCIENTIFIC_TERMINAL.json')
    for name,q in before['own_source_files'].items():c.copy(q,ov/PREFIX/'source'/Path(name).name)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),ov/PREFIX/'source/SOURCE_GRAPH_V1.json')
    files=dict(oldplan['file_sources']);files.update({p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()})
    manifest=copy.deepcopy(oldmanifest);manifest['files'].update({n:q['sha256'] for n,q in files.items() if n!='panel_manifest.json'});manifest['increment17'].update(DRAFT_ONLY=False,frozen_experiment_snapshot=True,scoped_report_review='PASS_DRAFT5',public_final_presentation_review='PENDING_BEFORE_PUBLICATION',publish_allowed=False);c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    plan=copy.deepcopy(oldplan);plan.update(schema='astra.report17.public_final_partial_overlay.v1',overlay=str(ov),file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},manifest=files['panel_manifest.json'],source_facts=files[source_prefix+'/FACTS_DRAFT_INCREMENT17_V1.json'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),parent_complete_fact_binding='FROZEN_DRAFT5',distinct_report_science_peer=c.PEER,actual_browser_QA='CURRENT_FINAL_LOCAL_PENDING',inherited_draft=h,inherited_draft_ref=c.DRAFT)
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    checks=dict(scientific_values_preserved=True,scientific_cutoff_unchanged=True,entire_historical_payload_byte_exact=True,old_asset_SHA_entries_preserved=all(manifest['files'][n]==s for n,s in oldmanifest['files'].items() if n not in ('index.html','summary.json',source_prefix+'/FACTS_DRAFT_INCREMENT17_V1.json')),bilingual_status_only=True,new_media=0,RAW_payload_reads=0,source_before=before,source_after=c.sources(),replacement_pairs=replacements)
    c.need(all(checks[k] for k in ('scientific_values_preserved','entire_historical_payload_byte_exact','old_asset_SHA_entries_preserved')),'status-only final delta')
    c.write(out/'PUBLIC_FINAL_DELTA_CHECKS_V1.json',checks)
    candidate=dict(schema='astra.report17.public_final_candidate.v1',status='IMMUTABLE_PUBLIC_FINAL_PENDING_LOCAL_AND_DISTINCT_PRESENTATION_DELTA',snapshot_utc=c.CUTOFF,base_public_commit=c.BASE,HTML=files['index.html'],facts=plan['source_facts'],plan=c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),checks=c.ref(out/'PUBLIC_FINAL_DELTA_CHECKS_V1.json'),inherited_draft=c.DRAFT,inherited_LOCAL=c.DRAFT_LOCAL,inherited_science=c.PEER,inherited_science_terminal=c.PEER_TERMINAL,bank_count=0,safety_acceptance=False,new_media=0,new_figures=0,publish_performed=False)
    c.write(out/'PUBLIC_FINAL_CANDIDATE_V17.json',candidate);c.need(c.sources()==before,'final source unchanged');print(json.dumps({'candidate':c.ref(out/'PUBLIC_FINAL_CANDIDATE_V17.json'),'HTML':candidate['HTML'],'changed_files':len(files),'snapshot_utc':c.CUTOFF}),flush=True)
if __name__=='__main__':main()
