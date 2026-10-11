"""One report wording repair. Reuse DATA, original images and closed browser QA."""
import argparse, ast, copy, hashlib, json, re
from html.parser import HTMLParser
from pathlib import Path
import common_v1 as c

class Structure(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.events=[]
    def handle_starttag(self,tag,attrs): self.events.append(('start',tag,attrs))
    def handle_startendtag(self,tag,attrs): self.events.append(('empty',tag,attrs))
    def handle_endtag(self,tag): self.events.append(('end',tag))
    def handle_comment(self,data): self.events.append(('comment',data))
    def handle_decl(self,data): self.events.append(('decl',data))

def structure(s):
    p=Structure();p.feed(s);p.close();return p.events

def keyed(j,k):
    if isinstance(j,dict):
        return [v for a,v in j.items() if a==k]+sum((keyed(v,k) for v in j.values()),[])
    if isinstance(j,list):return sum((keyed(v,k) for v in j),[])
    return []

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);a=p.parse_args()
    source_before=c.sources();b=c.read(c.ref(c.D/'TEXT_DELTA_BINDINGS_V1.json'))
    h=c.read(b['base_candidate']);delivery=c.read(b['base_delivery']);plan0=c.read(b['base_plan'])
    criteria=c.read(b['original_camera_criteria']);review=c.read(b['blocked_scientific_review'])
    peer=c.read(b['blocked_scientific_handoff']);terminal=c.read(b['blocked_scientific_final_tool'])
    c.need(keyed(criteria,'critical_native_forearm_and_wrist2_peak_pixels_min')==[128] and keyed(criteria,'three_wide_views_required')==[True],'exact original peak128 and three selected wide views')
    c.need(sum(x['result']=='PASS' for x in review['checks'])==7 and sum(x['result']=='FAIL' for x in review['checks'])==1 and len(peer['blocking_findings'])==1 and peer['blocking_findings'][0]['id']=='REPORT16-CAMERA-CRITICAL-PEAK-01','one observed report text blocker, seven inherited passing checks')
    c.need(peer['candidate']==b['base_candidate'] and peer['actual_exit_quadruple']==[0,0,0,0] and terminal['actual_tool_exit']==0,'actual blocked peer execution closed, not a science PASS')
    c.need(delivery['candidate_handoff']==b['base_candidate'] and h['plan']==b['base_plan'] and delivery['all_registered_current_fresh_absent'] is True,'actual LOCAL base closure')
    oldqa=c.read(h['browser_report'])
    c.need(oldqa['status']=='PASS_LOCAL_ACTUAL_PARTIAL_DATA_PRESENTATION_ONLY' and len(oldqa['modes'])==4 and sum(len(x['EN_ZH_detailed_summary']) for x in oldqa['modes'])==16 and oldqa['errors']==[] and all(oldqa['closure'].values()),'reuse actual unchanged base four modes and sixteen language/detail combinations')
    for stage in h['actual_current_stages'].values():
        c.wait(stage['worker_wait']);c.wait(stage['outer_wait']);c.tool(stage['tool_closure'])
    c.wait(delivery['actual_handoff_writer_worker_wait']);c.wait(delivery['actual_handoff_writer_outer_wait']);c.tool(delivery['actual_handoff_writer_tool'])
    c.need(all(c.absent(i) for i in delivery['registered_current_owned_identities']),'base registered identities freshly absent')
    facts=c.read(h['facts']);c.need(facts['snapshot_utc']==b['scientific_cutoff_utc'] and facts['bank_count']==0 and facts['V13']['full12_complete'] is False and facts['tasks']=='NOT_RUN' and facts['holdout']=='UNOPENED','unchanged snapshot and incomplete safety')
    old=c.buffer(h['HTML'],4*1024**2).decode();new=old
    old_render=c.buffer(c.read(b['base_source_graph'])['own_source_files']['render_v1.py'],2*1024**2).decode()
    expected_render=old_render
    for r in b['replacements']:
        c.need(new.count(r['old'])==1 and expected_render.count(r['old'])==1,'one exact authorized bilingual/title replacement')
        new=new.replace(r['old'],r['new'],1);expected_render=expected_render.replace(r['old'],r['new'],1)
    c.need(c.buffer(c.ref(c.D/'render_v1.py'),2*1024**2).decode()==expected_render,'render source differs only in three authorized text replacements')
    c.need(structure(new)==structure(old),'all element topology, attributes, CSS, links and media paths unchanged')
    c.need(re.findall(r'<style.*?</style>',new,re.S)==re.findall(r'<style.*?</style>',old,re.S),'stylesheet bytes unchanged')
    marker='<section id="nativeAttemptIncrement15">';c.need(new[new.index(marker):]==old[old.index(marker):],'all original15 and earlier section bytes unchanged')
    for kind in ('sitebar','toc'):
        navs=re.findall(r'<nav[^>]*class="'+kind+r'"[^>]*>.*?</nav>',new,re.S)
        c.need(len(navs)==1 and navs[0].count('href="#nativeAttemptIncrement16"')==1,'unique direct latest navigation '+kind)
    newpart=new[new.index('<section id="nativeAttemptIncrement16"'):new.index(marker)]
    c.need('每个关键forearm/wrist2 body在三张已选广角中的像素峰值至少为128。' in newpart and 'each critical forearm/wrist2 body requires a peak of at least 128 pixels across the three selected wide views.' in newpart,'exact bilingual original peak criterion')
    c.need('至少两张' not in newpart and 'at least two of the three wide views' not in newpart and 'for32 groups' not in newpart,'no added two-view requirement; title spacing')
    c.need(newpart.count('class="r16-gallery-links"')==1 and len(re.findall(r'href="'+re.escape(c.PREFIX)+r'/gallery/env_\d{3}\.html"',newpart))==32,'all current32 gallery links retained')
    oldast=ast.parse(old_render);newast=ast.parse(expected_render)
    gallery=lambda tree:ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='gallery_page'),include_attributes=False)
    c.need(gallery(oldast)==gallery(newast),'gallery renderer function AST byte behavior unchanged')
    out=Path(a.out);c.need(out.resolve().is_relative_to(c.O) and not out.exists(),'exclusive candidate03 output');out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir()
    files=copy.deepcopy(plan0['file_sources']);(ov/'index.html').write_text(new,encoding='utf-8');files['index.html']=c.ref(ov/'index.html')
    for name,q in source_before.items():files[c.PREFIX+'/source/'+name]=c.copy(q,ov/c.PREFIX/'source'/name,2*1024**2)
    graph=c.ref(c.D/'SOURCE_GRAPH_V1.json');files[c.PREFIX+'/source/SOURCE_GRAPH_V1.json']=c.copy(graph,ov/c.PREFIX/'source/SOURCE_GRAPH_V1.json',2*1024**2)
    checks=dict(schema='astra.report16.bounded_text_navigation_verification.v1',status='PASS_TEXT_DELTA_AND_NAVIGATION_WITH_INHERITED_BROWSER_QA',base_candidate=b['base_candidate'],blocked_peer=b['blocked_scientific_handoff'],original_criteria=b['original_camera_criteria'],peak_min=128,selected_wide_views_required=3,two_view_count_requirement=False,replacements=b['replacements'],exact_three_text_replacements=True,element_topology_attrs_CSS_navigation_unchanged=True,original15_tail_byte_identical=True,current480_image_refs_unchanged=True,current32_gallery_links_unchanged=True,gallery_function_AST_unchanged=True,base_LOCAL_browser_report=h['browser_report'],inherited_actual_LOCAL_modes=4,inherited_actual_EN_ZH_detail_summary_combinations=16,new_browser_runs=0,new_PNG_payload_reads=0,new_DATA_or_RAW_checks=0,new_Git_HTTP_GPU_SDK_actions=0,scientific_cutoff_utc=facts['snapshot_utc'],science_delta_review='PENDING_DISTINCT_HOOKE',source_before_after=source_before,source_unchanged=True,publish_performed=False)
    verification=c.write(out/'BOUNDED_TEXT_NAVIGATION_VERIFICATION_V1.json',checks)
    manifest=c.read(plan0['manifest']);manifest['files'].update({k:q['sha256'] for k,q in files.items() if k!='panel_manifest.json'});mq=c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=mq
    old_assets={k:v for k,v in plan0['file_sources'].items() if not k.startswith(c.PREFIX+'/source/') and k not in ('index.html','panel_manifest.json')}
    c.need(all(files[k]==v for k,v in old_assets.items()),'facts,summary,all480 originals,all32 gallery pages,archive and evidence refs unchanged')
    plan=copy.deepcopy(plan0);plan.update(overlay=str(ov),file_sources=files,changed_files={k:q['sha256'] for k,q in files.items()},manifest=mq,source_graph=graph,repaired_from=b['base_candidate'],repair_kind='BILINGUAL_CRITICAL_FOREARM_WRIST2_PEAK_ONLY_AND_TITLE_SPACING',bounded_text_navigation_verification=verification,LOCAL_browser_QA=h['browser_report'],LOCAL_browser_QA_reuse_scope='SAME_DOM_CSS_ATTRIBUTES_LINKS_IMAGE_REFS_EXACT_THREE_TEXT_DELTA',new_browser_runs=0)
    pq=c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    hand=copy.deepcopy(h);hand.update(status='LOCAL_TEXT_DELTA_CLOSED_WITH_INHERITED_FOUR_MODES_PENDING_ONE_DISTINCT_REPORT_SCIENCE_DELTA',HTML=files['index.html'],plan=pq,manifest=mq,source_graph=graph,source_before_after=source_before,base_candidate=b['base_candidate'],base_current_delivery=b['base_delivery'],blocked_scientific_review=b['blocked_scientific_review'],blocked_scientific_handoff=b['blocked_scientific_handoff'],inherited_scientific_checks=[x for x in review['checks'] if x['result']=='PASS'],inherited_scientific_passes=7,new_distinct_scientific_delta_required='REPORT16-CAMERA-CRITICAL-PEAK-01',actual_current_stages_scope='INHERITED_BASE_GALLERY_BUILD_LOCAL_QA; NEW_DELTA_WAIT_AND_TOOL_SEPARATE',new_local_text_navigation_verification=verification,browser_QA_reused=True,browser_QA_reuse_scope='DOM_CSS_ATTRIBUTES_LINKS_AND_IMAGE_REFS_UNCHANGED; TWO_CRITERIA_SENTENCES_AND_TITLE_SPACE_ONLY',new_browser_runs=0,new_PNG_payload_reads=0,current_delta_worker_outer_tool_proof='OBSERVED_EXTERNALLY_AFTER_RETURN',publish_allowed=False,publish_performed=False)
    hq=c.write(out/'LOCAL_TEXT_DELTA_CANDIDATE_HANDOFF_V16.json',hand)
    c.need(c.sources()==source_before,'release3 source unchanged')
    print(json.dumps(dict(candidate=hq,plan=pq,HTML=files['index.html'],verification=verification,source_graph=graph,cutoff=facts['snapshot_utc'],new_PNG_reads=0,inherited_LOCAL_modes=4,inherited_LOCAL_combinations=16,publish_performed=False)),flush=True)

if __name__=='__main__':main()
