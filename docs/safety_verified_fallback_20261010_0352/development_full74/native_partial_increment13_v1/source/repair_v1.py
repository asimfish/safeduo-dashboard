"""Only two reviewed wording/scope corrections; preserve the scientific instant."""
import argparse,ast,copy,html,json
from pathlib import Path
import report_v1 as r
OLD_CANDIDATE=r.rr(r.O/'release5/LOCAL_CLOSED_CANDIDATE_V13.json','6cf67291e89c47ea9b8083971f9c4ae80cce4668744b4b98a83fa47ae84fd2b4')
OLD_HANDOFF=r.rr(r.O/'release5/PARENT_HANDOFF_V13.json','665f959debdb4c20a645bc858ff890bf761941974cac530d0a1c03082f517543')
OLD_GRAPH=r.rr(r.H/'astra/report_increment13_v1/release5/SOURCE_GRAPH_V1.json','b104e03108f0b6a1e9b8e0308e79034084ae56c9b6510cbb35ef73cf8f1dfd1e')
REVIEW=r.rr(r.R/'astra_report13_independent_scientific_review_v1/actual03/REVIEW_V1.json','cacf555c867cf3d3db8a99a2a2d498fe5d8597d84cf2cffae038f40da7c8d4c0')
REVIEW_HANDOFF=r.rr(r.R/'astra_report13_independent_scientific_review_v1/actual03/FINAL_PARENT_REVIEW_HANDOFF_V1.json','b44e2cd6b9b0d83baaace49e804cc894535472c651bc793a506d1d1660b4818b')
CAMERA=r.rr(r.H/'CAMERA_SUFFICIENCY_ACCEPTANCE_SUPPLEMENT_V3.json','1069d1ed2a5be3683821d268cb5daa3cd48a8b9bb7dc056f7350ca7d1a688d58')
TEXT_EDITS=[('palm2048/finger512/body64/wide128、零覆盖缺口。','palm2048/finger512/body64；三幅广角中四臂与两桌各自峰值≥1024像素，关键原生forearm/wrist2 body各自峰值≥128像素；零覆盖缺口。'),('palm2048/finger512/body64/wide128,zero coverage gaps.','palm2048/finger512/body64; across the three wide views each of four arms and each of two tables must peak at≥1024 pixels, and each critical native forearm/wrist2 body at≥128 pixels; zero coverage gaps.')]
def main():
    a=argparse.ArgumentParser(allow_abbrev=False);a.add_argument('--out',required=True);a=a.parse_args();b=r.binding()
    old=b.read(OLD_CANDIDATE);oldh=b.read(OLD_HANDOFF);oldg=b.read(OLD_GRAPH);review=b.read(REVIEW);rh=b.read(REVIEW_HANDOFF)
    b.need(review['candidate']==rh['candidate']==OLD_CANDIDATE and review['author_handoff']==OLD_HANDOFF and rh['review']==REVIEW,'exact preserved NEEDS_FIX peer')
    criteria=b.read(CAMERA)['criteria'];b.need(criteria['each_of_four_arms_peak_pixels_min']==criteria['each_of_two_tables_peak_pixels_min']==1024 and criteria['critical_native_forearm_and_wrist2_peak_pixels_min']==128 and criteria['three_wide_views_required'],'original criteria unchanged')
    before={k:b.ref(v['path']) for k,v in oldg['own_source_files'].items()};b.need(before==oldg['own_source_files'],'old source remains byteexact')
    oldf=b.read(old['facts']);parent=b.read(oldf['static_refs']['V9_parent']);counts={};groups={}
    for role,key,exitcode in [('native','native_wait',-9),('CPU_check','CPU_check_wait',0),('outer','outer_wait',1)]:
        w=r.closed_wait(b,parent[key],exitcode);groups[role]={(i['pid'],i['startticks']) for i in w['owned_identities']};counts[role]=len(groups[role])
    campaign={(i['pid'],i['startticks']) for i in parent['fresh_owned_absence']}
    b.need(counts==dict(native=89,CPU_check=1,outer=1) and len(set.union(*groups.values()))==91 and set.union(*groups.values())==campaign,'campaign91 exact union89native+1CPU+1outer')
    packet=b.ref(r.D/'FROZEN_INPUTS_V1.json');newbuild=r.build(b,packet,Path(a.out));built=b.read(newbuild);plan=b.read(built['plan']);facts=b.read(built['facts'])
    expected=copy.deepcopy(oldf);expected['packet_ref']=packet
    expected['V9']['registered_campaign_owned_fresh_absent']=expected['V9'].pop('registered_native_owned_fresh_absent')
    expected['V9']['registered_identity_breakdown']=counts
    b.need(facts==expected,'only packet namespace and campaign scope labels differ in facts')
    oldliteral=dict(frozen_UTC=oldf['snapshot_utc'],V9=oldf['V9'],initial_numeric_scope=oldf['V9_initial_numeric_parent_diagnostic'],V8_reader=oldf['V8_reader'],foreign_geometry=oldf['foreign_geometry'],cstack=oldf['cstack'],held_state_scope=oldf['held_state_scope'])
    newliteral=dict(oldliteral,V9=facts['V9']);oldpage=b.buffer(old['HTML'],4*1024**2).decode();page=b.buffer(built['HTML'],4*1024**2).decode();expectedpage=oldpage
    for previous,replacement in TEXT_EDITS:
        b.need(expectedpage.count(previous)==1,'one inert bilingual text node per language');expectedpage=expectedpage.replace(previous,replacement,1)
    previous=html.escape(json.dumps(oldliteral,ensure_ascii=False,indent=2));replacement=html.escape(json.dumps(newliteral,ensure_ascii=False,indent=2))
    b.need(expectedpage.count(previous)==1,'exact original expert JSON text');expectedpage=expectedpage.replace(previous,replacement,1)
    b.need(page==expectedpage,'exact HTML: two camera text nodes and one campaign-label JSON block only')
    op=b.read(old['plan']);mp=b.read(plan['manifest']);oldm=b.read(old['manifest'])
    exceptions={'index.html','summary.json','panel_manifest.json',r.PREFIX+'/FACTS_INCREMENT13_V1.json'}
    shared=[]
    for name,src in op['file_sources'].items():
        if name in exceptions or name.startswith(r.PREFIX+'/source/'):continue
        b.need(plan['changed_files'][name]==src['sha256'],'same original image/sidecar/evidence bytes');shared.append(name)
    b.need(all(mp['files'][n]==sha for n,sha in oldm['files'].items() if n not in exceptions and not n.startswith(r.PREFIX+'/source/')),'all inherited assets and historical archives unchanged')
    oldsummary=b.read(op['file_sources']['summary.json']);newsummary=b.read(plan['file_sources']['summary.json']);es=copy.deepcopy(oldsummary);es['increment13']=facts
    b.need(newsummary==es,'summary only corrected typed scope')
    # The pre-existing four browser modes remain evidence for their old exact HTML.
    qa=b.read(old['local_browser']);b.need(qa['status']=='PASS_REAL_LOCAL_BROWSER' and len(qa['modes'])==4 and not qa['errors'] and all(qa['closed'].values()),'real previous four modes eligible for structural reuse')
    oldast=ast.parse(b.buffer(oldg['own_source_files']['report_v1.py'],2*1024**2));newast=ast.parse((r.D/'report_v1.py').read_bytes())
    unchanged=[]
    for name in ['numeric_scope','inputs','closed_wait','closed_tool']:
        left=[n for n in oldast.body if isinstance(n,ast.FunctionDef) and n.name==name];right=[n for n in newast.body if isinstance(n,ast.FunctionDef) and n.name==name]
        b.need(ast.dump(left[0],include_attributes=False)==ast.dump(right[0],include_attributes=False),'measurement/maths authority function unchanged '+name);unchanged.append(name)
    b.need(before=={k:b.ref(v['path']) for k,v in oldg['own_source_files'].items()},'old sources unchanged after correction')
    proof=dict(schema='astra.report13.narrow_camera_campaign_wording_delta.v1',status='PASS_EXACT_INERT_HTML_AND_METADATA_DELTA',old_candidate=OLD_CANDIDATE,old_handoff=OLD_HANDOFF,preserved_blocked_review=REVIEW,preserved_review_handoff=REVIEW_HANDOFF,original_camera_protocol=CAMERA,new_build=newbuild,new_plan=built['plan'],new_HTML=built['HTML'],new_facts=built['facts'],snapshot_utc=facts['snapshot_utc'],cutoff_unchanged=True,HTML_changes=['Two bilingual camera text nodes','Campaign identity-label/89+1+1 breakdown in existing expert JSON text'],DOM_elements_styles_controls_anchors_images_unchanged=True,source_functions_AST_byte_equivalent=unchanged,registered_campaign_count=91,campaign_identity_breakdown=counts,report_prior_CPU_scope_ID_count=51,campaign_and_report_scopes_separate=True,unchanged_original_small_file_names=shared,old_QA_reused=old['local_browser'],old_four_modes_are_old_actual_runs=True,new_fourmode_rerun=False,new_mobile_check_required=True,late_evidence_added=False,RAW_math_or_pixels_recomputed=False,native_GPU_SDK_Git_W_actions=0,bank_count=0,scientific_peer_pending=True)
    q=b.write(Path(a.out)/'NARROW_WORDING_DELTA_PROOF_V1.json',proof);print(json.dumps(dict(build=newbuild,proof=q)),flush=True)
if __name__=='__main__':main()
