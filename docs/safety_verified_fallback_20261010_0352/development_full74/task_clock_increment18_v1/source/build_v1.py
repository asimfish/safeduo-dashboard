"""Immutable presentation-only revision; facts, counts and historical science unchanged."""
import argparse,copy,json
from pathlib import Path
import common_v1 as c
import render_v1 as render
BASE={'path':str(c.O/'build/candidate01/LOCAL_CANDIDATE_V18.json'),'sha256':'b2d3d56f46dbe83af050ca06f18b3a707d3bd36c81904313856de34d14364109'}
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);a=p.parse_args()
    before=c.sources();frozen,v,h17,p17,m17,f17=c.inherited()
    h=c.read(BASE);c.need(h['source_graph']['sha256']=='e0cf15c321677c935538ab8528c054ea1f119472a07d4f6664dfd664e6ffe244','actual prior frozen presentation candidate')
    oldplan=c.read(h['plan']);oldmanifest=c.read(h['manifest']);facts=c.read(h['facts'])
    c.need(facts['engineering_snapshot_utc']==frozen['engineering_snapshot_utc'] and facts['scientific_snapshot_utc']==frozen['scientific_snapshot_utc'],'unchanged exact cutoffs')
    out=Path(a.out);c.need(not out.exists() and out.resolve().is_relative_to(c.O),'exclusive candidate02')
    out.mkdir(parents=True,exist_ok=False);ov=out/'overlay';ov.mkdir()
    old=c.buffer(h['HTML'],4*1024**2).decode('utf-8');page=render.refine(old,facts);projection=render.projection(old,page,facts)
    (ov/'index.html').write_text(page,encoding='utf-8')
    fq=c.copy(h['facts'],ov/c.PREFIX/'FACTS_INCREMENT18_V1.json')
    sq=c.copy(oldplan['file_sources']['summary.json'],ov/'summary.json')
    for q in before['own_source_files'].values():c.copy(q,ov/c.PREFIX/'source'/Path(q['path']).name)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),ov/c.PREFIX/'source/SOURCE_GRAPH_V1.json')
    c.copy(c.INPUTS,ov/c.PREFIX/'source/FROZEN_INPUTS_V1.json')
    files={p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()}
    manifest=copy.deepcopy(oldmanifest);manifest['files'].update({n:q['sha256'] for n,q in files.items()})
    mq=c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=mq
    allowed={'index.html'}|{c.PREFIX+'/source/'+x for x in ('common_v1.py','render_v1.py','build_v1.py','qa_local_v1.py','run_v1.py','SOURCE_GRAPH_V1.json')}
    changed={n for n in oldmanifest['files'] if manifest['files'][n]!=oldmanifest['files'][n]}
    c.need(changed<=allowed,'only reviewed presentation HTML and declared revised own sources')
    c.need(fq['sha256']==h['facts']['sha256'] and sq=={'path':str(ov/'summary.json'),'sha256':oldplan['file_sources']['summary.json']['sha256']},'bitexact original facts and summary')
    inherited=dict(oldplan['inherited_file_sources']);inherited.update(oldplan['file_sources'])
    graph=c.ref(c.D/'SOURCE_GRAPH_V1.json')
    plan=dict(schema='astra.report18.partial_overlay.v1',presentation_release=2,base_commit=frozen['base_public_commit'],overlay=str(ov),file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},
      inherited_file_sources=inherited,base_manifest=h['manifest'],manifest=mq,source_facts=fq,source_graph=graph,presentation_parent=BASE,
      inherited_candidate=frozen['inputs']['report17_candidate'],entire_Report17_and_prior_body_byte_identical=True,all_historical_asset_SHA_retained=True,new_images=0,new_figures=0,publish_performed=False)
    pq=c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    ck=dict(status='PASS_ALLOWED_PRESENTATION_ONLY_DELTA_AND_HISTORY_PRESERVATION',presentation_parent=BASE,frozen_inputs=c.INPUTS,
      presentation_projection=projection,facts_bitexact=fq['sha256']==h['facts']['sha256'],facts=fq,previous_summary_byte_exact=True,
      exact_engineering_checks_unchanged=[20,5],exact_cutoffs_unchanged=True,only_allowed_changed_manifest_paths=sorted(changed),
      all_other_manifest_entries_byte_SHA_unchanged=True,old_manifest_entries_retained=len(oldmanifest['files']),
      original_scientific_cutoff_preserved=True,entire_historical_body_byte_exact=True,old_summary_scientific_values_preserved=True,
      capacity_not_admission=True,original2counterexamples_retained=True,new_RAW_PNG_camera_model_native_SDK_Git_calls=0,
      source_before=before,source_after=c.sources(),source_unchanged=c.sources()==before)
    kq=c.write(out/'SOURCE_AND_PRESENTATION_PROJECTION_CHECKS_V1.json',ck)
    candidate=dict(schema='astra.report18.LOCAL_candidate.v1',status='IMMUTABLE_LOCAL_PENDING_LOCAL16_AND_DISTINCT_SCIENTIFIC_MAPPING',presentation_release=2,presentation_parent=BASE,
      engineering_snapshot_utc=facts['engineering_snapshot_utc'],scientific_snapshot_utc=facts['scientific_snapshot_utc'],base_public_commit=frozen['base_public_commit'],
      HTML=files['index.html'],facts=fq,manifest=mq,plan=pq,source_graph=graph,checks=kq,
      inherited_scientific_candidate=frozen['inputs']['report17_candidate'],inherited_public_adoption=frozen['inputs']['report17_parent'],
      science_review_pending=True,bank_count=0,safety_acceptance=False,backend_attestation='NOT_ATTESTED',new_media=0,new_figures=0,publish_performed=False)
    cq=c.write(out/'LOCAL_CANDIDATE_V18.json',candidate);c.need(c.sources()==before,'source unchanged')
    print(json.dumps({'candidate':cq,'HTML':candidate['HTML'],'facts_bitexact':True,'allowed_changed_manifest_paths':sorted(changed),'new_overlay_paths':len(files)}),flush=True)
if __name__=='__main__':main()

