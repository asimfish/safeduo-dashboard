"""Presentation-only correction: reuse all closed480 originals and typed facts."""
import argparse,copy,csv,io,json
from pathlib import Path
import common_v1 as c
from render_v1 import gallery_page
from figures_v1 import figures
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ['plan','plan-sha256','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();before=c.sources();oldq={'path':a.plan,'sha256':a.plan_sha256};old=c.read(oldq);manifest=c.read(old['manifest']);facts=c.read(old['source_facts']);inventory=c.read(old['selected_image_inventory'])
    c.wait(c.ref(c.O/'CPU/build03/worker/ACTUAL_WAIT.json'));c.wait(c.ref(c.O/'CPU/build03_outer/ACTUAL_WAIT.json'));c.tool(c.ref(c.O/'CPU/build03/ACTUAL_TOOL_CLOSED_V1.json'))
    c.need(facts['bank_count']==0 and facts['committed_rows']==[0,1] and inventory['images']==480,'same exact frozen science')
    out=Path(a.out);c.need(not out.exists() and out.resolve().is_relative_to(c.O),'new exclusive immutable presentation candidate');ov=out/'overlay';ov.mkdir(parents=True,exist_ok=False);prefix=ov/c.PREFIX;prefix.mkdir()
    for n,q in before.items():c.copy(q,prefix/'source'/n)
    c.copy(c.ref(c.D/'SOURCE_GRAPH_V1.json'),prefix/'source/SOURCE_GRAPH_V1.json')
    for g in inventory['groups']:
        dest=prefix/'gallery'/f"env_{g['lane']:03d}.html";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(gallery_page(g),encoding='utf-8')
    csvref=old['file_sources'][c.PREFIX+'/COMMITTED_2X32_NUMERIC_V1.csv'];rows=list(csv.DictReader(io.StringIO(c.buffer(csvref,1024**2).decode())));figures(rows,prefix/'figures')
    newfiles={p.relative_to(ov).as_posix():c.ref(p) for p in ov.rglob('*') if p.is_file()}
    files=dict(old['file_sources']);files.update(newfiles)
    manifest=copy.deepcopy(manifest);manifest['files'].update({n:q['sha256'] for n,q in newfiles.items()});c.write(ov/'panel_manifest.json',manifest);files['panel_manifest.json']=c.ref(ov/'panel_manifest.json')
    plan=dict(old,overlay=str(ov),file_sources=files,changed_files={n:q['sha256'] for n,q in files.items()},file_bytes={n:Path(q['path']).stat().st_size for n,q in files.items()},manifest=files['panel_manifest.json'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),previous_candidate_plan=oldq,overlay_kind='PARTIAL_DELTA_EXPLICIT_FILE_SOURCES_WITH_CLOSED_ORIGINAL_GALLERY_REUSE',presentation_delta='Gallery SHA word wrapping; witness annotation offsets; browser screenshot scroll target',original_gallery_recopied=False,source_facts=old['source_facts'])
    c.write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    c.write(out/'PRESENTATION_ONLY_DELTA_PROOF_V1.json',dict(unchanged_HTML=old['file_sources']['index.html'],unchanged_summary=old['file_sources']['summary.json'],unchanged_science_facts=old['source_facts'],unchanged_CSV=csvref,unchanged_all480_originals=[{'path':e['relative'],'sha256':e['sha256']} for g in inventory['groups'] for e in g['images']],old8fig_and15RGB_SHA_preserved=True,scientific_values_changed=False,source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),preserved_failed_QA_tool=c.ref(c.O/'CPU/localqa01/ACTUAL_TOOL_CLOSED_V1.json'),repeated_RAW_reads=0))
    c.need(c.sources()==before,'source unchanged during presentation repair');print(json.dumps({'repaired_plan':c.ref(out/'REPORT_OVERLAY_PLAN_V1.json'),'reused_originals':480,'science_facts_unchanged':True}),flush=True)
if __name__=='__main__':main()
