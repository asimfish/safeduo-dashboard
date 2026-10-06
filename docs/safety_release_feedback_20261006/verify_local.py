"""Review the scientific package, source immutability and owned root change."""
import argparse,json,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();d=a.repo/'docs/safety_release_feedback_20261006'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((d/'PUBLIC_MANIFEST.json').read_text())
for rel,digest in manifest['files'].items():assert sha(d/rel)==digest,rel
data=json.loads((d/'data.json').read_text());result=json.loads((d/'RESULTS.json').read_text());reg=json.loads((d/'REGISTRATION.json').read_text())
assert {k:v for k,v in data['results'].items() if k!='summary'}==result
assert len(result['cases'])==24 and len(data['images'])==360 and len(data['videos'])==3
assert all(t['windows']==8 for t in result['totals'].values())
assert all(c['held_command_max_rad']==0 for c in result['cases'])
assert all(not c['accepted_task'] for c in result['cases'] if c['aborted'] or not c['complete_reference'])
assert reg['physical_task_gates']==json.loads((R.parent/'safety_object_binding_20261006/REGISTRATION_V5.json').read_text())['physical_task_gates']
archive=json.loads((d/'MEDIA_ARCHIVE.json').read_text());assert archive['status']=='PASS_ALL_PUBLIC_ARCHIVE_ZIP_SHA256'
for image in data['images']:
    rel='safety_release_feedback_20261006/block_'+str(image['block'])+'/'+image['file']
    assert archive['files'][rel]==image['sha256'] and image['url']==archive['raw_base_url']+rel
for c in result['cases']:
    key=str(c['block'])+'_'+str(c['env']);curve=data['curves'][key];assert len(curve['time_s'])==301
    groups=[i for i in data['images'] if i['block']==c['block'] and i['env']==c['env']]
    assert len(groups)==15 and len({i['step'] for i in groups})==5
root=(a.repo/'index.html').read_text();original=(R/'ROOT_BEFORE.html').read_text()
new=re.sub(r'<section class="sci-note" id="releaseFeedbackEvidence">.*?</section>\n','',root,count=1,flags=re.S)
assert new==original,'Unrelated root change'
assert 'id="trackingReserveEvidence"' in root and root.count('id="releaseFeedbackEvidence"')==1
assert json.loads((d/'SOURCE_AFTER.json').read_text())['changed_files']==[]
assert result['full_safety_accepted'] is False and result['new_final_trials']==0
receipt=dict(status='PASS_LOCAL_OWNED_PACKAGE_INTEGRITY_AND_ALL_CASE_BINDINGS',manifest_files=len(manifest['files']),cases=24,images=360,original_task_gates_unchanged=True,other_main_sections_byte_preserved=True,full_safety_accepted=False)
(R/'LOCAL_GATE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
