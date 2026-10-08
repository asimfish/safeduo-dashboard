"""Close post-processing additions before the immutable asset commit."""
from pathlib import Path
import json,hashlib,shutil,datetime,re,ast
H=Path(__file__).resolve().parent
A=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007')/H.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for n in ['acceptance_assets_build','acceptance_astra_decision','acceptance_panel_data']:
        assert json.loads((H/(n+'_execution.json')).read_text())['actual_exit']==0,n
    old=(A/'asset_manifest.json').read_bytes()
    with (H/'ASSET_MANIFEST_BEFORE_SUPPLEMENT.json').open('xb') as f:f.write(old)
    names=['ADVERSE_CASE_DIAGNOSIS.json','ADVERSE_CASE_DIAGNOSIS.md','PANEL_DATA_VERIFICATION.json','acceptance_panel_data_execution.json']
    for n in names:shutil.copyfile(H/n,A/n)
    for p in H.glob('*.py'):
        ast.parse(p.read_text());shutil.copyfile(p,A/'sources'/p.name)
    patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[0-9A-Z]{16}',rb'sk-(?:proj-)?[A-Za-z0-9_-]{35,}']
    inspected=0
    for p in A.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.py','.md','.html']:
            data=p.read_bytes();assert not any(re.search(x,data) for x in patterns),'secret-shaped export: '+str(p);inspected+=1
    rows=[dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(A.rglob('*')) if p.is_file() and p.name!='asset_manifest.json']
    assert all(r['bytes']<95*1024**2 for r in rows),'oversized GitHub individual blob'
    manifest=json.loads(old);manifest.update(files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),postprocessing_supplement=names,public_source_secret_patterns_inspected=inspected,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (A/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt=dict(status='PASS_CLOSED_PRECOMMIT_ASSET_SUPPLEMENT',previous_manifest_sha256=hashlib.sha256(old).hexdigest(),final_manifest_sha256=sha(A/'asset_manifest.json'),files=len(rows),bytes=manifest['bytes'],source_secret_pattern_scan_files=inspected,science_decision_unchanged=True,additional_files=names)
    with (H/'PANEL_ASSET_SUPPLEMENT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt,flush=True)
if __name__=='__main__':main()
