import json,hashlib,re,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;dest=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006')/p.name
for name in ['browser_local']:
 assert (p/name).is_dir();shutil.copytree(p/name,dest/name,dirs_exist_ok=True)
for name in ['MAIN_PAYLOAD_GATE.json','LOCAL_ACCEPTANCE_MATRIX.json']:
 assert (p/name).exists();shutil.copy2(p/name,dest/name)
secret=re.compile(rb'(?:Bearer\s+ct-[0-9a-f]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})');entries=[]
for f in sorted(dest.rglob('*')):
 if not f.is_file() or f.name=='MEDIA_MANIFEST.json':continue
 b=f.read_bytes();assert len(b)<100*1024**2;assert not secret.search(b),('credential pattern',f);entries.append(dict(file=str(f.relative_to(dest)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
sources=json.loads((dest/'SOURCE_CLOSURE.json').read_text())
for e in sources['files']:assert hashlib.sha256((dest/e['file']).read_bytes()).hexdigest()==e['sha256']
r=dict(status='PASS_ALL_LOCAL_FILES_HASHED_AND_CREDENTIAL_PATTERN_SCAN',entries=entries,count=len(entries),bytes=sum(e['bytes'] for e in entries),largest=max(e['bytes'] for e in entries),source_closure_files=len(sources['files']),scope='Full original native recorded points, all negative/incomplete attempts, source inputs, screenshots, original and labelled replay clips, independent gates and local browser. Manifest self excluded. Public delivery gates generated subsequently.')
for f in [dest/'MEDIA_MANIFEST.json',p/'MEDIA_MANIFEST.json']:f.write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k!='entries'})
