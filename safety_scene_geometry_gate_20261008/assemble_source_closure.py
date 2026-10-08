import json,hashlib,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
old=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006/safety_hand_closure_qualification_20261007');snapshot=json.loads((old/'SOURCE_SNAPSHOT.json').read_text());snapshot_map={x['original_path']:x for x in snapshot['files']};sources={};registrations=[]
for reg in sorted(p.glob('REGISTRATION_*.json')):
 r=json.loads(reg.read_text());registrations.append(dict(file=reg.name,sha256=sha(reg)))
 for path,h in r.get('source_sha256',{}).items():
  assert path not in sources or sources[path]==h,('conflicting_seals',path);sources[path]=h
out=p/'source_dependencies';out.mkdir(exist_ok=True);entries=[]
for path,h in sorted(sources.items()):
 f=Path(path);assert sha(f)==h,('registered_source_drift',path)
 if path in snapshot_map:
  x=snapshot_map[path];assert x['sha256']==h and sha(old/x['file'])==h;dest=x['file']
 elif f.is_relative_to(p):dest=str(f.relative_to(p))
 else:
  dest='source_dependencies/'+h[:16]+'_'+f.name;assert not (p/dest).exists() or sha(p/dest)==h;shutil.copy2(f,p/dest)
 entries.append(dict(original_path=path,file=dest,sha256=h,bytes=f.stat().st_size))
r=dict(status='PASS_ALL_REGISTERED_SOURCE_BYTES_AVAILABLE',registrations=registrations,files=entries,count=len(entries),bytes=sum(x['bytes'] for x in entries),scope='All Python/runtime/config/trajectory inputs explicitly listed in registrations. Not full native simulator, USD references or material closure.')
(p/'SOURCE_CLOSURE.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['files','registrations']})
