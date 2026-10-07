from pathlib import Path
import shutil,json,hashlib,re
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007');dst=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006/safety_hand_closure_qualification_20261007');assert not dst.exists();dst.mkdir();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for run in ['development','validation','validation_v2','validation_v3','validation_fresh_v3','paired_v3']:
 assert (raw/run/'recording_receipt.json').exists() and not (raw/run/'failure.txt').exists();shutil.copytree(raw/run,dst/run)
for folder in ['runtime_v2','runtime_v3','source_snapshot','videos','browser_local']:
 if (p/folder).exists():shutil.copytree(p/folder,dst/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for f in p.iterdir():
 if f.is_file() and (f.suffix in ['.py','.json','.md','.log','.html','.js','.txt','.gz']):shutil.copyfile(f,dst/f.name)
patterns=[re.compile(rb'sk-[A-Za-z0-9]{20,}'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),re.compile(rb'Bearer ct-[a-f0-9]{20,}'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')];hits=[];files=[];largest=0;total=0
for f in sorted(dst.rglob('*')):
 if not f.is_file():continue
 f.chmod(0o644);size=f.stat().st_size;assert size<100*1024*1024,('GitHub blob cap',str(f),size);largest=max(largest,size);total+=size
 if f.suffix in ['.py','.json','.md','.log','.html','.js','.txt','.yaml','.yml']:
  body=f.read_bytes()
  if any(pattern.search(body) for pattern in patterns):hits.append(str(f.relative_to(dst)))
 files.append(dict(file=str(f.relative_to(dst)),bytes=size,sha256=sha(f)))
assert not hits,('credential pattern hits: filenames only',hits)
(dst/'MEDIA_MANIFEST.json').write_text(json.dumps(dict(status='FROZEN_NATIVE_EVIDENCE_DATASET',files=files,file_count=len(files),bytes=total,largest_file_bytes=largest,credential_pattern_scan_hits=hits,scope='All6 native run directories, original images, force points, source snapshot, preserved failures, scripts, protocols, local preflight receipt and8 videos in2 codecs. Publication/source/asset hashes do not certify full robot safety.'),indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=total,largest=largest,credential_scan_hits=hits),indent=2))
