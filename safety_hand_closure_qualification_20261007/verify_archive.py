"""Download every immutable dataset entry; compare exact bytes and keep a mirror."""
import argparse,json,hashlib,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--local',type=Path,required=True);p.add_argument('--mirror',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.local/'MEDIA_MANIFEST.json').read_text());base=f'https://raw.githubusercontent.com/asimfish/safeduo-dashboard/{a.commit}/safety_hand_closure_qualification_20261007/';a.mirror.mkdir(exist_ok=False);results=[];errors=[]
entries=manifest['files']+[dict(file='MEDIA_MANIFEST.json',sha256=hashlib.sha256((a.local/'MEDIA_MANIFEST.json').read_bytes()).hexdigest(),bytes=(a.local/'MEDIA_MANIFEST.json').stat().st_size)]
def check(x):
 response=requests.get(base+x['file'],timeout=45);response.raise_for_status();body=response.content;digest=hashlib.sha256(body).hexdigest();assert digest==x['sha256'] and len(body)==x['bytes'],x['file'];assert body==(a.local/x['file']).read_bytes(),x['file'];dest=a.mirror/x['file'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body);return x
with ThreadPoolExecutor(max_workers=8) as pool:
 futures={pool.submit(check,x):x for x in entries}
 for i,f in enumerate(as_completed(futures)):
  try:results.append(f.result())
  except Exception as error:errors.append(dict(file=futures[f]['file'],error=repr(error)))
  if (i+1)%50==0:print('ARCHIVE_DOWNLOAD',i+1,'/',len(entries),'errors',len(errors),flush=True)
receipt=dict(status='PASS_EVERY_IMMUTABLE_ENTRY_DOWNLOADED' if not errors else 'FAIL_ARCHIVE_DOWNLOAD',commit=a.commit,files=len(results),bytes=sum(x['bytes'] for x in results),manifest_sha256=entries[-1]['sha256'],errors=errors,checked_utc=datetime.now(timezone.utc).isoformat())
a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert not errors
