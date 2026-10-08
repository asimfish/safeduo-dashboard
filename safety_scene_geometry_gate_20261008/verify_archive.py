import json,hashlib,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests
p=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--commit',required=True);a=pa.parse_args();base=f'https://raw.githubusercontent.com/asimfish/safeduo-dashboard/{a.commit}/safety_scene_geometry_gate_20261008/';m=json.loads((p/'MEDIA_MANIFEST.json').read_text());entries=m['entries']+[dict(file='MEDIA_MANIFEST.json',sha256=hashlib.sha256((p/'MEDIA_MANIFEST.json').read_bytes()).hexdigest(),bytes=(p/'MEDIA_MANIFEST.json').stat().st_size)];out=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008_public_archive_mirror');out.mkdir(exist_ok=True)
def check(e):
    f=out/e['file']
    if f.exists():
        b=f.read_bytes()
        if len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']:return e
    response=requests.get(base+e['file'],timeout=90);response.raise_for_status();b=response.content;assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],e['file'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);return e
with ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,entries))
result=dict(status='PASS_ALL_PUBLIC_ARCHIVE_FILES_BYTE_AND_SHA256',commit=a.commit,base=base,files=len(checked),bytes=sum(x['bytes'] for x in checked),checked_utc=datetime.now(timezone.utc).isoformat(),mirror=str(out));(p/'ARCHIVE_DOWNLOAD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
