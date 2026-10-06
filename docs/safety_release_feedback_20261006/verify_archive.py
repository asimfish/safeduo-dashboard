import argparse,json,hashlib,urllib.request,zipfile,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
manifest=json.loads((a.repo/'safety_release_feedback_20261006/MEDIA_MANIFEST.json').read_text())
cache=Path('/mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006')/(a.commit+'.zip')
url='https://codeload.github.com/asimfish/safeduo-dashboard/zip/'+a.commit
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-artifact-integrity'}),timeout=60) as response,cache.open('wb') as f:
    while True:
        chunk=response.read(4*1024*1024)
        if not chunk:break
        f.write(chunk)
with zipfile.ZipFile(cache) as z:
    roots={n.split('/')[0] for n in z.namelist()};assert len(roots)==1;root=next(iter(roots))+'/'
    assert z.read(root+'safety_release_feedback_20261006/MEDIA_MANIFEST.json')==(a.repo/'safety_release_feedback_20261006/MEDIA_MANIFEST.json').read_bytes()
    for rel,digest in manifest['files'].items():assert hashlib.sha256(z.read(root+rel)).hexdigest()==digest,rel
receipt=dict(status='PASS_ALL_PUBLIC_ARCHIVE_ZIP_SHA256',commit=a.commit,branch='exp/release-feedback-media-20261006',raw_base_url='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+a.commit+'/',archive_zip_url=url,files=manifest['files'],checked_files=len(manifest['files']),archive_bytes=cache.stat().st_size,archive_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/'MEDIA_ARCHIVE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
