"""Read every native archive byte back from the exact public Git commit."""
import argparse,json,hashlib,urllib.request,concurrent.futures,datetime,time
from pathlib import Path
R=Path(__file__).resolve().parent

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SafeDuo-native-media-sha'}),timeout=40) as response:return response.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    raw=(a.repo/'MEDIA_MANIFEST.json').read_bytes();manifest=json.loads(raw);files=manifest['files']
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+a.commit+'/'
    assert get(base+'MEDIA_MANIFEST.json')==raw
    def check(item):
        rel,digest=item
        for attempt in range(3):
            try:
                got=hashlib.sha256(get(base+rel)).hexdigest();assert got==digest,(rel,got,digest);return rel
            except (OSError,TimeoutError):
                if attempt==2:raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,files.items()))
    v=dict(status='PASS_ALL_REMOTE_MEDIA_SHA',commit=a.commit,branch='exp/object-binding-media-20261006',raw_base_url=base,archive_zip_url='https://github.com/asimfish/safeduo-dashboard/archive/'+a.commit+'.zip',manifest_sha256=hashlib.sha256(raw).hexdigest(),checked_files=len(checked),files=files,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='all native source media, dense record and metadata SHA read from exact public commit; archive ZIP composition not downloaded in this check; no scientific acceptance')
    (R/'MEDIA_ARCHIVE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k!='files'}))
if __name__=='__main__':main()
