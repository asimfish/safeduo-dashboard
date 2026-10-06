"""Two independent full reads of every original asset at a fixed public commit."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from verify_remote import get,NETWORK_FAILURES
HERE=Path(__file__).resolve().parent
WORK=Path('/home/liyufeng/safeduo-dashboard-fifo-assets-20261006')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip();assert len(commit)==40
    branch='exp/fifo-predictor-assets-20261006'
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=WORK,text=True).split()[0];assert remote==commit
    paths=sorted(p for p in WORK.rglob('*') if p.is_file() and p.name!='.git' and '.git' not in p.parts)
    frozen={p.relative_to(WORK).as_posix():dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in paths}
    assert set(frozen)=={HERE.name+'/'+r['path'] for r in json.loads((HERE/'ASSET_BUILD.json').read_text())['files']}|{'README.md',HERE.name+'/asset_manifest.json'}
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+commit+'/'
    readings=[]
    for turn in (1,2):
        def read(rel):
            body=get(base+rel+'?fifo_read='+str(turn));v=frozen[rel];assert len(body)==v['bytes'] and sha(body)==v['sha256'],rel
            return dict(path=rel,**v)
        with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(read,frozen))
        readings.append(rows);print('ASSET_FULL_READ_PASS',turn,len(rows),sum(r['bytes'] for r in rows),flush=True)
    assert readings[0]==readings[1]
    assert all(sha((WORK/p).read_bytes())==v['sha256'] for p,v in frozen.items())
    receipt=dict(status='PASS_TWO_PUBLIC_FULL_READS_PINNED_ASSETS',commit=commit,branch=branch,base=base,count=len(frozen),
        bytes=sum(r['bytes'] for r in readings[0]),files=readings[0],read_rounds=2,
        source_sha256=sha(Path(__file__).read_bytes()),build_sha256=sha((HERE/'ASSET_BUILD.json').read_bytes()),network_failures_retried=NETWORK_FAILURES,
        utc=datetime.now(timezone.utc).isoformat(),scope='unaltered PNG/state/nativeNPZ/causalfirstfail/inputrecipes/banks and lossless complete JSONgzip; all bytes publicly read twice')
    with (HERE/'ASSET_PUBLICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'],commit,flush=True)
if __name__=='__main__':main()
