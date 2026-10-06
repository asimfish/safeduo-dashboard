"""Two closed raw reads and immutable metadata snapshot, own namespace only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,tarfile
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def files(root):
    result=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),p
        if p.is_file() and '__pycache__' not in p.parts:result.append(p)
    return result
def closed():
    for block in range(3):
        c=json.loads((RAW/f'holdout_{block}/campaign.json').read_text())
        assert c['status']=='complete' and len(c['jobs'])==4 and all(j['status']=='complete' for j in c['jobs'])
        for j in c['jobs']:
            p=Path('/proc')/str(j['pid'])
            if p.exists():
                try:argv=p.joinpath('cmdline').read_bytes().decode(errors='replace')
                except (FileNotFoundError,PermissionError):continue
                assert str(HERE) not in argv,'owned completed numeric child still exists'
    v=json.loads((HERE/'VISUAL_EFFECTIVE_EXECUTION.json').read_text())
    assert v['status']=='complete' and len(v['jobs'])==2 and all(j['status']=='complete' for j in v['jobs'])
    for j in v['jobs']:
        assert not (Path('/proc')/str(j['pid'])).exists(),'owned completed camera child still exists'
    assert json.loads((HERE/'ANALYSIS_EFFECTIVE_EXECUTION.json').read_text())['status']=='PASS_ALL4_SOURCE_BOUND_EFFECTIVE_READBACKS'
    # Query only owned namespace processes; never stop another task.
    names=['guard_runner.py','visual_runner.py','fresh_bank.py','execute_numeric.py','execute_block2_parallel.py',
           'execute_visual_parallel.py','execute_visual_parallel_v3.py','execute_camera_retry.py','follow_analysis.py','execute_readback_parallel.py','run_lp_compatible.py','finalize_analysis_effective.py']
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=[v.decode(errors='replace') for v in p.joinpath('cmdline').read_bytes().split(b'\0') if v]
            owns=(p.joinpath('cwd').resolve()==HERE) or any(str(HERE) in a for a in argv)
            active=any(a==n or a==str(HERE/n) for n in names for a in argv[1:])
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if owns and active:raise ValueError('owned writer still alive: '+p.name)
def raw():
    closed();assert not (RAW/'final_evidence').exists()
    paths=files(RAW);rows=[]
    for i,p in enumerate(paths):
        st=p.stat();digest=sha(p);assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        rows.append(dict(path=str(p.relative_to(RAW)),bytes=st.st_size,sha256=digest))
        if i%80==0:print('RAW_FIRST_READ',i,len(paths),flush=True)
    assert files(RAW)==paths
    for i,r in enumerate(rows):
        p=RAW/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        if i%80==0:print('RAW_SECOND_READ',i,len(rows),flush=True)
    assert files(RAW)==paths;closed();by_path={r['path']:r for r in rows}
    result=json.loads((HERE/'holdout_results.json').read_text());bound=0
    for row in result['rows']:
        for rel,d in row['input_sha256'].items():
            key=str((Path(row['path'])/rel).relative_to(RAW));assert by_path[key]['sha256']==d;bound+=1
    h6=json.loads((HERE/'H6_PREDICTION_AUDIT.json').read_text());h6_bound=0
    for row in h6['rows']:
        for path,d in row['input_sha256'].items():assert by_path[str(Path(path).relative_to(RAW))]['sha256']==d;h6_bound+=1
    visual=json.loads((HERE/'visual_verification.json').read_text());camera_bound=0
    roots=json.loads((HERE/'VISUAL_EFFECTIVE_PLAN.json').read_text())['actual_visual_roots']
    for g in visual['galleries']:
        for rel,d in [(g['state'],g['state_sha256']),(g['before'],g['before_sha256']),(g['after'],g['after_sha256']),*[(i['path'],i['sha256']) for i in g['images']]]:
            key=str((Path(roots[g['mode']])/rel).relative_to(RAW));assert by_path[key]['sha256']==d;camera_bound+=1
    write(HERE/'raw_evidence_manifest.json',dict(status='PASS_TWO_READS_CLOSED_RAW',root=str(RAW),count=len(rows),
        bytes=sum(r['bytes'] for r in rows),files=rows,parent_raw_hashes_bound=bound,h6_raw_hashes_bound=h6_bound,
        camera_raw_hashes_bound=camera_bound,utc=datetime.now(timezone.utc).isoformat(),scope='all owned closed raw files, two reads, including every retained source/scheduling outcome; finalmetadata appended separately'))
    print('RAW_TWO_READS_PASS',len(rows),sum(r['bytes'] for r in rows),flush=True)
def metadata(initial):
    closed();required=['raw_evidence_manifest.json','remote_verification_initial.json','live_browser_initial.json'] if initial else ['raw_evidence_manifest.json','remote_verification.json','live_browser.json','VERIFICATION.json','FINAL_DELIVERY_CLOSURE.json']
    for n in required:assert json.loads((HERE/n).read_text())['status'].startswith('PASS'),n
    dest=RAW/'final_evidence'/('metadata_initial' if initial else 'metadata_final');dest.mkdir(parents=True,exist_ok=False)
    excludes={'SEALED.json','metadata_archive_receipt.json','metadata_archive_initial.log','metadata_archive.log','preview_server.log'};rows=[]
    for p in files(HERE):
        rel=str(p.relative_to(HERE))
        if rel in excludes:continue
        dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True);d=sha(p);shutil.copy2(p,dst);assert sha(p)==sha(dst)==d
        rows.append(dict(path=rel,bytes=dst.stat().st_size,sha256=d))
    write(dest/'snapshot_manifest.json',dict(count=len(rows),files=rows,excludes_self=True,excluded_runtime=['__pycache__',*sorted(excludes)]))
    archive=dest.parent/('metadata_initial.tar.gz' if initial else 'metadata_final.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname='metadata')
    digest=sha(archive)
    with tarfile.open(archive,'r:gz') as tar:
        for r in rows:
            f=tar.extractfile('metadata/'+r['path']);assert f is not None;h=hashlib.sha256()
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==r['sha256']
        f=tar.extractfile('metadata/snapshot_manifest.json');assert f and hashlib.sha256(f.read()).hexdigest()==sha(dest/'snapshot_manifest.json')
    assert sha(archive)==digest
    receipt=dict(status='PASS_METADATA_COPY_TAR_READBACK',archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
        metadata_count=len(rows),metadata_bytes=sum(r['bytes'] for r in rows),raw_manifest_sha256=sha(HERE/'raw_evidence_manifest.json'),utc=datetime.now(timezone.utc).isoformat(),hardware_approved=False,production_promoted=False)
    write(HERE/('metadata_archive_receipt_initial.json' if initial else 'metadata_archive_receipt.json'),receipt)
    if not initial:write(HERE/'SEALED.json',dict(status='SEALED',receipt=receipt,immutable_scope='rawmanifest files plus metadata snapshots/tars; futurework uses newnamespace'))
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['raw','metadata_initial','metadata']);args=p.parse_args()
    raw() if args.stage=='raw' else metadata(args.stage=='metadata_initial')
