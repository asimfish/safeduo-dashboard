"""Closed namespace only: two complete raw reads and metadata member readback."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,tarfile

HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def closed():
    assert json.loads((HERE/'NUMERIC_EXECUTION.json').read_text())['status']=='PASS_ALL_NUMERIC_CLOSED'
    assert json.loads((HERE/'RECOVERY_EXECUTION.json').read_text())['status']=='PASS_COMBINED_RECOVERY_WITH_RETAINED_FAILURES'
    assert json.loads((HERE/'RECOVERY_TERMINAL_CLOSURE.json').read_text())['status'].startswith('PASS')
    assert json.loads((HERE/'CAMERA_AMENDMENT_EXECUTION_BINDING.json').read_text())['status'].startswith('PASS')
    first=json.loads((HERE/'initial_attempts/holdout_results.json').read_text());assert first['completed_method_windows']==448 and first['invalid_method_windows']==128
    for block in range(3):
        c=json.loads((RAW/f'holdout_{block}/campaign.json').read_text());assert c['status']=='complete' and len(c['jobs'])==3
        for j in c['jobs']:
            assert j['status']=='complete' and j['exit_code']==0
            proc=Path('/proc')/str(j['pid'])
            if proc.exists():
                assert str(HERE).encode() not in proc.joinpath('cmdline').read_bytes(),'owned numerical child still exists'
    v=json.loads((HERE/'visual_execution.json').read_text());assert v['status']=='complete' and len(v['jobs'])==2
    for j in v['jobs']:
        assert j['status']=='complete' and j['exit_code']==0
        proc=Path('/proc')/str(j['pid'])
        if proc.exists():assert str(HERE).encode() not in proc.joinpath('cmdline').read_bytes(),'owned camera child still exists'
    assert json.loads((HERE/'ALL_ANALYSIS_EXECUTION.json').read_text())['status']=='PASS_ALL6_FRESH_ANALYSIS_JOBS'
    assert json.loads((HERE/'ASTRA_ZERO_FINAL_REVIEW.json').read_text())['status'].startswith('PASS')
    names=['guard_runner.py','visual_runner.py','fresh_bank.py','execute_numeric.py','execute_all.py','execute_visual.py','follow_analysis.py','execute_analysis.py','failure_audit.py','audit_fixed_feasibility.py','audit_commands.py','verify_visual.py','audit_camera_state.py','recover_unstarted.py','follow_recovery_analysis.py','audit_zero_intent.py','finalize_recovery_termination.py','recover_original_camera.py','close_recovered_study.py']
    for p in Path('/proc').glob('[0-9]*'):
        try:args=[a.decode(errors='replace') for a in p.joinpath('cmdline').read_bytes().split(b'\0') if a]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(a==str(HERE/n) for n in names for a in args[1:]):raise ValueError('owned writer still alive '+p.name)


def paths(root):
    values=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),p
        if p.is_file() and '__pycache__' not in p.parts:values.append(p)
    return values


def raw():
    closed();assert not (RAW/'final_evidence').exists();files=paths(RAW)
    def inspect(p):
        before=p.stat();s=sha(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        return dict(path=str(p.relative_to(RAW)),bytes=before.st_size,sha256=s)
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(inspect,files))
    print('RAW_FIRST_COMPLETE',len(rows),flush=True);assert paths(RAW)==files
    with ThreadPoolExecutor(max_workers=2) as pool:second=list(pool.map(inspect,files))
    assert rows==second and paths(RAW)==files;closed()
    by_path={r['path']:r for r in rows};bound=0
    result=json.loads((HERE/'holdout_results.json').read_text())
    for condition in result['rows']:
        for rel,s in condition['input_sha256'].items():
            key=str((Path(condition['path'])/rel).relative_to(RAW));assert by_path[key]['sha256']==s;bound+=1
    independent=json.loads((HERE/'ASTRA_ZERO_FINAL_SCORE.json').read_text());independent_bound=0
    for path,s in independent['input_sha256'].items():
        file=Path(path)
        if file.is_relative_to(RAW):
            key=str(file.relative_to(RAW));assert by_path[key]['sha256']==s;independent_bound+=1
    assert independent_bound>0,'independent raw-file identities must bind the closed archive'
    visual=json.loads((HERE/'visual_verification.json').read_text());camera_bound=0
    for g in visual['galleries']:
        for rel,s in [(g['state'],g['state_sha256']),(g['before'],g['before_sha256']),(g['after'],g['after_sha256']),*[(im['path'],im['sha256']) for im in g['images']]]:
            key='visual/'+g['mode']+'/'+rel;assert by_path[key]['sha256']==s;camera_bound+=1
    write(HERE/'raw_evidence_manifest.json',dict(status='PASS_TWO_CLOSED_RAW_READS',root=str(RAW),count=len(rows),bytes=sum(r['bytes'] for r in rows),files=rows,analysis_bound_files=bound,independent_raw_bound_files=independent_bound,camera_bound_files=camera_bound,workers=2,utc=datetime.now(timezone.utc).isoformat()))
    print('RAW_TWO_READS_COMPLETE',len(rows),sum(r['bytes'] for r in rows),flush=True)


def metadata(final):
    closed();assert json.loads((HERE/'raw_evidence_manifest.json').read_text())['status']=='PASS_TWO_CLOSED_RAW_READS'
    required=['PUBLIC_initial.json','live_initial_browser.json'] if not final else ['PUBLIC_final.json','FINAL_DELIVERY_CLOSURE.json','VERIFICATION.json']
    for name in required:assert json.loads((HERE/name).read_text())['status'].startswith('PASS'),name
    dest=RAW/'final_evidence'/('metadata_final' if final else 'metadata_initial');dest.mkdir(parents=True,exist_ok=False)
    exclusions={'SEALED.json','metadata_archive_receipt.json','metadata_archive_initial.log','metadata_archive_final.log','raw_seal.log'}
    rows=[]
    for p in paths(HERE):
        rel=str(p.relative_to(HERE))
        if rel in exclusions:continue
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);s=sha(p);shutil.copy2(p,target);assert sha(p)==sha(target)==s
        rows.append(dict(path=rel,bytes=target.stat().st_size,sha256=s))
    write(dest/'snapshot_manifest.json',dict(files=rows,count=len(rows),excluded=sorted(exclusions),excluded_self=True))
    archive=dest.parent/('metadata_final.tar.gz' if final else 'metadata_initial.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname='metadata')
    s=sha(archive)
    with tarfile.open(archive,'r:gz') as tar:
        for row in rows:
            item=tar.extractfile('metadata/'+row['path']);assert item is not None
            h=hashlib.sha256()
            for block in iter(lambda:item.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==row['sha256']
        item=tar.extractfile('metadata/snapshot_manifest.json');assert hashlib.sha256(item.read()).hexdigest()==sha(dest/'snapshot_manifest.json')
    assert sha(archive)==s
    receipt=dict(status='PASS_METADATA_TAR_FULL_MEMBER_READBACK',archive=str(archive),archive_sha256=s,archive_bytes=archive.stat().st_size,metadata_count=len(rows),metadata_bytes=sum(r['bytes'] for r in rows),raw_manifest_sha256=sha(HERE/'raw_evidence_manifest.json'),utc=datetime.now(timezone.utc).isoformat())
    write(HERE/('metadata_archive_receipt.json' if final else 'metadata_archive_receipt_initial.json'),receipt)
    if final:write(HERE/'SEALED.json',dict(status='SEALED',receipt=receipt,scope='all raw manifest files and initial/final metadata member-verified archives; futurework separate namespace',hardware_approved=False,production_promoted=False))
    print('METADATA_COMPLETE',receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['raw','metadata_initial','metadata_final']);args=p.parse_args()
    raw() if args.stage=='raw' else metadata(args.stage=='metadata_final')
