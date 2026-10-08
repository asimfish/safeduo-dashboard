"""Seal last, after actual exits, public readback, deployed CI and browser closure."""
from pathlib import Path
import datetime,hashlib,json,shutil,tarfile
H=Path(__file__).resolve().parent
R=Path('/mnt/nas/data/lyf/double_hand')/H.name
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):d.update(b)
    return d.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def stat(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def main():
    assert not (H/'SEALED.json').exists() and not (R/'SEALED.json').exists()
    closed=json.loads((H/'FINAL_DELIVERY_CLOSURE.json').read_text())
    assert closed['status']=='PASS_COMPLETE_FINITE_EXPERIMENT_DELIVERY_CLOSED'
    assert all(closed[k] for k in ['astra_closed','preview_closed','writers_closed'])
    assert json.loads((H/'PREVIEW_EXIT.json').read_text())['actual_exit']==-15
    for n,status in [('PUBLIC_final.json','PASS_DEPLOYED_EXACT_COMMIT_TWO_BYTE_READS'),('ASSET_PUBLICATION.json','PASS_TWO_PUBLIC_BYTE_READS'),('live_browser.json','PASS_ACTUAL_CHROMIUM_NATIVE_EVIDENCE_JOURNEYS'),('legacy_live_browser.json','pass')]:
        assert json.loads((H/n).read_text())['status']==status,n
    raw=json.loads((H/'RAW_CLOSED_READBACK.json').read_text());assert raw['status']=='PASS_ACTUAL_ALL_CLOSED_RAW_FILE_HASH_READBACK'
    assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}=={r['path'] for r in raw['files']}
    # The actual complete byte read is a separately waited process. All science
    # writers were closed before that read; every file's inode, size and mtime
    # must still match. This does not relabel stat checks as another byte read.
    for row in raw['files']:assert stat(R/row['path'])==row['closed_stat'],row['path']
    dest=R/'final_evidence';dest.mkdir(exist_ok=False);copy=dest/'metadata';copy.mkdir();rows=[]
    for p in sorted(H.rglob('*')):
        if not p.is_file():continue
        assert p.name!='SEALED.json'
        relative=str(p.relative_to(H));digest=sha(p);q=copy/relative;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);assert sha(q)==digest
        rows.append(dict(path=relative,bytes=p.stat().st_size,sha256=digest))
    manifest=dict(status='PASS_COMPLETE_CLOSED_METADATA_SNAPSHOT',source_root=str(H),files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),excludes_only_final_sealed_receipt=True)
    write(dest/'metadata_manifest.json',manifest);archive=dest/'metadata.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in rows:tar.add(copy/row['path'],arcname='metadata/'+row['path'],recursive=False)
        tar.add(dest/'metadata_manifest.json',arcname='metadata_manifest.json',recursive=False)
    digest=sha(archive);assert sha(archive)==digest
    expected={'metadata/'+r['path']:r for r in rows};expected['metadata_manifest.json']=dict(bytes=(dest/'metadata_manifest.json').stat().st_size,sha256=sha(dest/'metadata_manifest.json'))
    read=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            assert member.isfile() and member.name in expected
            data=tar.extractfile(member).read();row=expected[member.name]
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];read.append(member.name)
    assert len(read)==len(expected) and set(read)==set(expected) and sha(archive)==digest
    for row in rows:assert sha(H/row['path'])==row['sha256'] and sha(copy/row['path'])==row['sha256']
    for row in raw['files']:assert stat(R/row['path'])==row['closed_stat']
    receipt=dict(status='PASS_FULL_ARCHIVE_MEMBER_READBACK_AND_TWO_ARCHIVE_HASHES',metadata_files=len(rows),metadata_bytes=manifest['bytes'],archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=digest,members_read=len(read),raw_scientific_files=raw['count'],raw_scientific_bytes=raw['bytes'],raw_final_full_hash_actual_exit=0,raw_stat_stability_after_archive=True,metadata_manifest_sha256=sha(dest/'metadata_manifest.json'),source_copy_closed=True)
    write(dest/'ARCHIVE_RECEIPT.json',receipt)
    seal=dict(status='SEALED_COMPLETE_FINITE_SIMULATION_ACCEPTANCE',candidate_decision=closed['candidate_decision'],sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_root=str(H),raw_root=str(R),public_url=closed['public_url'],commit=closed['commit'],asset_commit=closed['asset_commit'],metadata_manifest_path=str(dest/'metadata_manifest.json'),metadata_manifest_sha256=sha(dest/'metadata_manifest.json'),archive_receipt_path=str(dest/'ARCHIVE_RECEIPT.json'),archive_receipt_sha256=sha(dest/'ARCHIVE_RECEIPT.json'),archive_sha256=digest,raw_readback_sha256=sha(H/'RAW_CLOSED_READBACK.json'),immutable_after_seal=True,distinct_paired_cases=128,complete_method_windows=512,actual_original_PNG=4494,physical_safety_certified=False,queued_future_status='UNKNOWN',excludes_sealed_receipt_from_metadata_archive=True)
    write(H/'SEALED.json',seal);shutil.copyfile(H/'SEALED.json',R/'SEALED.json')
    # No H/R writes after this line. The caller logs only outside these roots.
    print('SEALED',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
