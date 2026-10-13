"""Keep every original raw file byte while reducing API upload payload size."""
from pathlib import Path
import hashlib,html,json,re,shutil,zipfile,zlib

K=Path(__file__).resolve().parent.parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
P=W/'docs/safety_multitask_20261013'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=P/'data/native119_raw_manifest.json';original=json.loads(manifest.read_text());rows=original['files']
    old_names={r['archive'] for r in rows};readers={name:zipfile.ZipFile(P/'data'/name) for name in old_names}
    index=-1;archive=None;name=None;new_rows=[];new_names=[];predicted=0
    try:
        for row in rows:
            content=readers[row['archive']].read(row['path']);assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
            compressed=len(zlib.compress(content,1))+len(row['path'].encode())*2+512
            assert compressed<8*1024**2,'one source file needs explicit binary chunk custody'
            if archive is None or predicted+compressed>7*1024**2:
                if archive is not None:archive.close()
                index+=1;name=f'native119_raw_small{index:02d}.zip';new_names.append(name)
                archive=zipfile.ZipFile(P/'data'/name,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1);predicted=0
            archive.writestr(row['path'],content);predicted+=compressed;new_rows.append(dict(row,archive=name))
        if archive is not None:archive.close();archive=None
    finally:
        if archive is not None:archive.close()
        for reader in readers.values():reader.close()
    for name in new_names:
        assert (P/'data'/name).stat().st_size<8*1024**2
        with zipfile.ZipFile(P/'data'/name) as z:
            for row in [r for r in new_rows if r['archive']==name]:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    assert {(r['path'],r['bytes'],r['sha256']) for r in rows}=={(r['path'],r['bytes'],r['sha256']) for r in new_rows}
    # Original large archives remain recoverable from immutable local commit
    # d075886c and its named backup branch; only owned draft packaging changes.
    for name in old_names:(P/'data'/name).unlink()
    manifest.write_text(json.dumps(dict(original,files=new_rows),indent=2)+'\n')
    code=P/'index.html';page=code.read_text();links=''.join(f'<li><a href="data/{html.escape(f.name)}">{html.escape(f.name)}</a></li>' for f in sorted((P/'data').iterdir()) if f.suffix in ['.json','.zip','.npz','.md'])
    page,n=re.subn(r'(<summary>下载原始记录、独立判定、场景与协议</summary><ul>).*?(</ul></details>)',lambda m:m[1]+links+m[2],page,count=1,flags=re.S);assert n==1;code.write_text(page)
    shutil.copyfile(Path(__file__),P/'sources/repack_report_small_blobs_v1.py')
    files=[{'path':f.relative_to(P).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(P.rglob('*')) if f.is_file() and f.name!='PUBLIC_MANIFEST_V1.json']
    (P/'PUBLIC_MANIFEST_V1.json').write_text(json.dumps({'report':'safety_multitask_20261013','files':files,'fullSystem0_accepted':False},indent=2)+'\n')
    result={'status':'SMALL_BLOB_PACKAGING_ALL_ORIGINAL_FILE_BYTES_PRESERVED','original_archive_count':len(old_names),'new_archive_count':len(new_names),
        'original_source_files_verified':len(rows),'maximum_new_archive_bytes':max((P/'data'/name).stat().st_size for name in new_names),
        'large_local_commit_retained':'d075886c2d4a6b4b5b46099af3be5759cbb2e1df','report_files':len(files)+1,
        'numerical_result_and_native_video_changes':0,'fullSystem0_accepted':False}
    (K/'SMALL_BLOB_REPACK_CUSTODY_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
