"""One actual final hash read of every closed scientific raw file, with inode/stat stability."""
from pathlib import Path
import hashlib,json,datetime
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent
R=Path('/mnt/nas/data/lyf/double_hand')/H.name
def sig(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def read(p):
    before=sig(p);digest=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):digest.update(b)
    assert sig(p)==before,str(p)
    return dict(path=str(p.relative_to(R)),bytes=before[2],sha256=digest.hexdigest(),closed_stat=before)
def main():
    assert not (R/'SEALED.json').exists() and not (R/'final_evidence').exists()
    files=sorted(p for p in R.rglob('*') if p.is_file());print('CLOSED_RAW_BEGIN',len(files),sum(p.stat().st_size for p in files),flush=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,row in enumerate(pool.map(read,files)):
            rows.append(row)
            if (i+1)%500==0:print('CLOSED_RAW_HASHED',i+1,flush=True)
    assert files==sorted(p for p in R.rglob('*') if p.is_file())
    with (H/'RAW_HASH_ROWS_V2.json').open('x') as f:json.dump(rows,f,indent=2)
    differences=[dict(path=row['path'],before=row['closed_stat'],after=sig(R/row['path'])) for row in rows if sig(R/row['path'])!=row['closed_stat']]
    with (H/'RAW_STAT_DIAGNOSTIC_V2.json').open('x') as f:json.dump(differences,f,indent=2)
    assert not differences, differences[:10]
    value=dict(status='PASS_ACTUAL_ALL_CLOSED_RAW_FILE_HASH_READBACK',count=len(rows),bytes=sum(r['bytes'] for r in rows),files=rows,all_scientific_producers_and_auditors_already_closed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'RAW_CLOSED_READBACK.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print('CLOSED_RAW_COMPLETE',value['count'],value['bytes'],flush=True)
if __name__=='__main__':main()
