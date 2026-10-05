"""Use corrected pre-physics launch plans, preserving original registration."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import execute
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    files=sorted((HERE/'registered_launch').glob('block*.json'));assert len(files)==3
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(execute.worker,files[::2]),pool.submit(execute.worker,files[1::2])]
        results=[f.result() for f in futures]
    execute.write(HERE/'campaign_completion.json',dict(status='complete' if all(r['status']=='complete' for rs in results for r in rs) else 'contains_invalid',blocks=[r for rs in results for r in rs],finished_utc=execute.now()))
