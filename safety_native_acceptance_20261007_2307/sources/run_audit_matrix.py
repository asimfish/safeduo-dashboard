from pathlib import Path
import json, sys, subprocess, concurrent.futures, datetime
H = Path(__file__).resolve().parent
def run(argv):
    child = subprocess.Popen(argv, cwd=H, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = child.communicate()
    result = dict(argv=argv,pid=child.pid,actual_exit=child.wait(),output=stdout.decode(errors='replace'))
    print(json.dumps(result),flush=True)
    return result
if __name__ == '__main__':
    jobs=json.loads(Path(sys.argv[1]).read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,jobs))
    with (H/sys.argv[2]).open('x') as f:
        json.dump(dict(jobs=results,status='complete' if all(r['actual_exit']==0 for r in results) else 'failed',closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2)
    sys.exit(0 if all(r['actual_exit']==0 for r in results) else 1)
