"""Frozen full-cell readback with original aggregation order and two workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,sys,time

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import analyze


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'plans').glob('holdout_*_plan.json'))]
    frozen={k:v for p in plans for k,v in p['research_source_sha256'].items()}
    assert all(sha(p)==s for p,s in frozen.items())
    args=[(Path(p['output_root'])/j['id'],j,p) for p in plans for j in p['jobs']]
    assert len(args)==9
    inspect=analyze.inspect
    began=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={str(a[0]):pool.submit(inspect,*a,dense=True) for a in args}
        def routed(path,job,plan,dense=True):
            original=next(a for a in args if a[0]==path)
            assert dense and original[1]==job and original[2]==plan
            return futures[str(path)].result()
        analyze.inspect=routed
        try:analyze.main('holdout')
        finally:analyze.inspect=inspect
    assert all(sha(p)==s for p,s in frozen.items())
    result=HERE/'holdout_results.json'
    with (HERE/'ANALYSIS_EXECUTION.json').open('x') as f:
        json.dump(dict(status='PASS_FROZEN_FRESH_FULL_READBACK',sources=frozen,source_after_verified=True,
            result_sha256=sha(result),workers=2,seconds=time.perf_counter()-began,persisted_cache=False),f,indent=2)
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
