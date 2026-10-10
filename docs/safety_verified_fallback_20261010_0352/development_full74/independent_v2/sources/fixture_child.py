"""Explicitly synthetic CPU child, incapable of claiming native provenance."""
import argparse
import os
import resource
import time
from evidence_io import cpu_limits,THREADS,require

if __name__=='__main__':
    cpu_limits();p=argparse.ArgumentParser();p.add_argument('mode',choices=('zero','seven','timeout','premature-zero'));a=p.parse_args()
    require(resource.getrlimit(resource.RLIMIT_AS)==(1024**3,1024**3),'1GiB hard/soft address space')
    require(all(os.environ[k]=='1' for k in THREADS),'single thread environment')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='','CPU fixture cannot see GPU')
    print('CPU_FIXTURE_ONLY; native_pass=false; limits=1GiB; threads=1; bytecode=false',flush=True)
    if a.mode=='premature-zero':os._exit(0)
    if a.mode=='timeout':time.sleep(4)
    raise SystemExit(7 if a.mode=='seven' else 0)
