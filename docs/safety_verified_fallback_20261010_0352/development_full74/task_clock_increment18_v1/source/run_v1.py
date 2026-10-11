"""Original finite CPU/Chromium guardian and truefork/wait4, no native or Git."""
import argparse,json,os
from pathlib import Path
import common_v1 as c
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--mode',choices=('build','browser'),required=True)
    for k in ('out','products'):p.add_argument('--'+k,required=True)
    p.add_argument('--candidate');p.add_argument('--candidate-sha256');p.add_argument('--run',action='store_true');a=p.parse_args();before=c.sources();root=Path(a.out)
    c.need(not root.exists() and root.resolve().is_relative_to(c.O),'exclusive finite stage')
    c.need(os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in c.THREADS),'B threads1 noCUDA')
    c.inherited()
    if a.mode=='browser':c.authenticate({'path':a.candidate,'sha256':a.candidate_sha256})
    if not a.run:
        obs=c.module(c.OBSERVER,c.OBSERVER_SHA,'_r18_original_truewait');w=obs.observe([c.PY,'-B',str(c.D/'run_v1.py'),*os.sys.argv[1:],'--run'],root.parent/(root.name+'_outer'))
        print(json.dumps({'actual_outer_exit':w['actual_wait_exit'],'outer_wait':c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json')}),flush=True);return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    core=c.module(c.LOADER,c.LOADER_SHA,'_r18_original_core').load_reviewed_core(c.CORE_SHA);root.mkdir(parents=True,exist_ok=False)
    limited=a.mode=='build';argv=[c.PY if limited else c.BPY,'-B',str(c.D/('build_v1.py' if limited else 'qa_local_v1.py')),'--out',a.products]
    if not limited:argv+=['--candidate',a.candidate,'--candidate-sha256',a.candidate_sha256]
    w=core.bounded_child(argv,dict(os.environ),root/'worker',backend='CPU_ONLY_REPORT18_LOCAL_'+a.mode,max_wall=60 if limited else 360,cpu_limit=limited)
    c.need(c.sources()==before,'own/reused source unchanged');c.write(root/'ACTUAL_STAGE_DELIVERY_V1.json',dict(actual_worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_before=before,source_after=c.sources(),max_wall_seconds=60 if limited else 360,CPU_AS_bytes=1073741824 if limited else 'INFINITY_SOFTWARE_CHROMIUM',native_GPU_SDK_Git_calls=0,publish_performed=False))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
if __name__=='__main__':raise SystemExit(main())
