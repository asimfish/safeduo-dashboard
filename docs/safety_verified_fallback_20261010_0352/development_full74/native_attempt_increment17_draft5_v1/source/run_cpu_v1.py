"""Finite metadata projection only; original guardian and truefork/wait4 unchanged."""
import argparse,json,os
from pathlib import Path
import common_v1 as c
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);p.add_argument('--candidate-out',required=True);p.add_argument('--run',action='store_true');a=p.parse_args();g=c.sources();root=Path(a.out)
    c.need(root.resolve().is_relative_to(c.O) and not root.exists(),'exclusive own CPU operation')
    c.need(os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in c.THREADS),'B CPUonly threads1')
    if not a.run:
        obs=c.module(c.OBSERVER,c.OBSERVER_SHA,'_r17_original_observer');w=obs.observe([c.PY,'-B',str(c.D/'run_cpu_v1.py'),*os.sys.argv[1:],'--run'],root.parent/(root.name+'_outer'))
        print(json.dumps({'actual_outer_exit':w['actual_wait_exit'],'outer_wait':c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json')}),flush=True)
        return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    core=c.module(c.LOADER,c.LOADER_SHA,'_r17_original_loader').load_reviewed_core(c.CORE_SHA);root.mkdir(parents=True,exist_ok=False)
    w=core.bounded_child([c.PY,'-B',str(c.D/'build_v1.py'),'--out',a.candidate_out],dict(os.environ),root/'worker',backend='CPU_ONLY_REPORT17_SOURCE_DRAFT',max_wall=60,cpu_limit=True)
    c.need(c.sources()==g,'own SOURCE unchanged')
    c.write(root/'CPU_DELIVERY_V1.json',dict(status='CLOSED_SOURCE_ONLY_METADATA_PROJECTION_STAGE',actual_worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),source_unchanged=True,max_wall_seconds=60,CPU_AS_bytes=1073741824,threads=1,native_GPU_SDK_Git_network_calls=0,PNG_decodes=0,browser_runs=0,publish_performed=False))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
if __name__=='__main__':raise SystemExit(main())
