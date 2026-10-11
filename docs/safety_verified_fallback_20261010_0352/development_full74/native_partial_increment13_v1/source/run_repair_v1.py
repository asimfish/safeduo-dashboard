"""Original sole CPU guardian and namespace true wait for narrow repair only."""
import argparse,json,os
from pathlib import Path
import report_v1 as r
def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--mode',choices=['repair','mobile'],required=True)
    for k in ('out','graph','graph-sha256'):ap.add_argument('--'+k,required=True)
    for k in ('candidate-out','plan','plan-sha256','browser-out'):ap.add_argument('--'+k)
    ap.add_argument('--run',action='store_true');a=ap.parse_args();b=r.binding();root=Path(a.out);g=b.read(r.rr(a.graph,a.graph_sha256));before={k:b.ref(v['path']) for k,v in g['own_source_files'].items()};b.need(before==g['own_source_files'],'bound repair source before launch')
    b.need(root.is_absolute() and root.resolve().is_relative_to(r.O) and not root.exists(),'exclusive own CPU output')
    if not a.run:
        observer=r.authenticated_module(r.OBSERVER,r.OBSERVER_SHA,'_report13_repair_observer');root.parent.mkdir(parents=True,exist_ok=True)
        w=observer.observe([r.PY,'-B',str(r.D/'run_repair_v1.py'),*os.sys.argv[1:],'--run'],root.parent/(root.name+'_outer'))
        print(json.dumps(dict(actual_outer_exit=w['actual_wait_exit'],wait=b.ref(root.parent/(root.name+'_outer/ACTUAL_WAIT.json')))),flush=True);return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    if a.mode=='repair':
        b.need(a.candidate_out,'new candidate output required');argv=[r.PY,'-B',str(r.D/'repair_v1.py'),'--out',a.candidate_out];wall=180;limited=True
    else:
        b.need(a.plan and a.plan_sha256 and a.browser_out,'actual repaired plan required');argv=[r.BPY,'-B',str(r.D/'qa_repair_v1.py'),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--out',a.browser_out];wall=180;limited=False
    root.mkdir(parents=True,exist_ok=False);core=r.authenticated_module(r.LOADER,r.LOADER_SHA,'_report13_repair_loader').load_reviewed_core(r.CORE_SHA);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in b.THREADS},OMP_THREAD_LIMIT='1')
    w=core.bounded_child(argv,env,root/'worker',backend='CPU_ONLY_REPORT13_NARROW_'+a.mode.upper(),max_wall=wall,cpu_limit=limited);after={k:b.ref(v['path']) for k,v in g['own_source_files'].items()};b.need(after==before,'source unchanged after actual repair worker')
    b.write(root/'CPU_DELIVERY_V1.json',dict(mode=a.mode,actual_worker_wait=b.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_graph=r.rr(a.graph,a.graph_sha256),source_before_after=before,source_unchanged=True,wall_seconds=wall,CPU_AS_bytes=1073741824 if limited else 'REUSED_SOFTWARE_BROWSER_VIRTUAL_ADDRESS_RESERVATION',native_GPU_SDK_Git_W_actions=0))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
if __name__=='__main__':raise SystemExit(main())
