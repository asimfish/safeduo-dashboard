"""Finite owned SOURCE/build/browser supervisor; no native, Git, or publication."""
import argparse
import json
import os
from pathlib import Path
import report_v1 as r

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode',choices=('source-tests','build','local-qa'),required=True)
    for k in ('out','graph','graph-sha256'):p.add_argument('--'+k,required=True)
    for k in ('packet','packet-sha256','candidate-out','plan','plan-sha256','browser-out'):p.add_argument('--'+k)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    b=r.binding();root=Path(a.out)
    b.need(root.is_absolute() and root.resolve().is_relative_to(r.O) and not root.exists(),'exclusive own CPU root')
    graph=b.read(r.rr(a.graph,a.graph_sha256));before={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()}
    b.need(before==graph['own_source_files'],'exact report12 source before launch')
    if not a.run:
        obs=r.authenticated_module(r.OBSERVER,r.OBSERVER_SHA,'_report12_observer')
        argv=[r.PY,'-B',str(r.D/'run_cpu_v1.py'),*os.sys.argv[1:],'--run']
        w=obs.observe(argv,root.parent/(root.name+'_outer'))
        print(json.dumps(dict(actual_outer_exit=w['actual_wait_exit'],wait=b.ref(root.parent/(root.name+'_outer/ACTUAL_WAIT.json')))),flush=True)
        return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    if a.mode=='source-tests':
        argv=[r.PY,'-B',str(r.D/'test_source_v1.py'),'--out',str(root/'source_tests')];wall=60;limited=True
    elif a.mode=='build':
        b.need(a.packet and a.packet_sha256 and a.candidate_out,'closed V8 inputs required')
        r.terminal_inputs(b,r.rr(a.packet,a.packet_sha256))
        argv=[r.PY,'-B',str(r.D/'report_v1.py'),'--packet',a.packet,'--packet-sha256',a.packet_sha256,'--out',a.candidate_out];wall=180;limited=True
    else:
        b.need(a.plan and a.plan_sha256 and a.browser_out,'real built candidate required')
        plan=b.read(r.rr(a.plan,a.plan_sha256));facts=b.read(plan['source_facts']);r.terminal_inputs(b,facts['packet_ref'])
        b.need(plan['schema']=='astra.report.increment12.overlay_plan.v1' and plan['old8fig_and15RGB_SHA_preserved'] is True,'actual report12 candidate plan')
        argv=[r.BPY,'-B',str(r.D/'qa_local_v1.py'),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--out',a.browser_out];wall=360;limited=False
    core=r.authenticated_module(r.LOADER,r.LOADER_SHA,'_report12_owned_loader').load_reviewed_core(r.CORE_SHA)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in b.THREADS},OMP_THREAD_LIMIT='1')
    root.mkdir(parents=True,exist_ok=False)
    w=core.bounded_child(argv,env,root/'worker',backend='CPU_ONLY_REPORT12_'+a.mode,max_wall=wall,cpu_limit=limited)
    after={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()};b.need(after==before,'source unchanged after real worker')
    b.write(root/'CPU_DELIVERY_V1.json',dict(mode=a.mode,status='ACTUAL_CPU_WORKER_CLOSED',actual_worker_wait=b.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_graph=r.rr(a.graph,a.graph_sha256),source_before_after=before,source_unchanged=True,max_wall_seconds=wall,CPU_AS_bytes=1073741824 if limited else 'REUSED_SOFTWARE_BROWSER_VIRTUAL_ADDRESS_RESERVATION',new_native_GPU_Git_W_actions=0,publication_performed=False))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2

if __name__=='__main__':raise SystemExit(main())
