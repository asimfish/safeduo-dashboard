"""Finite owned CPU build/local-QA supervisor. No native, Git or publication.

The build worker has AS1GiB/threads1. Future software Chromium uses the closed
REPORT10 large-virtual-address browser reservation, bounded360s and own core.
"""
import argparse
import json
import os
from pathlib import Path
import report_v1 as r

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--mode',choices=('build','local-qa','source-tests'),required=True);p.add_argument('--out',required=True);p.add_argument('--packet');p.add_argument('--packet-sha256');p.add_argument('--candidate-out');p.add_argument('--plan');p.add_argument('--plan-sha256');p.add_argument('--browser-out');a=p.parse_args()
    b=r.binding();root=Path(a.out);b.need(root.is_absolute() and root.resolve().is_relative_to(r.O) and not root.exists(),'exclusive owned CPU execution output')
    graph=b.read(b.ref(r.D/'SOURCE_GRAPH_V1.json'))
    for ref in graph['own_source_files'].values():b.buffer(ref,1024**2)
    before={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()}
    if a.mode=='build':
        b.need(a.packet and a.packet_sha256 and a.candidate_out,'explicit future terminal input');r.terminal_inputs(b,r.rr(a.packet,a.packet_sha256))
        argv=[r.PY,'-B',str(r.D/'report_v1.py'),'--packet',a.packet,'--packet-sha256',a.packet_sha256,'--out',a.candidate_out];wall=180;limited=True
    elif a.mode=='local-qa':
        b.need(a.plan and a.plan_sha256 and a.browser_out,'actual built candidate only');plan=b.read(r.rr(a.plan,a.plan_sha256));facts=b.read(plan['source_facts']);r.terminal_inputs(b,facts['packet_ref'])
        b.need(plan['schema']=='astra.report.increment11.overlay_plan.v1' and plan['old8fig_and15RGB_SHA_preserved'] is True,'actual overlay candidate')
        argv=[r.BPY,'-B',str(r.D/'qa_local_v1.py'),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--out',a.browser_out];wall=360;limited=False
    else:
        argv=[r.PY,'-B',str(r.D/'test_source_v1.py'),'--out',str(root/'source_tests')];wall=60;limited=True
    core=r.authenticated_module(r.LOADER,r.LOADER_SHA,'_report11_reviewed_loader').load_reviewed_core(r.CORE_SHA)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in b.THREADS},OMP_THREAD_LIMIT='1')
    root.mkdir(parents=True,exist_ok=False)
    w=core.bounded_child(argv,env,root/'worker',backend=dict(max_wall_seconds=wall),max_wall=wall,cpu_limit=limited)
    b.need(before=={k:b.ref(q['path']) for k,q in graph['own_source_files'].items()},'source unchanged')
    b.write(root/'CPU_DELIVERY_V1.json',dict(mode=a.mode,status='CLOSED_SOURCE_PREPARATION_ONLY' if a.mode=='source-tests' and w['actual_wait_exit']==0 else 'ACTUAL_CPU_WORKER_CLOSED',actual_worker_wait=b.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_before_after=before,source_unchanged=True,max_wall_seconds=wall,CPU_AS_bytes=1073741824 if limited else 'INHERITED_REVIEWED_SOFTWARE_BROWSER_VIRTUAL_ADDRESS_RESERVATION',new_native_GPU_Git_actions=0,publication_performed=False))
    raise SystemExit(0 if w['actual_wait_exit']==0 and not w['resource_abort'] and not w['error'] and not w['signals'] else 2)

if __name__=='__main__':main()
