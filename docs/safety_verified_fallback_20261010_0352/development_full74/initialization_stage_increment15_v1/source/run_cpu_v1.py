"""One immutable owned guardian; original truefork/wait4 external observer."""
import argparse,json,os
from pathlib import Path
import common_v1 as c
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode',choices=('build','local-qa','seal','handoff'),required=True);p.add_argument('--out',required=True)
    for k in ['candidate-out','plan','plan-sha256','browser-out','tools','tools-sha256','seal-tool','seal-tool-sha256']:p.add_argument('--'+k)
    p.add_argument('--run',action='store_true');a=p.parse_args();before=c.sources();root=Path(a.out)
    c.need(root.resolve().is_relative_to(c.O) and not root.exists(),'exclusive owned CPU stage')
    if not a.run:
        obs=c.module(c.OBSERVER,c.OBSERVER_SHA,'_r15_observer')
        argv=[c.PY,'-B',str(c.D/'run_cpu_v1.py'),*os.sys.argv[1:],'--run'];w=obs.observe(argv,root.parent/(root.name+'_outer'))
        print(json.dumps({'outer_exit':w['actual_wait_exit'],'outer_wait':c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json')}),flush=True)
        return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    if a.mode=='build':
        c.need(a.candidate_out,'new candidate output');argv=[c.PY,'-B',str(c.D/'build_v1.py'),'--out',a.candidate_out];wall=180;limited=True
    elif a.mode=='local-qa':
        c.need(a.plan and a.plan_sha256 and a.browser_out,'actual built plan');argv=[c.BPY,'-B',str(c.D/'qa_local_v1.py'),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--out',a.browser_out];wall=360;limited=False
    elif a.mode=='seal':
        c.need(a.plan and a.plan_sha256 and a.tools and a.tools_sha256,'actual prior stage/tool refs');argv=[c.PY,'-B',str(c.D/'seal_v1.py'),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--tools',a.tools,'--tools-sha256',a.tools_sha256];wall=60;limited=True
    else:
        c.need(a.seal_tool and a.seal_tool_sha256,'actual seal worker/outer/tool closure');argv=[c.PY,'-B',str(c.D/'finalize_v1.py'),'--mode','handoff','--seal-tool',a.seal_tool,'--seal-tool-sha256',a.seal_tool_sha256];wall=60;limited=True
    core=c.module(c.LOADER,c.LOADER_SHA,'_r15_loader').load_reviewed_core(c.CORE_SHA)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in c.THREADS},MPLCONFIGDIR=str(root/'mpl_cache'))
    root.mkdir(parents=True,exist_ok=False);w=core.bounded_child(argv,env,root/'worker',backend='CPU_ONLY_REPORT15_'+a.mode,max_wall=wall,cpu_limit=limited)
    c.need(c.sources()==before,'frozen report source unchanged')
    c.write(root/'CPU_DELIVERY_V1.json',dict(status='CLOSED_OWNED_CPU_STAGE',mode=a.mode,actual_worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_before_after=before,source_unchanged=True,max_wall_seconds=wall,CPU_AS_bytes=1073741824 if limited else 'REUSED_SOFTWARE_BROWSER_VIRTUAL_ADDRESS_RESERVATION',threads=1,CUDA_VISIBLE_DEVICES='',native_GPU_SDK_Git_W_actions=0))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
if __name__=='__main__':raise SystemExit(main())
