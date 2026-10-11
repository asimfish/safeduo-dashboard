"""One bounded CPU projection; immutable owned guardian and genuine external wait4."""
import argparse,json,os
from pathlib import Path
import common_v1 as c
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);p.add_argument('--candidate-out',required=True);p.add_argument('--run',action='store_true');a=p.parse_args();before=c.sources();root=Path(a.out)
    c.need(root.resolve().is_relative_to(c.O) and not root.exists(),'exclusive delta CPU root')
    if not a.run:
        obs=c.module(c.OBSERVER,c.OBSERVER_SHA,'_r16_delta_observer');argv=[c.PY,'-B',str(c.D/'run_text_delta_v1.py'),*os.sys.argv[1:],'--run'];w=obs.observe(argv,root.parent/(root.name+'_outer'))
        print(json.dumps(dict(outer_exit=w['actual_wait_exit'],outer_wait=c.ref(root.parent/(root.name+'_outer')/'ACTUAL_WAIT.json'))),flush=True)
        return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
    core=c.module(c.LOADER,c.LOADER_SHA,'_r16_delta_loader').load_reviewed_core(c.CORE_SHA)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in c.THREADS})
    root.mkdir(parents=True,exist_ok=False)
    w=core.bounded_child([c.PY,'-B',str(c.D/'text_delta_v1.py'),'--out',a.candidate_out],env,root/'worker',backend='CPU_ONLY_REPORT16_BOUNDED_BILINGUAL_PEAK_TEXT_DELTA',max_wall=60,cpu_limit=True)
    c.need(c.sources()==before,'frozen release3 source unchanged')
    c.write(root/'CPU_DELIVERY_V1.json',dict(status='CLOSED_BOUNDED_TEXT_DELTA_EXECUTION',worker_wait=c.ref(root/'worker/ACTUAL_WAIT.json'),actual_worker_exit=w['actual_wait_exit'],source_before_after=before,source_unchanged=True,max_wall_seconds=60,CPU_AS_bytes=1073741824,threads=1,new_PNG_RAW_SDK_GPU_Git_HTTP_browser_actions=0))
    return 0 if w['actual_wait_exit']==0 and not w['error'] and not w['resource_abort'] and not w['signals'] else 2
if __name__=='__main__':raise SystemExit(main())
