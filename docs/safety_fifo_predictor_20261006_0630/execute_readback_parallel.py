"""Scheduling-only CPU readback: frozen functions, fresh all12 input reads.

Original main aggregation runs serially in registered order. Its inspect calls
are routed to independent fresh futures executing the exact compiled original
inspect function. No persisted results cache, numerical rule or policy change.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,sys,time,types
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def frozen_environment():
    reg=json.loads((HERE/'ANALYSIS_PARALLEL_REGISTRATION.json').read_text())
    assert all(sha(p)==d for p,d in reg['sources'].items())
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'plans').glob('holdout_*_plan.json'))];assert len(plans)==3
    pinned={p:d for plan in plans for p,d in plan['research_source_sha256'].items()}
    coverage=HERE.parent/'safety_random_space_20261004/coverage_metrics.py'
    blob=subprocess.check_output(['git','show','e8a9db408c1a6d61ff1314477ef92b56c4e73f70:docs/safety_random_space_20261004/coverage_metrics.py'],cwd='/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
    assert coverage.read_bytes()==blob;pinned[str(coverage)]=hashlib.sha256(blob).hexdigest()
    pinned.update(reg['sources']);assert all(sha(p)==d for p,d in pinned.items())
    for plan in plans:
        c=json.loads((Path(plan['output_root'])/'campaign.json').read_text());assert c['status']=='complete' and len(c['jobs'])==4 and all(j['status']=='complete' and j['exit_code']==0 for j in c['jobs'])
    return plans,pinned,coverage

def await_memory():
    while True:
        lines=Path('/proc/meminfo').read_text().splitlines()
        available=int(next(l.split()[1] for l in lines if l.startswith('MemAvailable:')))*1024
        if available>=25*1024**3:return
        print('CPU_READBACK_RESOURCE_WAIT',available,flush=True);time.sleep(10)

def compile_module(name,path,expected):
    source=Path(path).read_bytes();assert hashlib.sha256(source).hexdigest()==expected[str(path)]
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(source,str(path),'exec'),m.__dict__);return m

def analyze_parallel():
    plans,pinned,coverage=frozen_environment()
    compile_module('dense_audit',HERE/'dense_audit.py',pinned);compile_module('coverage_metrics',coverage,pinned)
    mod=compile_module('frozen_analyze_cpu_parallel',HERE/'analyze.py',pinned);exact_inspect=mod.inspect
    args=[(Path(plan['output_root'])/j['id'],j,plan) for plan in plans for j in plan['jobs']]
    timings=[];started=time.perf_counter()
    def inspect(arg):
        await_memory();t=time.perf_counter();answer=exact_inspect(*arg,dense=True)
        timings.append(dict(id=arg[1]['id'],seconds=time.perf_counter()-t));print('FROZEN_CELL_CPU_CLOSED',arg[1]['id'],answer[0]['violations'],answer[0]['deep'],flush=True);return answer
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={str(arg[0]):pool.submit(inspect,arg) for arg in args}
        def routed(path,job,plan,dense=True):
            assert dense and len(futures)==12
            original=next(a for a in args if a[0]==path)
            assert job==original[1] and plan==original[2]
            return futures[str(path)].result()
        mod.inspect=routed
        try:mod.main('holdout')
        finally:mod.inspect=exact_inspect
    assert all(sha(p)==d for p,d in pinned.items())
    result=HERE/'holdout_results.json';r=json.loads(result.read_text());assert r['analysis_source_sha256']==pinned[str(HERE/'analyze.py')]
    assert r['registered_method_windows']==r['completed_method_windows']==768 and r['invalid_method_windows']==0
    write(HERE/'offline_analysis_execution.json',dict(status='PASS_EXACT_REGISTERED_SOURCE_EXECUTION',
        executed_source_sha256={str(HERE/'analyze.py'):pinned[str(HERE/'analyze.py')],str(HERE/'dense_audit.py'):pinned[str(HERE/'dense_audit.py')],str(coverage):pinned[str(coverage)]},
        wrapper_sha256=sha(Path(__file__)),result_sha256=sha(result),cache_used=False,persisted_result_cache_used=False,
        execution='exact frozen inspect functions on4 independent fresh CPU futures; original frozen main aggregates in original serial order; only scheduling routed',
        outcome_selection=False,policy_tuning=False,source_bindings_before_after=True,total_seconds=time.perf_counter()-started,condition_timings=timings,
        supersession_sha256=sha(HERE/'ANALYSIS_CPU_SUPERSESSION.json'),
        coverage_dependency_provenance=dict(commit='e8a9db408c1a6d61ff1314477ef92b56c4e73f70',path='docs/safety_random_space_20261004/coverage_metrics.py',sha256=pinned[str(coverage)],bound_in_holdout_plan=False)))
    print('PARALLEL_FROZEN_ANALYSIS_PASS',flush=True)

def h6_parallel():
    plans,pinned,coverage=frozen_environment();path=HERE/'audit_prediction_h6.py'
    mod=compile_module('frozen_h6_cpu_parallel',path,pinned)
    args=[(Path(p['output_root'])/j['id'],j) for p in plans for j in p['jobs']]
    started=time.perf_counter()
    def inspect(arg):
        await_memory();r=mod.inspect(arg[0]);r.update(mode=arg[1]['env']['SAFEDUO_JOINT_MODE'],id=arg[1]['id'])
        print('H6_FROZEN_CPU_CLOSED',arg[1]['id'],r['geometry']['pd']['optimistic_raw_negative_row_endpoints'],flush=True);return r
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect,args))
    assert all(sha(p)==d for p,d in pinned.items())
    write(HERE/'H6_PREDICTION_AUDIT.json',dict(status='COMPLETE_OWN_TRAJECTORY_AUDIT',rows=rows,physical_safety_certified=False))
    write(HERE/'H6_PARALLEL_EXECUTION.json',dict(status='PASS_FROZEN_H6_SCHEDULING_ONLY',source_sha256=pinned[str(path)],wrapper_sha256=sha(Path(__file__)),
         row_order=[j['id'] for _,j in args],workers=4,all12_original_inspect_functions_used=True,total_seconds=time.perf_counter()-started,source_bindings_before_after=True))

def driver():
    plans,pinned,coverage=frozen_environment();tasks=[('run_analysis.py','analyze'),('audit_prediction_h6.py','h6'),('failure_audit.py',None),('audit_commands.py',None)]
    def run(task):
        name,stage=task;argv=['python3',str(HERE/'execute_readback_parallel.py'),stage] if stage else ['python3',str(HERE/name)]
        log=HERE/(name[:-3]+'_parallel.log')
        with log.open('x') as f:rc=subprocess.run(argv,cwd=HERE,stdout=f,stderr=subprocess.STDOUT).returncode
        print('CLOSED_PARALLEL_READBACK',name,rc,flush=True);return dict(name=name,returncode=rc,source_sha256=sha(HERE/name),actual_argv=argv,log=str(log))
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,tasks))
    assert all(sha(p)==d for p,d in pinned.items())
    write(HERE/'ANALYSIS_FOLLOW_EXECUTION.json',dict(status='PASS' if all(r['returncode']==0 for r in results) else 'FAIL',jobs=results,sources=pinned,
        scheduling='4 analysis cells +4 H6 cells concurrently; exact original frozen calculation functions; failureLP andcommands original CLI',supersession_sha256=sha(HERE/'ANALYSIS_CPU_SUPERSESSION.json'),utc=datetime.now(timezone.utc).isoformat()))
    assert all(r['returncode']==0 for r in results)
if __name__=='__main__':{'driver':driver,'analyze':analyze_parallel,'h6':h6_parallel}[sys.argv[1]]()
