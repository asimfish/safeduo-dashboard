"""Compile exact registered bytes; actual raw CLI reads only, no result cache."""
from pathlib import Path
import json,hashlib,sys,types,subprocess
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'plans').glob('*_plan.json'))];assert len(plans)==3
    expected={}
    for p in plans:
        for name,sha in p['research_source_sha256'].items():
            if name in expected:assert expected[name]==sha
            expected[name]=sha
    artifacts={}
    coverage_path=HERE.parent/'safety_random_space_20261004/coverage_metrics.py'
    prior_blob=subprocess.check_output(['git','show','e8a9db408c1a6d61ff1314477ef92b56c4e73f70:docs/safety_random_space_20261004/coverage_metrics.py'],cwd='/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
    assert coverage_path.read_bytes()==prior_blob, 'coverage algorithm differs from publication preceding this study'
    expected[str(coverage_path)]=digest(prior_blob)
    for module,path in [('dense_audit',HERE/'dense_audit.py'),('coverage_metrics',HERE.parent/'safety_random_space_20261004/coverage_metrics.py')]:
        source=path.read_bytes();sha=digest(source);assert str(path) in expected and sha==expected[str(path)]
        m=types.ModuleType(module);m.__file__=str(path);sys.modules[module]=m
        exec(compile(source,str(path),'exec'),m.__dict__);artifacts[str(path)]=sha
    path=HERE/'analyze.py';source=path.read_bytes();sha=digest(source);assert sha==expected[str(path)]
    artifacts[str(path)]=sha
    exec(compile(source,str(path),'exec'),dict(__file__=str(path),__name__='__main__'))
    for p,sha in artifacts.items():assert digest(Path(p).read_bytes())==sha
    result=HERE/'holdout_results.json';r=json.loads(result.read_text());assert r['analysis_source_sha256']==artifacts[str(path)]
    receipt=dict(status='PASS_EXACT_REGISTERED_SOURCE_EXECUTION',executed_source_sha256=artifacts,result_sha256=digest(result.read_bytes()),
        wrapper_sha256=digest(Path(__file__).read_bytes()),cache_used=False,scope='verified bytes compiled before exec; registered helpers, fresh raw-file caller data, source and input readback after scoring')
    receipt['coverage_dependency_provenance']=dict(commit='e8a9db408c1a6d61ff1314477ef92b56c4e73f70',path='docs/safety_random_space_20261004/coverage_metrics.py',sha256=digest(prior_blob),
        bound_in_holdout_plan=False,scope='same exact helper bytes from publication preceding new study; additional dependency binding established before full scoring, after numerical launch, before full scoring')
    (HERE/'offline_analysis_execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],flush=True)
if __name__=='__main__':main()
