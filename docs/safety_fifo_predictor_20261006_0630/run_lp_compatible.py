"""Execute unchanged first-failure math in compatible existing isolated runtime.

Only output/plan lookup HERE is pointed at a fresh mirror. Original __file__,
all source hashes, raw paths, float32 physical thresholds and LP options stay.
No global packages are installed, removed or downgraded.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,platform,shutil,sys,types
import numpy as np,scipy
HERE=Path(__file__).resolve().parent
OUT=HERE/'lp_backend_compatible'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    registration=json.loads((HERE/'LP_BACKEND_REGISTRATION.json').read_text());assert all(sha(p)==d for p,d in registration['sources'].items())
    assert sys.executable==registration['interpreter'] and np.__version__==registration['numpy'] and scipy.__version__==registration['scipy']
    OUT.mkdir(exist_ok=False);(OUT/'plans').mkdir()
    for p in sorted((HERE/'plans').glob('*_plan.json')):
        plan=json.loads(p.read_text());assert all(sha(Path(plan['cwd'])/f)==d for f,d in plan['source_sha256'].items());assert all(sha(f)==d for f,d in plan['research_source_sha256'].items())
        shutil.copy2(p,OUT/'plans'/p.name);assert sha(p)==sha(OUT/'plans'/p.name)
    path=HERE/'failure_audit.py';code=path.read_bytes();assert hashlib.sha256(code).hexdigest()==registration['sources'][str(path)]
    m=types.ModuleType('frozen_first_failure_compatible_runtime');m.__file__=str(path);exec(compile(code,str(path),'exec'),m.__dict__)
    m.HERE=OUT;m.main()
    assert all(sha(p)==d for p,d in registration['sources'].items())
    result=OUT/'first_failure_audit.json';r=json.loads(result.read_text());assert r['status']=='PASS_EXACT_FIRST_FAILURE_BINDING_AND_LP' and r['cases']==79 and r['source_sha256']==registration['sources'][str(path)]
    receipt=dict(status='PASS_FROZEN_FIRST_FAILURE_COMPATIBLE_BACKEND',source_sha256=r['source_sha256'],wrapper_sha256=sha(Path(__file__)),result_sha256=sha(result),
      result_path=str(result),python=platform.python_version(),interpreter=sys.executable,numpy=np.__version__,scipy=scipy.__version__,cases=79,
      lp_math_and_options_unchanged=True,geometry_native_float32_threshold_unchanged=True,global_package_mutation=False,
      output_root_only_redirected=True,registered_all12conditions_used=True,utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'LP_COMPATIBLE_EXECUTION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'],flush=True)
if __name__=='__main__':main()
