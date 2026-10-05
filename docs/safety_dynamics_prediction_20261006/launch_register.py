"""Bind completed registration log before any new outcomes; retain first plans."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
from execute import check
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    directory=HERE/'registered_launch';directory.mkdir(exist_ok=False)
    additions=['launch.py','analyze_registered.py','launch_register.py']
    for file in sorted((HERE/'registered').glob('block*.json')):
        plan=json.loads(file.read_text());changes=[]
        for path,h in plan['research_source_sha256'].items():
            actual=sha(path)
            if actual!=h:
                assert path==str(HERE/'registration.log')
                changes.append(dict(path=path,original=h,final=actual))
                plan['research_source_sha256'][path]=actual
        assert len(changes)==1
        plan['research_source_sha256'].update({str(HERE/name):sha(HERE/name) for name in additions})
        plan.update(supersedes_plan_sha256=sha(file),prephysics_correction=changes,
                    correction_reason='registration log was hashed while its stdout file was empty; no physical job has launched; commands/model/metrics/inputs unchanged',
                    registered_utc=datetime.now(timezone.utc).isoformat())
        assert not Path(plan['output_root']).exists()
        check(plan)
        with (directory/file.name).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('PREPHYSICS ALL SOURCE GATES PASS',flush=True)
if __name__=='__main__':main()
