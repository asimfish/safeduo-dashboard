"""Immutable correction after diagnosed pre-physics seam failure."""
from pathlib import Path
import json,copy
from execute import sha,check,now
HERE=Path(__file__).resolve().parent
def main():
    directory=HERE/'registered_retry';directory.mkdir(exist_ok=False)
    completion=json.loads((HERE/'campaign_completion.json').read_text());assert completion['status']=='contains_invalid'
    additions=['observer_runner_v2.py','test_observer_seam.py','seam_regression.log','SEAM_FAILURE.md','execute_retry.py','score.py','retry_register.py','campaign_completion.json']
    for file in sorted((HERE/'registered_launch').glob('block*.json')):
        plan=copy.deepcopy(json.loads(file.read_text()));check(plan)
        for j in plan['jobs']:
            previous=Path(plan['output_root'])/j['id']
            p=json.loads((previous/'protocol.json').read_text());a=json.loads((previous/'guard_abort.json').read_text())
            assert p['completed_cells']==0 and p['error']=="AttributeError: 'Observer' object has no attribute 'full_J'"
            assert a['step']==0 and a['abort_stage']=='after target integration; before physics'
            assert not (previous/'cell_001.npz').exists()
        root=Path(plan['output_root']).parent/'retry1'/Path(plan['output_root']).name
        for j in plan['jobs']:
            j['argv'][1]=str(HERE/'observer_runner_v2.py');j['argv'][j['argv'].index('--out')+1]=str(root/j['id'])
        plan.update(output_root=str(root),registered_utc=now(),supersedes_launch_sha256=sha(file),
                    correction='actual original_rows observer seam; no physical outcome collected in failed attempt; same model/input/control/metrics')
        plan['research_source_sha256'].update({str(HERE/name):sha(HERE/name) for name in additions})
        check(plan)
        with (directory/file.name).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print('RETRY FROZEN; same192 commands, model, controls and metric oracle',flush=True)
if __name__=='__main__':main()
