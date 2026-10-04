import json
import sys

from isolated_campaign_v2 import run_campaign


def test_zero_exit_failed_protocol_is_retained_and_next_condition_runs(tmp_path):
    child=tmp_path/'child.py'
    child.write_text('''import json,pathlib,sys
out=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
failed=sys.argv[1]=='failed'
p=dict(status='failed' if failed else 'complete',completed_cells=0 if failed else 1,design=[dict(seed=1)])
if failed:p['error']='critical-row union exceeds declared capacity: 131 > 128'
(out/'protocol.json').write_text(json.dumps(p))
if not failed:(out/'summary.json').write_text(json.dumps(dict(windows=[dict(seed=1)])))
''')
    p=dict(cwd=str(tmp_path),continue_after_child_failure=True,
           jobs=[dict(id=name,argv=[sys.executable,str(child),mode],expected=dict(seed=1))
                 for name,mode in [('bad','failed'),('next','complete')]])
    out=tmp_path/'out';m=run_campaign(p,out)
    assert m['status']=='complete_with_failures'
    assert [j['status'] for j in m['jobs']]==['failed','complete']
    assert m['jobs'][0]['exit_code']==0
    assert '131 > 128' in m['jobs'][0]['error']
    assert json.loads((out/'bad/protocol.json').read_text())['status']=='failed'
    assert json.loads((out/'next/protocol.json').read_text())['status']=='complete'
