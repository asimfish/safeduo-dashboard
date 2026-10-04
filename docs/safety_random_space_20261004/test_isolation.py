import hashlib
import json
from pathlib import Path
import sys

import pytest

from isolated_campaign import run_campaign


def plan(tmp_path):
    child=tmp_path/'child.py'
    child.write_text('''import json,pathlib,sys
out=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
(out/'protocol.json').write_text(json.dumps(dict(status='complete',completed_cells=1,design=[{'seed':99}])))
(out/'summary.json').write_text(json.dumps(dict(windows=[{'seed':99}])))
if sys.argv[1]=='fail':sys.exit(7)
''')
    return dict(cwd=str(tmp_path),continue_after_child_failure=True,
                jobs=[dict(id=name,argv=[sys.executable,str(child),mode],expected={'seed':99})
                      for name,mode in [('first','fail'),('second','ok')]])


def test_independent_failure_retained_next_runs_and_campaign_not_green(tmp_path):
    p=plan(tmp_path);out=tmp_path/'out';m=run_campaign(p,out)
    assert m['status']=='complete_with_failures'
    assert [j['status'] for j in m['jobs']]==['failed','complete']
    assert m['jobs'][0]['exit_code']==7
    assert (out/'first'/'protocol.json').exists() and (out/'first.log').exists()
    assert m['jobs'][0]['pid']!=m['jobs'][1]['pid']
    with pytest.raises(ValueError,match='overwritten'):run_campaign(p,out)


def test_actor_drift_aborts_without_launch(tmp_path):
    p=plan(tmp_path);actor=tmp_path/'actor.pt';actor.write_bytes(b'changed')
    p.update(checkpoint_path=str(actor),checkpoint_sha256=hashlib.sha256(b'old').hexdigest())
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='actor changed'):run_campaign(p,out)
    m=json.loads((out/'campaign.json').read_text());assert m['status']=='failed' and m['jobs']==[]


def test_helper_drift_aborts_without_launch(tmp_path):
    p=plan(tmp_path);f=tmp_path/'helper.py';f.write_text('changed')
    p['research_source_sha256']={str(f):hashlib.sha256(b'old').hexdigest()}
    with pytest.raises(ValueError,match='helper changed'):run_campaign(p,tmp_path/'out')
