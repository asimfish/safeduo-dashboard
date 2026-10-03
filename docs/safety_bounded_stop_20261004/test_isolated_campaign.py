import json
from pathlib import Path
import sys

import pytest

from isolated_campaign import run_campaign


def plan(tmp_path, mode='complete'):
    child = tmp_path / 'child.py'
    child.write_text('''import json, os, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()
mode = sys.argv[1]
design = [{'seed': 12}] * (2 if mode == 'multi' else 1)
(out/'protocol.json').write_text(json.dumps(dict(status='complete',completed_cells=len(design),design=design)))
(out/'summary.json').write_text(json.dumps(dict(windows=[dict(seed=12)]*len(design))))
(out/'pid.txt').write_text(str(os.getpid()))
if mode == 'fail': sys.exit(7)
''')
    return dict(cwd=str(tmp_path), jobs=[dict(id=i, argv=[sys.executable, str(child), mode], expected={'seed':12})
                                       for i in ['first', 'second']])


def test_real_fresh_processes_and_no_overwrite(tmp_path):
    p = plan(tmp_path)
    out = tmp_path / 'out'
    m = run_campaign(p, out)
    assert m['status'] == 'complete'
    assert m['jobs'][0]['pid'] != m['jobs'][1]['pid']
    for j in m['jobs']:
        assert j['status'] == 'complete' and j['completed_cells'] == 1
        assert int((out/j['id']/'pid.txt').read_text()) == j['pid']
    prior = (out/'campaign.json').read_bytes()
    with pytest.raises(ValueError, match='overwritten'):
        run_campaign(p, out)
    assert (out/'campaign.json').read_bytes() == prior


@pytest.mark.parametrize('mode,exception', [('multi',ValueError),('fail',RuntimeError)])
def test_failed_child_preserves_evidence_and_does_not_run_next(tmp_path, mode, exception):
    p = plan(tmp_path, mode)
    out = tmp_path/'out'
    with pytest.raises(exception):
        run_campaign(p, out)
    m=json.loads((out/'campaign.json').read_text())
    assert m['status']=='failed' and m['jobs'][0]['status']=='failed'
    assert len(m['jobs'])==1 and not (out/'second').exists()
    assert (out/'first'/'protocol.json').exists() and (out/'first.log').exists()


def test_duplicate_ids_rejected_before_simulation(tmp_path):
    p=plan(tmp_path)
    p['jobs'][1]['id']='first'
    with pytest.raises(ValueError, match='unique'):
        run_campaign(p,tmp_path/'out')
    assert not (tmp_path/'out').exists()
