"""Finish this already-running finite experiment on owned-process exit events."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--pids',type=int,nargs='+',required=True);args=parser.parse_args()
state=dict(status='waiting_for_owned_campaigns',pids=args.pids,started_utc=datetime.now(timezone.utc).isoformat(),
           wait_mechanism='Linux pidfd process-exit event; no polling of simulation or conversation',
           final_action='analysis, 19 contracts, legacy preservation, local and live browser checks; normal scoped Git publication')
path=HERE/'finalization_state.json'
assert not path.exists(),'one finalizer only'


def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)


save();fds=[]
try:
    for pid in args.pids:
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert any(x.endswith('/isolated_campaign_v2.py') for x in command),command
            assert any('safety_random_space_20261004/campaign_plan_' in x for x in command),command
            fds.append(os.pidfd_open(pid))
        except ProcessLookupError:pass
        except FileNotFoundError:pass
    while fds:
        ready,_,_=select.select(fds,[],[],60)
        for fd in ready:os.close(fd);fds.remove(fd)
    state.update(status='validating_and_publishing',campaigns_exited_utc=datetime.now(timezone.utc).isoformat());save()
    with (HERE/'final_release.log').open('x') as log:
        result=subprocess.run(['/home/liyufeng/miniforge3/envs/safeduo/bin/python',str(HERE/'release.py')],
                              cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'final evidence gate failed; prior public results and complete logs retained'
    delivery=json.loads((HERE/'DELIVERY.json').read_text())
    assert delivery['campaign_status']=='complete' and delivery['pending_windows']==0
    state.update(status='complete',delivery=delivery,finished_utc=datetime.now(timezone.utc).isoformat());save()
except BaseException as error:
    state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=datetime.now(timezone.utc).isoformat());save();raise
finally:
    for fd in fds:os.close(fd)
