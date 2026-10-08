"""Owned temporary service; persist actual signal exit without inventing zero."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
WORK='/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_worktree_20261007'
if __name__=='__main__':
    argv=[sys.executable,'-B','-m','http.server','18917','--bind','127.0.0.1','--directory',WORK]
    with (HERE/'preview.log').open('xb') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]
        with (HERE/'PREVIEW_IDENTITY.json').open('x') as f:json.dump(dict(pid=child.pid,start_ticks=ticks,argv=argv),f,indent=2);f.write('\n')
        rc=child.wait()
    with (HERE/'PREVIEW_EXIT.json').open('x') as f:json.dump(dict(actual_exit=rc,pid=child.pid),f,indent=2);f.write('\n')
    print('PREVIEW_ACTUAL_EXIT',rc,flush=True)
