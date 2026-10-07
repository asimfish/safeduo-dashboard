"""Owned local preview with an OS-assigned free port and exact PID identity."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from datetime import datetime,timezone
import os,json
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(WORK)))
with (HERE/'PREVIEW_EXECUTION.json').open('x') as f:json.dump(dict(status='OWNED_READONLY_PREVIEW_RUNNING',pid=os.getpid(),argv=['python3',str(Path(__file__).resolve())],process_start_ticks=Path('/proc/self/stat').read_text().split()[21],root='http://127.0.0.1:'+str(server.server_port)+'/',log='/tmp/safeduo_zero_http_dynamic_20261006.log',utc=datetime.now(timezone.utc).isoformat()),f,indent=2)
print('PREVIEW_LISTENING',server.server_port,flush=True)
server.serve_forever()
