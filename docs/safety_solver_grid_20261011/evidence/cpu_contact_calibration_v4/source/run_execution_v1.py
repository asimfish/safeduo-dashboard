"""Use the existing scoped NVML recovery for the finite calibration supervisor."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parent
H=P.parent
def main():
    assert not (P/'SUPERVISOR_SUBPROCESS_EXIT_V1.json').exists()
    env=dict(os.environ,LD_LIBRARY_PATH=json.loads((H/'fourhand_native_probe_v45/REGISTRATION_V1.json').read_text())['scoped_driver_library_path'])
    result=subprocess.run(['/usr/bin/python3','-B',str(H/'execute_physical_plan_v4.py'),str(P/'PLAN_V1.json')],env=env,cwd=H)
    (P/'SUPERVISOR_SUBPROCESS_EXIT_V1.json').write_text(json.dumps(dict(actual_subprocess_exit=result.returncode,actual_outer_tool_exit=None),indent=2)+'\n')
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
