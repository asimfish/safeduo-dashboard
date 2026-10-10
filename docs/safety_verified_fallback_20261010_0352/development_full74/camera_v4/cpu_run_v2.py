"""Only named CPU preparation/tests; real owned fork/wait, all logs on NAS."""
import argparse,resource
from repair_common_v2 import *

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--script',required=True,choices=['test_bootstrap_environment_v4.py','prepare_reused_release_v4.py'])
    p.add_argument('--tag',required=True);args=p.parse_args()
    require(args.tag.isalnum(),'exclusive simple tag');assert_threads()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU wrapper must hide CUDA')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    CPU.mkdir(exist_ok=True)
    wait=core.bounded_child([PY,'-B',str((CANDIDATE if args.script=='test_actual_angles_v2.py' else HERE)/args.script)],dict(os.environ),CPU/args.tag,
        backend='CPU_ONLY_NO_NATIVE_AUTHORITY',max_wall=120,cpu_limit=True)
    print(json.dumps(wait),flush=True)
    sys.exit(wait['actual_wait_exit'] if wait['actual_wait_exit'] is not None else 125)

if __name__=='__main__':main()
