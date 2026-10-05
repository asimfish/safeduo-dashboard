"""Continue the authorized batch after its driver exits; fail closed on audits."""
from pathlib import Path
import argparse,json,os,select,subprocess,time
HERE=Path(__file__).resolve().parent
REPO=Path('/home/liyufeng/safeduo-dashboard-dynamics-prediction-20261006')
SIM='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
def write(value):(HERE/'postprocess_status.json').write_text(json.dumps(value,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--driver-pid',type=int,required=True);a=p.parse_args()
    write(dict(status='waiting_for_registered_physics',driver_pid=a.driver_pid,pid=os.getpid(),started_utc=time.time()))
    try:
        fd=os.pidfd_open(a.driver_pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    except ProcessLookupError:pass
    result=json.loads((HERE/'retry_completion.json').read_text());assert result['status']=='complete',result['status']
    write(dict(status='auditing_full_stream',pid=os.getpid(),utc=time.time()))
    with (HERE/'analysis.log').open('x') as log:
        subprocess.run([SIM,str(HERE/'score.py')],cwd=HERE.parent.parent,env={**os.environ,'PYTHONPATH':'src','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    write(dict(status='publishing_verified_final_results',pid=os.getpid(),utc=time.time()))
    subprocess.run(['git','fetch','origin','main'],cwd=REPO,check=True)
    subprocess.run(['git','merge','--ff-only','origin/main'],cwd=REPO,check=True)
    subprocess.run([SIM,str(HERE/'publish.py'),'--repo',str(REPO),'--phase','complete'],cwd=HERE.parent.parent,check=True)
    write(dict(status='complete',delivery=json.loads((HERE/'delivery_complete.json').read_text()),finished_utc=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(dict(status='blocked',error=type(error).__name__+': '+str(error),utc=time.time()));raise
