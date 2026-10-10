"""Only owns bounded CPU readers/tests; never launches a native producer."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import select
import signal
import sys
import time
from evidence_io import HERE,OUTPUT,THREADS,contained,cpu_limits,require,sha,write_new


def identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+1:].split()
    require(int(raw[:raw.find('(')])==pid,'owned proc PID')
    return dict(pid=pid,ppid=int(v[1]),pgid=int(v[2]),sid=int(v[3]),startticks=int(v[19]))


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_hashes():
    return {str(p):sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.txt')}


def wait_cpu(script,args,out,max_wall=1800):
    script=contained(script,HERE)
    require(script.name in ('angle_diagnostic.py',),'fixed CPU entrypoints only')
    out=contained(out,OUTPUT);out.mkdir(parents=True,exist_ok=False)
    require(0<max_wall<=7200,'bounded CPU deadline')
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    environment.update({k:'1' for k in THREADS});environment['TMPDIR']=str(out)
    argv=[sys.executable,'-B',str(script),*map(str,args)]
    sources=source_hashes();started=utc();clock=time.monotonic()
    rr,rw=os.pipe();gr,gw=os.pipe();pid=None;owned=None;raw=None;usage=None;waited=None;abort=None;signals=[]
    with (out/'child.log').open('xb') as log:
        try:
            pid=os.fork()
            if pid==0:
                try:
                    os.close(rr);os.close(gw);os.setsid()
                    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
                    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
                    os.write(rw,b'R');os.close(rw)
                    require(os.read(gr,1)==b'G','parent enrollment gate');os.close(gr)
                    os.chdir(out);os.execve(sys.executable,argv,environment)
                except BaseException:
                    import traceback
                    traceback.print_exc();os._exit(127)
            os.close(rw);rw=None;os.close(gr);gr=None
            require(select.select([rr],[],[],5)[0] and os.read(rr,1)==b'R','CPU fork handshake')
            owned=identity(pid)
            require(owned['ppid']==os.getpid() and owned['pgid']==owned['sid']==pid,'own direct fork identity')
            write_new(out/'ENROLLED.json',dict(schema='astra.full74.cpu_enrollment.v1',parent_identity=identity(os.getpid()),
                child_identity=owned,provenance='direct os.fork, pipe enrollment before exec',argv=argv,sources=sources))
            os.write(gw,b'G');os.close(gw);gw=None
            deadline=clock+max_wall
            while True:
                found,status,rusage=os.wait4(pid,os.WNOHANG)
                if found:waited,raw,usage=found,status,rusage;break
                if time.monotonic()>=deadline:raise TimeoutError('CPU own child exceeded wall bound')
                time.sleep(.03)
        except BaseException as exc:
            abort=type(exc).__name__+': '+str(exc)
        finally:
            for fd in (rr,rw,gr,gw):
                if fd is not None:
                    try:os.close(fd)
                    except OSError:pass
            if pid and raw is None:
                # Direct unreaped child cannot have its PID reused. No process group or foreign signal.
                now=identity(pid)
                require(now['ppid']==os.getpid() and (owned is None or now==owned),'owned child changed before cleanup')
                os.kill(pid,signal.SIGKILL);signals.append(dict(pid=pid,startticks=now['startticks'],signal=int(signal.SIGKILL)))
                waited,raw,usage=os.wait4(pid,0)
            log.flush();os.fsync(log.fileno())
    after=source_hashes()
    if sources!=after:abort='executed CPU source changed'
    report=dict(schema='astra.full74.independent_owned_cpu_wait.v1',backend='CPU_ONLY_NEVER_NATIVE',child_pid=pid,
        waited_pid=waited,child_identity=owned,parent_identity=identity(os.getpid()),waitpid_observed=raw is not None,
        raw_wait_status=raw,actual_wait_exit=os.waitstatus_to_exitcode(raw) if raw is not None else None,
        wait_mechanism='os.wait4(owned_pid, WNOHANG); blocking wait4 only for own timeout cleanup',
        started_utc=started,closed_utc=utc(),elapsed_seconds=time.monotonic()-clock,resource_abort=abort,signals=signals,
        max_rss_KiB=usage.ru_maxrss if usage else None,max_address_space_bytes=1024**3,threads=1,bytecode=False,
        argv=argv,source_before=sources,source_after=after,log=str(out/'child.log'),log_sha256=sha(out/'child.log'),
        enrollment=str(out/'ENROLLED.json'),enrollment_sha256=sha(out/'ENROLLED.json') if (out/'ENROLLED.json').exists() else None,
        native_pass=False,safety_acceptance=False)
    write_new(out/'ACTUAL_WAIT.json',report)
    return report


def main():
    cpu_limits()
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True)
    a=p.parse_args();out=contained(a.out,OUTPUT)
    w=wait_cpu(HERE/'angle_diagnostic.py',['--proof-dir',str(out/'proofs')],out,900)
    result=dict(status='CLOSED_CPU_ANGLE_DIAGNOSTIC_ONLY' if w['actual_wait_exit']==0 and not w['resource_abort'] else 'FAILED_CPU_DIAGNOSTIC',
        actual_wait_exit=w['actual_wait_exit'],raw_wait_status=w['raw_wait_status'],child_pid=w['child_pid'],startticks=w['child_identity']['startticks'],
        actual_wait=str(out/'ACTUAL_WAIT.json'),actual_wait_sha256=sha(out/'ACTUAL_WAIT.json'),log_sha256=w['log_sha256'],
        source_hashes=w['source_before'],max_rss_KiB=w['max_rss_KiB'],native_pass=False,safety_acceptance=False)
    if result['status']=='CLOSED_CPU_ANGLE_DIAGNOSTIC_ONLY':
        result['report']=str(out/'proofs/ANGLE_DIAGNOSTIC.json');result['report_sha256']=sha(result['report'])
    write_new(out/'CLOSED_DELIVERY.json',result);print(json.dumps(result),flush=True)
    return 0 if result['status']=='CLOSED_CPU_ANGLE_DIAGNOSTIC_ONLY' else 2

if __name__=='__main__':raise SystemExit(main())
