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
    require(script.name in ('failed_reader.py','test_diagnostic.py','fixture_child.py'),'fixed CPU entrypoints only')
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
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('cpu-tests','diagnose-closed'),required=True);p.add_argument('--out',required=True)
    p.add_argument('--attempt-root');p.add_argument('--parent-wait-sha256');p.add_argument('--request-sha256')
    p.add_argument('--outer-execution-sha256')
    p.add_argument('--max-wall-seconds',type=int,default=1800)
    a=p.parse_args();out=contained(a.out,OUTPUT)
    if a.mode=='cpu-tests':
        require(not any((a.attempt_root,a.parent_wait_sha256,a.request_sha256,a.outer_execution_sha256)),'CPU tests have no native target')
        script=HERE/'test_diagnostic.py'
        args=['--proof-dir',str(out/'proofs')]
    else:
        require(all((a.attempt_root,a.parent_wait_sha256,a.request_sha256,a.outer_execution_sha256)),'explicit actual closure bindings required')
        # Check closure before even spawning a CPU reader. No background observer/poller.
        from failed_closure import preflight
        preflight(a)
        script=HERE/'failed_reader.py';args=['--attempt-root',a.attempt_root,'--parent-wait-sha256',a.parent_wait_sha256,'--request-sha256',a.request_sha256,'--output',str(out/'FAILED_INITIAL_DIAGNOSTIC.json')]
        args+=['--outer-execution-sha256',a.outer_execution_sha256]
    w=wait_cpu(script,args,out,a.max_wall_seconds)
    summary=dict(schema='astra.full74.independent_cpu_delivery.v1',mode=a.mode,actual_wait_exit=w['actual_wait_exit'],
        raw_wait_status=w['raw_wait_status'],child_pid=w['child_pid'],startticks=w['child_identity']['startticks'] if w['child_identity'] else None,
        actual_wait=str(out/'ACTUAL_WAIT.json'),actual_wait_sha256=sha(out/'ACTUAL_WAIT.json'),
        source_hashes=w['source_before'],log_sha256=w['log_sha256'],max_rss_KiB=w['max_rss_KiB'],native_pass=False,safety_acceptance=False)
    if w['actual_wait_exit']==0 and w['resource_abort'] is None:
        product=out/'FAILED_INITIAL_DIAGNOSTIC.json' if a.mode=='diagnose-closed' else out/'proofs/CPU_TEST_REPORT.json'
        summary['product']=str(product);summary['product_sha256']=sha(product)
        summary['status']='CLOSED_CPU_FAILED_INITIAL_DIAGNOSTIC_ONLY' if a.mode=='diagnose-closed' else 'CLOSED_CPU_TESTS_ONLY'
    else:summary['status']='FAILED_CPU_CHILD'
    write_new(out/'CLOSED_DELIVERY.json',summary);print(json.dumps(summary),flush=True)
    return 0 if summary['status']!='FAILED_CPU_CHILD' else 2


if __name__=='__main__':raise SystemExit(main())
