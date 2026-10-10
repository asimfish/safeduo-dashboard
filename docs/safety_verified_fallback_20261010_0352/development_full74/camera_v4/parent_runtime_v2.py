"""Parent-invoked serial supervisor. Default/read-only mode cannot launch native."""
import argparse,datetime,resource,time
from repair_common_v2 import *
from request_review_builder_v2 import review_plan,freeze_after_review
from fresh_status_v2 import fresh_status

def dead_identity(identity):
    status=fresh_status(dict(pid=int(identity['pid']),startticks=int(identity['startticks'])))
    require(status['process_stat_error'] is None and not status['same_PID_startticks'],'registered owned identity still present/unknown')
    return status

def inventory():
    records=[];sources={};seen=set()
    def add(identity,origin):
        pair=(int(identity['pid']),int(identity['startticks']))
        if pair in seen:return
        seen.add(pair);records.append(dict(identity=dict(pid=pair[0],startticks=pair[1]),
            source=str(origin),source_sha256=sha(origin),fresh_status=dead_identity(identity)))
    for p in sorted(H.glob('*_execution.json')):
        value=read(p)
        if not value.get('jobs'):continue
        sources[str(p)]=sha(p)
        require(value.get('status') in ['complete','failed'],'owned registered campaign not closed: '+p.name)
        for job in value['jobs']:
            if 'pid' not in job:continue
            require(type(job.get('actual_exit')) is int,'owned job lacks actual terminal receipt')
            add(dict(pid=job['pid'],startticks=int(job['start_ticks'])),p)
    waits=list((R/'astra_full74_screen_native_v2').glob('parent_supervision_*/PARENT_NATIVE_WAIT.json'))
    waits+=list(BASE.glob('*/native_process/ACTUAL_WAIT.json'))
    waits.append(V9/'native_process/ACTUAL_WAIT.json')
    for p in sorted(set(waits)):
        wait=read(p);sources[str(p)]=sha(p)
        require(wait['waitpid_observed'] is True and wait['waited_pid']==wait['child_pid'] and
            type(wait['actual_wait_exit']) is int and os.waitstatus_to_exitcode(wait['raw_wait_status'])==wait['actual_wait_exit'],
            'prior real native wait absent/inconsistent')
        for identity in wait['owned_identities']:add(identity,p)
    return dict(observed_utc=core.utc(),registered_owned_inventory=records,source_files=sources,
        scope='REGISTERED_OWNED_ONLY_PLUS_EXPLICIT_PARENT_COMPLETENESS_DECLARATION',foreign_registry_read=False)

def predecessors():
    native=read(V9/'native_process/ACTUAL_WAIT.json');outer=read(H/'full74_screen_parent_failure_v9_execution.json')
    require(native['actual_wait_exit']==-9 and native['raw_wait_status']==9 and native['waitpid_observed'] is True
        and native['waited_pid']==native['child_pid'] and outer['actual_exit']==1 and outer.get('closed_utc'),
        'exact V9 failed native-9/raw9/outer1 actual reaped evidence')
    require(all(x['live'] is False for x in native['owned_identities']),'V9 all recorded owned tasks dead')
    for identity in native['owned_identities']:dead_identity(identity)
    for key in ['log','outer_monitor']:require(sha(native[key])==native[key+'_sha256'],'V9 failed proof bytes')
    return dict(native_wait_sha256=sha(V9/'native_process/ACTUAL_WAIT.json'),
        outer_wait_sha256=sha(H/'full74_screen_parent_failure_v9_execution.json'),native_actual=-9,outer_actual=1,
        old858_camera_images_remain_failed=True)

def child_environment(root,*,native):
    env={k:os.environ[k] for k in ['HOME','USER','LOGNAME','PATH','LANG','LC_ALL','LD_LIBRARY_PATH'] if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0' if native else '',
        PYTHONPATH='/home/liyufeng/safeduo/src',OMNI_KIT_ACCEPT_EULA='YES',PRIVACY_CONSENT='Y',
        TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'),
        XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'config'),XDG_DATA_HOME=str(root/'data'),
        CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCH_HOME=str(root/'cache/torch'),
        WARP_CACHE_PATH=str(root/'cache/warp'))
    return env

def validate_products(root,request_sha,wait):
    successful_wait(wait)
    require(wait['backend']=='isaac_physx_native','native backend provenance')
    entry_path=root/'entry/ENTRY_RECEIPT_V1.json'
    entry=read(entry_path) # Missing ENTRY MUST reject even app.close's actual0.
    require(entry['status']=='diagnostic_complete_only' and entry['request_sha256']==request_sha and
            entry.get('measurement_bytes_verified') is True and not entry.get('cleanup_errors'),
            'successful entry AND immutable measurement bytes required')
    require(not (root/'entry/APP_CLOSE_ERROR_V1.json').exists(),'app close error sidecar')
    receipt_path=root/'native/CAMERA_REPAIR_DIAGNOSTIC_RECEIPT_V2.json'
    receipt=read(receipt_path)
    require(receipt['status']=='SINGLE_CASE_OBSERVATION_ONLY' and receipt['request_sha256']==request_sha and
        receipt['case']==CASE and receipt['native_hold']['status']=='pass' and receipt['strict_group']['status']=='qualified' and
        receipt['v9_point0_exact_replay']['status']=='pass' and
        receipt['physical_steps']==receipt['policy_actions']==0 and receipt['prefix_qualification'] is False,
        'closed exact single-case held27 diagnostic required')
    for label in ['before','after']:
        for ref in entry['measurement_boundary'][label].values():bound_bytes(ref['path'],ref['sha256'])
    return dict(status='NATIVE_DIAGNOSTIC_CLOSED_REQUIRES_INDEPENDENT_RAW_AUDIT',
        entry=dict(path=str(entry_path),sha256=sha(entry_path)),diagnostic=dict(path=str(receipt_path),sha256=sha(receipt_path)),
        independent_raw_acceptance=False,full32_qualified=False,prefix_qualified=False,
        physical_safety_certified=False,CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN')

def execute(args):
    plan,approval,draft=review_plan(args.plan,args.plan_sha256,args.approval,args.approval_sha256)
    predecessors()
    # One acquisition for the whole pipeline; no repeated flock or PGID ownership.
    with SerialLease(SERIAL_LOCK) as lease:
        serial=inventory();serial['lease']=lease.assert_held()
        fresh=resource_gate(core.probe_resources(gpu=0),initial=True)
        # Recheck after probes, before request freeze. No stale30s registration.
        age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(serial['observed_utc'])).total_seconds()
        require(0<=age<=30,'fresh owned inventory age <=30s')
        for row in serial['registered_owned_inventory']:dead_identity(row['identity'])
        core.verify(serial['source_files']);lease.assert_held()
        root=Path(plan['attempt_root']);require(not root.exists(),'new attempt only')
        root.mkdir(parents=True,exist_ok=False)
        for name in ['tmp','cache','config','data','logs']:(root/name).mkdir()
        for name in ['ov','cuda','torch','warp']:(root/'cache'/name).mkdir()
        (root/'data/ov').mkdir()
        write(root/'FRESH_PARENT_SERIAL_INVENTORY_V1.json',serial)
        write(root/'PRELAUNCH_RESOURCES_V1.json',fresh)
        state=dict(status='failed',started_utc=core.utc(),parent_identity=core.proc_identity(os.getpid()),
            request_frozen=False,native_launched=False)
        try:
            rp,rs,vp,vs=freeze_after_review(root,draft,plan,approval,args.approval,args.approval_sha256)
            state['request_frozen']=True
            base=[PY,'-B',str(HERE/'native_camera_repair_entry_v2.py'),'--request',str(rp),'--request-sha256',rs,
                  '--review',str(vp),'--review-sha256',vs]
            cpu=core.bounded_child(base+['--check-cpu'],child_environment(root,native=False),root/'cpu_check',
                backend='CPU_REQUEST_CHECK_ONLY',max_wall=120,cpu_limit=True)
            successful_wait(cpu);state['cpu_check_actual_wait']=cpu
            review_plan(args.plan,args.plan_sha256,args.approval,args.approval_sha256)
            validate_pair(rp,rs,vp,vs,native=True)
            inventory() # Fresh scan again after CPU wait, with same continuous lock.
            fresh=resource_gate(core.probe_resources(gpu=0),initial=True)
            permit=dict(observed_utc=core.utc(),parent_identity=core.proc_identity(os.getpid()),
                request_sha256=rs,review_sha256=vs,resources=fresh,serial_lease=lease.assert_held())
            pp=root/'NATIVE_LIVE_PERMIT_V1.json';write(pp,permit)
            previous_write=None
            with (root/'FRESH_STATUS_V1.jsonl').open('x') as status_stream:
                def monitor(tree,deadline):
                    nonlocal previous_write
                    lease.assert_held()
                    observation=core.probe_resources(tree,gpu=0,deadline=deadline)
                    identity=tree.members[tree.root_pid]['identity'] if hasattr(tree,'root_pid') else tree.members[next(iter(tree.members))]['identity']
                    status=fresh_status(identity,log_path=root/'native_process/child.log',previous_log_write_ns=previous_write)
                    previous_write=status['log_file'].get('last_write_ns')
                    core.append_event(status_stream,status)
                    return observation
                state['native_launched']=True
                wait=core.bounded_child(base+['--execute-native-diagnostic','--permit',str(pp),'--permit-sha256',sha(pp),
                    '--headless','--enable_cameras','--device','cuda:0'],child_environment(root,native=True),
                    root/'native_process',backend='isaac_physx_native',max_wall=3600,monitor=monitor)
            write(root/'PARENT_NATIVE_WAIT_V1.json',wait);state['native_actual_wait']=wait
            result=validate_products(root,rs,wait);write(root/'NATIVE_DIAGNOSTIC_CLOSED_V1.json',result)
            state['status']='closed_diagnostic_independent_raw_audit_required'
        except BaseException as exc:
            import traceback
            state.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            state['closed_utc']=core.utc();state['final_serial_lease']=lease.assert_held()
            write(root/'PARENT_CONTINUATION_CLOSED_V1.json',state)

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ['plan','plan-sha256','approval','approval-sha256']:p.add_argument('--'+name,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-review-cpu',action='store_true');mode.add_argument('--execute-after-parent-review',action='store_true')
    args=p.parse_args();assert_threads()
    _,hard=resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,hard))
    if args.check_review_cpu:
        review_plan(args.plan,args.plan_sha256,args.approval,args.approval_sha256)
        print('PARENT_REVIEW_VALID_NO_FREEZE_NO_NATIVE');return
    require(hard==resource.RLIM_INFINITY,'native parent requires unlimited hard AS; own soft AS1GiB/RSS guard unchanged')
    execute(args)

if __name__=='__main__':main()
