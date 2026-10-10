"""CPU-only bindings for the prospective, single-counterexample camera repair."""
import hashlib,json,os,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=HERE.parent
R=Path('/mnt/nas/data/lyf/double_hand')/H.name
BASE=R/'astra_full74_camera_repair_development_v1'
CPU=R/'native_camera_repair_CPU_v4'
CANDIDATE=H/'astra/full74_camera_upgrade_v3'
LIFECYCLE=H/'native_measurement_shutdown_v2/lifecycle_v2.py'
LIFE_WAIT=H/'native_measurement_shutdown_cpu_v2_retry1_execution.json'
LIFE_REPORT=R/'native_measurement_shutdown_CPU_v2/CPU_MEASUREMENT_SHUTDOWN_REPORT_V2.json'
V9=R/'astra_full74_screen_native_v2/parent_supervision_failure_v2_runtime_v9_attempt1'
SERIAL_LOCK=R/'astra_full74_screen_native_v2/.parent_owned_native_serial_v2.lock'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
RESERVE=4*1024**3
NAS_FLOOR=32*1024**3
for folder in [H/'astra/full74_screen',H/'astra/full74_screen/parent_supervision_v1',
        H/'astra/full74_screen/parent_supervision_v2',H/'astra/full74_screen/parent_supervision_v3',
        H/'parent_runtime_identity_v4',H/'native_root_restore_equivalence_v1',
        H/'native_measurement_shutdown_v2',CANDIDATE]:
    sys.path.insert(0,str(folder))
import runtime_core_v4 as core
from serial_lease_v3 import SerialLease
from camera_repair_diagnostic_v2 import CASE,check_request,CAMERA_REPAIR_DIAGNOSTIC
from native_full74_screen_v3 import array,geometry_identity,check_parameters,frozen_import_paths
from screen_core_v1 import ARMS,N
require=core.require
sha=core.sha
read=core.read
write=core.write_new

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def digest(value):return hashlib.sha256(encoded(value)).hexdigest()

def write_exact(path,value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())

def bound_bytes(path,expected,maximum=16*1024**2):
    with Path(path).open('rb') as stream:raw=stream.read(maximum+1)
    require(len(raw)<=maximum and hashlib.sha256(raw).hexdigest()==expected,'bounded byte SHA: '+str(path))
    return raw

def bound_json(path,expected):
    return json.loads(bound_bytes(path,expected),object_pairs_hook=core.unique_pairs)

def lifecycle_proof():
    wait,report=read(LIFE_WAIT),read(LIFE_REPORT)
    require(wait['actual_exit']==0 and wait['pid']==4082375 and wait.get('closed_utc'),'lifecycle V2 realwait0')
    require(report['status']=='PASS_CPU_MEASUREMENT_SHUTDOWN_ONLY' and report['tests']==8 and
            report['failures']==report['errors']==0,'lifecycle V2 CPU8 proof')
    core.verify(report['sources']);core.verify(report['artifacts'])
    require(report['sources'][str(LIFECYCLE)]==sha(LIFECYCLE),'lifecycle V2 source proof')
    return dict(wait=str(LIFE_WAIT),wait_sha256=sha(LIFE_WAIT),report=str(LIFE_REPORT),report_sha256=sha(LIFE_REPORT))

def validate_pair(request_path,request_sha,review_path,review_sha,*,native=False):
    request_snapshot=bound_json(request_path,request_sha)
    review=bound_json(review_path,review_sha)
    request,binding,out=check_request(request_path,request_sha,review_path,review_sha)
    require(request==request_snapshot,'request changed during legacy checker')
    request=request_snapshot
    require(request['entry_protocol']=='CAMERA_REPAIR_ENTRY_V2_LIFECYCLE_V2','entry protocol')
    require(request['gpu_index']==0 and request['SSD_operational_reserve_bytes']==RESERVE and
            request['min_NAS_free_bytes']==NAS_FLOOR,'unchanged parent operating reserves')
    require(Path(request['attempt_root']).resolve()==out.parent and out.name=='native','exact NAS attempt layout')
    for p in HERE.glob('*.py'):
        require(request['source_files'].get(str(p))==sha(p),'all integration sources bound: '+p.name)
    require(request['source_files'].get(str(LIFECYCLE))==sha(LIFECYCLE),'lifecycle V2 mandatory')
    require(request['lifecycle_V2_proof']==lifecycle_proof(),'closed lifecycle provenance')
    if native:
        require(request.get('cpu_test_only') is False and review.get('cpu_test_only') is False,'CPU fixture has no native authority')
        require(review.get('parent_execute_authorized') is True,'independent parent native authorization')
    return request,binding,out

def validated_lifecycle(request_path,request_sha,review_path,review_sha,*,native=False):
    # No lifecycle module is imported at module load time. Expected request and
    # review hashes, then every source byte, are checked BEFORE executing V2.
    request,_,_=validate_pair(request_path,request_sha,review_path,review_sha,native=native)
    return module_from_verified_bytes(LIFECYCLE,request['source_files'][str(LIFECYCLE)])

def module_from_verified_bytes(path,expected):
    import types
    raw=bound_bytes(path,expected,1024**2)
    module=types.ModuleType('_bound_measurement_lifecycle_v2')
    module.__file__=str(path)
    # Execute precisely the verified buffer, never spec.exec_module/reread path.
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

def successful_wait(wait):
    require(wait.get('actual_wait_exit')==0 and wait.get('raw_wait_status')==0 and
            wait.get('waitpid_observed') is True and wait.get('waited_pid')==wait.get('child_pid') and
            type(wait.get('child_pid')) is int and wait['child_pid']>0 and
            wait.get('resource_abort') is None and wait.get('error') is None,'real reaped wait0 required')
    require(all(x.get('live') is False for x in wait['owned_identities']),'owned descendants must be dead')
    for path_key in ['log','outer_monitor']:
        require(sha(wait[path_key])==wait[path_key+'_sha256'],'wait artifact SHA')

def resource_gate(observation,*,initial):
    failures=core.resource_failures(observation,initial=initial,nas_floor=NAS_FLOOR)
    if initial and observation['SSD_free_bytes']<core.LIMITS['min_SSD_free_bytes']+RESERVE:
        failures.append('SSD_operational_reserve_4GiB')
    require(not failures,'resource gates: '+','.join(failures))
    return observation

def assert_threads():
    require(sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Python -B required')
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']), 'single thread environment')
