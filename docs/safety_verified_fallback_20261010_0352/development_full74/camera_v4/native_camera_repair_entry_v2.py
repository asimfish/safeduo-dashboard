"""Explicit native mode only; --check-cpu imports no Isaac, Kit or torch."""
import argparse,os,sys,traceback
from pathlib import Path
import numpy as np
from repair_common_v2 import *
from native_measurements_v2 import create_environment_in_existing_app,native_parameter_readback
from parent_app_integration_v2 import run_in_existing_parent_app
from repair_phase_trace_v2 import PhaseTrace

def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ['request','request-sha256','review','review-sha256']:p.add_argument('--'+key,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-cpu',action='store_true')
    group.add_argument('--execute-native-diagnostic',action='store_true')
    p.add_argument('--permit');p.add_argument('--permit-sha256')
    return p

def verify_permit(args,request):
    require(args.permit and args.permit_sha256 and sha(args.permit)==args.permit_sha256,'parent live permit SHA')
    permit=read(args.permit)
    require(permit['request_sha256']==args.request_sha256 and permit['review_sha256']==args.review_sha256,'permit binding')
    identity=core.proc_identity(os.getppid())
    require(core.same_identity(identity,permit['parent_identity']),'live direct parent PID/startticks')
    import datetime
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(permit['observed_utc'])).total_seconds()
    require(0<=age<=30,'fresh parent prelaunch permit')
    lease=permit['serial_lease'];fd=lease['kernel_lock']['fd']
    evidence=Path(f'/proc/{identity["pid"]}/fdinfo/{fd}').read_text()
    require(any(line.split(':',1)[0]=='lock' and line.split(':',1)[1].split()==lease['kernel_lock']['lock_record']
                for line in evidence.splitlines() if ':' in line),'parent continuous locked FD still held')
    resource_gate(permit['resources'],initial=True)
    require(request['source_files'].get(str(Path(__file__).resolve()))==sha(__file__),'entry source')

def save_measurement(root,label,value):
    params=root/('PARAMETER_READBACK_'+label+'_V1.npz')
    with params.open('xb') as stream:
        np.savez_compressed(stream,**value['parameters']);stream.flush();os.fsync(stream.fileno())
    geom=root/('GEOMETRY_'+label+'_V1.json');write(geom,value['geometry'])
    return dict(parameters=dict(path=str(params),sha256=sha(params)),geometry=dict(path=str(geom),sha256=sha(geom)))

def verify_measurement(before,after):
    require(before['geometry']==after['geometry'],'original 9021 geometry identity changed')
    require(before['parameters'].keys()==after['parameters'].keys(),'parameter inventory changed')
    for k,a in before['parameters'].items():
        b=after['parameters'][k]
        require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'parameter bytes changed: '+k)

def main():
    p=parser();args,unknown=p.parse_known_args();assert_threads()
    request,binding,out=validate_pair(args.request,args.request_sha256,args.review,args.review_sha256,
                                    native=args.execute_native_diagnostic)
    if args.check_cpu:
        require(not unknown and args.permit is None,'CPU mode accepts no native launcher options')
        require(not any(k=='torch' or k.startswith(('isaac','omni.')) for k in sys.modules),'CPU imported GPU/native module')
        print('CPU_REQUEST_VALID_NATIVE_NOT_STARTED',flush=True);return
    verify_permit(args,request)
    lifecycle=validated_lifecycle(args.request,args.request_sha256,args.review,args.review_sha256,native=True)
    # All validation precedes native imports. Only parent runtime can supply permit.
    from isaaclab.app import AppLauncher
    class SingleThreadAppLauncher(AppLauncher):
        def _config_resolution(self,launcher_args):
            super()._config_resolution(launcher_args)
            self._sim_app_config['limit_cpu_threads']=1
            root=Path(request['attempt_root'])
            self._sim_app_config.setdefault('extra_args',[]).extend([
                '--/log/file='+str(root/'logs/kit.log'),
                '--/app/tokens/cache='+str(root/'cache/ov'),
                '--/app/tokens/data='+str(root/'data/ov'),
                '--/app/userConfigPath='+str(root/'config/kit.config.json')])
    AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    require(args.headless and args.enable_cameras and args.device=='cuda:0' and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='0','fixed GPU0/headless/strong camera')
    root=Path(request['attempt_root']);entry=root/'entry'
    entry.mkdir(exist_ok=False)
    state=dict(schema='safeduo.camera_repair_parent_entry.v1',status='failed',
        request_sha256=args.request_sha256,review_sha256=args.review_sha256,
        scope='SINGLE_CASE_DEVELOPMENT_CAMERA_DIAGNOSTIC_ONLY',physical_steps=0,policy_actions=0,
        prefix_qualification=False,aggregate_acceptance=False,hidden_solver_state_restored=False,
        CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN',cleanup_errors=[])
    def write_entry(value):write(entry/'ENTRY_RECEIPT_V1.json',value)
    def write_error(value):write(entry/'APP_CLOSE_ERROR_V1.json',value)
    trace=app=launcher=None;delegated=False
    try:
        trace=PhaseTrace(root);trace.phase('BEFORE_APP_LAUNCH')
        launcher=SingleThreadAppLauncher.__new__(SingleThreadAppLauncher)
        launcher.__init__(args);app=launcher.app
        trace.phase('AFTER_APP_LAUNCH');assert_threads()
        with np.load(binding['parameter_path'],allow_pickle=False) as z:params={k:z[k] for k in z.files}
        def measure(env):
            check_parameters(env,params)
            return dict(parameters=native_parameter_readback(env),geometry=geometry_identity(env))
        delegated=True
        run_in_existing_parent_app(app,lambda:create_environment_in_existing_app(args.device),
            collector_factory=lambda env:CAMERA_REPAIR_DIAGNOSTIC(env,args.request,args.request_sha256,args.review,args.review_sha256),
            read_measurements=measure,save_before=lambda x:save_measurement(entry,'BEFORE',x),
            save_after=lambda x:save_measurement(entry,'AFTER',x),verify_measurements=verify_measurement,
            write_entry=write_entry,write_app_error=write_error,state=state,phase=trace.phase,close_trace=trace.close,
            request_path=args.request,request_sha256=args.request_sha256,review_path=args.review,review_sha256=args.review_sha256)
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if not delegated:
            # Failures after AppLauncher but before integration (including params,
            # thread assertion, trace I/O) also always attempt the available app.
            if app is None and launcher is not None:app=getattr(launcher,'_app',None)
            try:
                if trace is not None:trace.close()
            except BaseException as exc:
                state['cleanup_errors'].append('trace: '+type(exc).__name__+': '+str(exc))
            finally:lifecycle.close_environment_and_app(state,None,app,write_entry,write_error)

if __name__=='__main__':main()
