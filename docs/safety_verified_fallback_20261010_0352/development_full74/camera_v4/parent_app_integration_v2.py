"""Lifecycle V2 integration; callback failures cannot bypass owned app.close."""
import traceback
from repair_common_v2 import require,validated_lifecycle

def run_in_existing_parent_app(app,env_factory,*,collector_factory,read_measurements,
        save_before,save_after,verify_measurements,write_entry,write_app_error,
        state,request_path,request_sha256,review_path,review_sha256,
        phase=lambda _:None,close_trace=lambda:None):
    try:
        lifecycle=validated_lifecycle(request_path,request_sha256,review_path,review_sha256,native=True)
    except BaseException as exc:
        state.update(status='failed',error='binding: '+type(exc).__name__+': '+str(exc))
        try:
            try:close_trace()
            finally:write_entry(state)
        finally:
            if app is not None:app.close()
        raise
    env=collector=before=None
    original_error=None
    try:
        phase('BEFORE_ENVIRONMENT_INITIALIZATION')
        env=env_factory()
        phase('AFTER_ENVIRONMENT_INITIALIZATION')
        # Measure BEFORE camera factory as well: factory failure retains evidence.
        before=read_measurements(env)
        collector=collector_factory(env)
        phase('BEFORE_SINGLE_CASE_CAMERA_DIAGNOSTIC')
        state['diagnostic']=collector.run()
        phase('AFTER_SINGLE_CASE_CAMERA_DIAGNOSTIC')
        state['status']='diagnostic_complete_only'
    except BaseException as exc:
        original_error=exc
        state.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        try:
            if before is not None:
                after_box={}
                def get_after():
                    after_box['value']=read_measurements(env)
                    return after_box['value']
                state['measurement_boundary']=lifecycle.record_measurement_boundary(before,get_after,save_before,save_after)
                verify_measurements(before,after_box['value'])
                state['measurement_bytes_verified']=True
            else:
                require(False,'before measurement unavailable')
        except BaseException as exc:
            state.update(status='failed',measurement_error=type(exc).__name__+': '+str(exc))
            if original_error is None:original_error=exc
        try:
            if collector is not None:
                state['camera_closure']=collector.close()
                require(state['camera_closure']['status']=='closed','camera overlay cleanup failed')
        except BaseException as exc:
            state.setdefault('cleanup_errors',[]).append('camera: '+type(exc).__name__+': '+str(exc))
            state['status']='failed'
        try:close_trace()
        except BaseException as exc:
            state.setdefault('cleanup_errors',[]).append('trace: '+type(exc).__name__+': '+str(exc))
            state['status']='failed'
        # This exact parent V2 helper wraps writer failure in an app-close finally.
        lifecycle.close_environment_and_app(state,env,app,write_entry,write_app_error)
    if original_error is not None:raise original_error
    return state
