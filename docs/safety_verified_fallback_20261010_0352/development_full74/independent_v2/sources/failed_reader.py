"""Closed FAILED V9 initial-camera diagnostic. Parent alone authorizes a later launch."""
import argparse
from pathlib import Path
import traceback
from evidence_io import HERE,OUTPUT,Evidence,cpu_limits,contained,write_new,sha
from failed_closure import terminal_gate,PendingClosure


def main():
    cpu_limits()
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('attempt-root','parent-wait-sha256','request-sha256','outer-execution-sha256','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();output=contained(a.output,OUTPUT);e=Evidence()
    report=dict(schema='astra.full74.failed_initial_camera_diagnostic.v1',status='FAILED_DIAGNOSTIC_INTEGRITY',
        purpose='DEVELOPMENT_INITIAL_RAW_DIAGNOSTIC_ONLY',attempt_root=a.attempt_root,
        native_pass=False,stage_pass=False,safety_acceptance=False,all_requirements_completed=False,
        prefix12_completed=False,final_camera_present=False,selected_count=0,formal_holdout_count=0,
        UNKNOWN=['CONTACT_FRESHNESS','ACTUAL_IMPLICIT_DRIVE_TORQUE','TERMINAL_SET','GLOBAL_ERROR_BOUND',
                 'UNOBSERVED_PREFIX_AND_FINAL','UNRECORDED_NATIVE_PARAMETER_READBACKS','UNRECORDED_NATIVE_GEOMETRY_IDENTITY'])
    code=2
    try:
        anchors=e.js(HERE/'anchors.json')
        w,outer,receipts,closure=terminal_gate(e,anchors,Path(a.attempt_root),a.parent_wait_sha256,a.request_sha256,a.outer_execution_sha256)
        report['actual_process_closure']=closure
        # Import numerical kernels only after applying CPU resource limits.
        from initial_reader import read_initial
        from failed_camera import audit_all
        initial=read_initial(e,w,receipts,Path(a.attempt_root),anchors);report['initial']=initial['report']
        report['camera']=audit_all(e,initial,output.parent)
        report['parent_observation_not_oracle']=anchors['reported_initial_metadata']
        if report['camera']['complete_raw_diagnostic']:
            report['status']='CLOSED_FAILED_INITIAL_CAMERA_DIAGNOSTIC_ONLY';code=0
    except PendingClosure as exc:report.update(status='PENDING_REAL_TERMINAL_CLOSURE',error=str(exc));code=3
    except Exception as exc:report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    try:e.recheck()
    except Exception as exc:report.update(status='FAILED_EVIDENCE_CHANGED',error=str(exc));code=2
    ledger=output.parent/'EVIDENCE_LEDGER.json';write_new(ledger,e.ledger)
    report['evidence_ledger']=str(ledger);report['evidence_ledger_sha256']=sha(ledger)
    write_new(output,report);print(report['status'],flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
