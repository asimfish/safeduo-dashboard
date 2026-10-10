"""CPU reader; explicit actual closure binding required, never polls native data."""
from evidence_io import cpu_limits
cpu_limits()
import argparse
import traceback
from pathlib import Path
from evidence_io import H,OUTPUT,Evidence,write_new,contained
from single_binding import closure,PendingRelease
from single_state import prepare_state
from single_camera import audit_camera


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--binding',required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--release-review',required=True);p.add_argument('--release-review-sha256',required=True)
    args=p.parse_args();out=contained(args.output,OUTPUT);e=Evidence()
    report=dict(schema='astra.single_case.independent_raw_diagnostic.v1',status='FAILED_INTEGRITY',
        native_pass=False,stage_pass=False,safety_acceptance=False,full32_qualified=False,prefix_qualified=False)
    try:
        context=closure(e,args.binding,args.binding_sha256,args.release_review,args.release_review_sha256);report['closure']=context['closure']
        report['streamed_receipt']=context['streamed_receipt']
        audit,reference,state=prepare_state(e,context);report['state']=state
        asset=e.js(H/'astra/next_mechanism/CAMERA_PALM_USD_IDENTITIES_V1.json');e.verify(asset['source_files_sha256'])
        result=audit_camera(audit,context['receipt'],reference,asset,out.parent);report['camera']=result
        if result['independent_observation_sufficient']:report['status']='CLOSED_SINGLE_CASE_RAW_OBSERVATION_ONLY'
    except PendingRelease as exc:report.update(status='HOLD_RELEASE_REVIEW',error=str(exc))
    except Exception as exc:report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    try:e.recheck()
    except Exception as exc:report.update(status='FAILED_INPUT_MUTATION',closure_error=str(exc))
    write_new(out.parent/'EVIDENCE_LEDGER.json',e.ledger);write_new(out,report)
    return 0 if report['status']=='CLOSED_SINGLE_CASE_RAW_OBSERVATION_ONLY' else 3 if report['status']=='HOLD_RELEASE_REVIEW' else 2


if __name__=='__main__':raise SystemExit(main())
