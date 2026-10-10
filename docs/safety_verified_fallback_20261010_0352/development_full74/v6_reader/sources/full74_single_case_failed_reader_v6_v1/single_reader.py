"""Read failed V6 raw products after outer reap; never invoke the complete gate."""
from evidence_io import cpu_limits
cpu_limits()
import argparse,collections,itertools,traceback
from pathlib import Path
import numpy as np
from evidence_io import H,OUTPUT,Evidence,write_new,contained,require,parse_json
from failed_binding import load,recheck_closed
from failed_state import prepare
from single_camera import audit_camera
from actual_angle_oracle import measured,circular_error


def diagnostic_envelope():
    return dict(schema='astra.single_case.failed_raw_diagnostic.v6.v1',status='FAILED_DIAGNOSTIC',audit_completed=False,
        native_pass=False,data_pass=False,qualification=False,stage_pass=False,safety_acceptance=False,
        full32_qualified=False,prefix_qualified=False,actual_CUDA_identity='UNKNOWN',actual_Kit_identity='UNKNOWN',native_actual_exit='UNKNOWN_MISSING_NATIVE_WAIT')


def extra_inventory(e,audit,receipt,camera,out):
    records=[];angles=[];refs={};extensions=collections.Counter()
    def collect(v):
        if isinstance(v,dict):
            if set(v)=={'path','sha256'}:
                path=contained(audit.root/v['path'],audit.root);old=refs.get(str(path));require(old is None or old==v['sha256'],'conflicting raw reference');refs[str(path)]=v['sha256']
            for x in v.values():collect(x)
        elif isinstance(v,list):
            for x in v:collect(x)
    collect(receipt)
    for ref in receipt['attempts']:
        rec=parse_json(audit.ref(ref));collect(rec)
        points=np.asarray(rec['hand_framing_points_world_m'],float);raw=measured(rec['camera'],points)
        delta=circular_error(raw['actual_azimuth_deg'],rec['actual_azimuth_deg'])
        angles.append(dict(attempt=rec['attempt'],actual_azimuth_deg=raw['actual_azimuth_deg'],declared_actual_azimuth_deg=rec['actual_azimuth_deg'],
            requested_azimuth_deg=rec['requested_azimuth_deg'],actual_minus_declared_deg=delta,horizontal_baseline_m=raw['horizontal_baseline_m'],condition=raw['condition'],
            identity_within_unchanged_1e_4=abs(delta)<1e-4))
        records.append(rec)
    for path,digest in refs.items():e.hash(path,digest);extensions[Path(path).suffix]+=1
    folder=audit.root/'camera_repair_env002_F_R';raw_files={str(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.npz','.png')}
    declared={p for p in refs if Path(p).suffix in ('.npz','.png')}
    require(raw_files==declared,'raw NPZ/PNG file inventory matches referenced packet exactly')
    byid={v['attempt']:v for v in angles};chosen=[byid[n] for n in receipt['selected_attempts']]
    distances=[dict(attempts=[a['attempt'],b['attempt']],separation_deg=abs(circular_error(a['actual_azimuth_deg'],b['actual_azimuth_deg']))) for a,b in itertools.combinations(chosen,2)]
    counts=[r['counts'] for r in camera['attempts'] if r['counts'] is not None]
    require(len(counts)==96,'all96 original masks independently decoded')
    body_paths=list(camera['strict_group']['per_native_body_peak_pixels'])
    peaks={p:max(c['per_native_object_pixels'][p] for c in counts) for p in body_paths}
    families={f:max(c['family_by_arm']['F_R'][f] for c in counts) for f in ('thumb','index','middle','ring','little')}
    all96=dict(scope='EXPLORATORY_ALL_RECORDED_96_PER_VIEW_MAXIMA_NO_RESELECTION',palm_peak=max(c['palm_by_arm']['F_R'] for c in counts),
        finger_family_peaks=families,per_native_body_peaks=peaks,body_gaps_below64=[p for p,n in peaks.items() if n<64],
        family_gaps_below512=[f for f,n in families.items() if n<512],qualification=False,universal_visibility_claim=False)
    write_new(out/'RAW_FILE_INDEX.json',refs);write_new(out/'ACTUAL_ANGLE_RECOMPUTATION.json',angles)
    return dict(referenced_file_counts=dict(extensions),original_PNGs=len([p for p in refs if p.endswith('.png')]),
        original_NPZs=len([p for p in refs if p.endswith('.npz')]),matrix_records=len(angles),angles_within_1e_4=sum(a['identity_within_unchanged_1e_4'] for a in angles),
        max_actual_declared_angle_error_deg=max(abs(a['actual_minus_declared_deg']) for a in angles),max_azimuth_condition=max(a['condition'] for a in angles),
        selected_pairwise_separations=distances,selected_pairwise_all_ge30=all(v['separation_deg']>=30 for v in distances),all96_diagnostic_peaks=all96)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--binding',required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=contained(a.output,OUTPUT);e=Evidence();report=diagnostic_envelope();context=None
    try:
        context=load(e,a.binding,a.binding_sha256);report['failure_closure']=context['closure'];report['streamed_receipt']=context['stream']
        print('FAILED_OUTER_BOUND; begin raw all74/state comparisons',flush=True)
        audit,reference,state=prepare(e,context);report['raw_state']=state
        print('RAW_STATE_BOUND; begin all96 PNG/mask/native27 audit',flush=True)
        asset=e.js(H/'astra/next_mechanism/CAMERA_PALM_USD_IDENTITIES_V1.json');e.verify(asset['source_files_sha256'])
        camera=audit_camera(audit,context['receipt'],reference,asset,out.parent);report['raw_camera']=camera
        report['raw_inventory_and_angles']=extra_inventory(e,audit,context['receipt'],camera,out.parent)
        report['producer_error']=context['receipt']['error'];report['producer_status']=context['receipt']['status']
        report['audit_completed']=camera['all_attempts_visited']==camera['raw_pixels_decoded']==96
        report['diagnostic_integrity_errors']=camera['integrity_errors']
        print('RAW_96_COMPLETE; recheck all evidence hashes and absence',flush=True)
    except Exception as exc:
        report.update(reader_error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    try:
        e.recheck()
        if context is not None:recheck_closed(context)
    except Exception as exc:report.update(audit_completed=False,input_closure_error=str(exc))
    write_new(out.parent/'EVIDENCE_LEDGER.json',e.ledger);write_new(out,report)
    return 0 if report['audit_completed'] else 2
if __name__=='__main__':raise SystemExit(main())
