"""Raw state/parameter comparisons without a fabricated ENTRY measurement receipt."""
from pathlib import Path
import numpy as np
from evidence_io import ARMS,require,exact
from science import parameters_layout,NAMESPACE,SEEDS
from single_state import SingleCameraAudit
from camera_adapter import exact_snapshot
from root_restore_reader import evaluate,validate_protocol

def prepare(e,c):
    b=c['original'];req=c['request'];receipt=c['receipt'];root=c['root']
    params=e.npz(b['parameter_path']);layout=parameters_layout(params)
    expected={k for k in params if not k.endswith('_full_initial_position_target') and k!='initial_strata'}
    geom=e.js(b['geometry_identity_path']);geom.pop('scope',None);measurements={}
    for label in ('BEFORE','AFTER'):
        p=root/'entry'/f'PARAMETER_READBACK_{label}_V1.npz';g=root/'entry'/f'GEOMETRY_{label}_V1.json'
        result=dict(parameter_status='UNKNOWN_MISSING',geometry_status='UNKNOWN_MISSING',measurement_lifecycle_provenance='UNKNOWN_MISSING_ENTRY')
        if p.is_file():
            actual=e.npz(p);require(set(actual)==expected,'raw parameter field inventory')
            for k in expected:exact(actual[k],params[k],'raw parameter '+label+'/'+k)
            result.update(parameter_status='RAW_ARTIFACT_BYTES_MATCH_FROZEN',parameter_sha256=e.hash(p),parameter_fields=len(expected))
        if g.is_file():require(e.js(g)==geom,'raw geometry identity differs');result.update(geometry_status='RAW_ARTIFACT_IDENTITY_MATCHES_FROZEN',geometry_sha256=e.hash(g))
        measurements[label]=result
    manifest=e.js(req['proposals_manifest'],req['input_files'][req['proposals_manifest']]);reg=manifest['registration']
    require(reg['namespace']==NAMESPACE and reg['seeds']==SEEDS and reg['proposals']==8192 and reg['per_seed']==2048 and reg['lanes']==32 and
        reg['purpose']=='DEVELOPMENT_ONLY' and reg['selected']==reg['formal_holdout']==0 and reg['policy_outcomes_used'] is False,'fixed first32 development only')
    folder=Path(req['proposals_manifest']).parent;e.verify(manifest['files'],folder)
    allq=e.npy(folder/'q74.npy',(8192,74),'float32');allv=e.npy(folder/'qd74.npy',(8192,74),'float32')
    q=e.npy(req['requested_q74_path'],(32,74),'float32');v=e.npy(req['requested_qd74_path'],(32,74),'float32')
    exact(q,allq[:32],'first32 requested q');exact(v,allv[:32],'first32 requested qd')
    require(np.isfinite(allq).all() and np.isfinite(allv).all() and np.all(allq>=layout['bounds'][:,0]) and np.all(allq<=layout['bounds'][:,1]) and np.all(abs(allv)<=layout['vmax']),'8192 proposal bounds')
    for k in ('hard','soft','bounds','vmax','controlled','names'):exact(e.npy(folder/(k+'.npy')),layout[k],'registered '+k)
    body=e.npz(Path(b['parameter_path']).parent/'body_identity.npz');identity=e.js(b['contact_identity_path'])
    audit=SingleCameraAudit(root/'native',e,body,identity,layout);baseline=audit.snapshot(receipt['native_before'])
    reference=e.npz(req['v9_native_point0_path'],req['input_files'][req['v9_native_point0_path']]);exact_snapshot(reference,baseline)
    exact_snapshot(baseline,audit.snapshot(receipt['native_after']))
    initial=e.npz(b['initial_path']);proto=c['anchors']['root_protocol'];protocol=validate_protocol(e.js(proto['path'],proto['sha256']));roots={}
    for arm in ARMS:
        lo,hi=layout['slices'][arm]
        exact(baseline[arm+'_native_q'],q[:,lo:hi],'native initial q '+arm)
        exact(baseline[arm+'_native_qd'],v[:,lo:hi],'native initial qd '+arm)
        exact(baseline[arm+'_native_position_targets'],q[:,lo:hi],'native initial targets '+arm)
        roots[arm]=evaluate(baseline[arm+'_native_root_xyzw'],initial[arm+'_root'],baseline[arm+'_native_root_velocity'],initial[arm+'_root_velocity'],protocol)
    return audit,reference,dict(raw_q74_qd74_targets_bitwise_all32=True,raw_V9_all27_bitwise=True,root_equivalence=roots,
        raw_measurements=measurements,native_parameter_certification=False,proposals=8192,fixed_ids=list(range(32)),unused_proposals=8160,
        recorded_additional_physics_steps=0,recorded_policy_actions=0,CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN',global_error_bound='UNKNOWN')
