"""Exact all74/all32 replay and native parameter/geometry readbacks."""
from pathlib import Path
import numpy as np
from evidence_io import ARMS,SCREEN,require,exact,contained
from science import parameters_layout,NAMESPACE,SEEDS
from root_restore_reader import evaluate,validate_protocol
from camera_adapter import CameraAudit,exact_snapshot
from receipt_schema import reference


class SingleCameraAudit(CameraAudit):
    def ref(self,ref):
        reference(ref);p=contained(self.root/ref['path'],self.root)
        return self.evidence.raw(p,ref['sha256'],16*1024**2 if p.suffix=='.json' else 128*1024**2)

    def npz(self,ref):
        reference(ref);p=contained(self.root/ref['path'],self.root)
        return self.evidence.npz(p,ref['sha256'])


def measurements(e,context,params):
    root=context['root'];entry=context['entry'];b=context['original']
    boundary=entry['measurement_boundary'];require(boundary['phase']=='NATIVE_MEASUREMENT_COMPLETE_BEFORE_RENDER_OVERLAY_ENV_APP_TEARDOWN','measurement teardown order')
    expected={k for k in params if not k.endswith('_full_initial_position_target') and k!='initial_strata'}
    reference=e.js(b['geometry_identity_path']);reference.pop('scope',None);result={}
    for label in ('before','after'):
        refs=boundary[label];require(set(refs)=={'parameters','geometry'},'measurement field inventory')
        for key,filename in [('parameters',f'PARAMETER_READBACK_{label.upper()}_V1.npz'),('geometry',f'GEOMETRY_{label.upper()}_V1.json')]:
            ref=refs[key];require(ref['path']==str(root/'entry'/filename),'actual measurement location')
            e.hash(ref['path'],ref['sha256'])
        actual=e.npz(refs['parameters']['path']);require(set(actual)==expected,'actual getter field inventory')
        for key in expected:exact(actual[key],params[key],'actual parameter '+label+'/'+key)
        geom=e.js(refs['geometry']['path']);require(geom==reference,'actual9021/tip geometry identity')
        result[label]=refs
    return dict(status='ACTUAL_GETTER_BYTES_AND_GEOMETRY_IDENTITIES_MATCH_FROZEN',artifacts=result,
                implicit_drive_torque='UNKNOWN',hidden_solver_state='UNKNOWN',physical_safety_certified=False)


def prepare_state(e,context):
    req=context['request'];b=context['original'];root=context['root'];receipt=context['receipt'];a=context['anchors']
    params=e.npz(b['parameter_path']);layout=parameters_layout(params);measured=measurements(e,context,params)
    manifest=e.js(req['proposals_manifest'],req['input_files'][req['proposals_manifest']]);reg=manifest['registration']
    require(reg['namespace']==NAMESPACE and reg['seeds']==SEEDS and reg['proposals']==8192 and reg['per_seed']==2048 and
        reg['lanes']==32 and reg['purpose']=='DEVELOPMENT_ONLY' and reg['selected']==reg['formal_holdout']==0 and
        reg['policy_outcomes_used'] is False,'same first32 development cohort')
    folder=Path(req['proposals_manifest']).parent;e.verify(manifest['files'],folder)
    allq=e.npy(folder/'q74.npy',(8192,74),'float32');allv=e.npy(folder/'qd74.npy',(8192,74),'float32')
    q=e.npy(req['requested_q74_path'],(32,74),'float32');v=e.npy(req['requested_qd74_path'],(32,74),'float32')
    exact(q,allq[:32],'first32 request q74');exact(v,allv[:32],'first32 request qd74')
    require(np.isfinite(allq).all() and np.isfinite(allv).all() and np.all(allq>=layout['bounds'][:,0]) and
        np.all(allq<=layout['bounds'][:,1]) and np.all(abs(allv)<=layout['vmax']),'8192 bounded support')
    for key in ('hard','soft','bounds','vmax','controlled','names'):exact(e.npy(folder/(key+'.npy')),layout[key],'registered '+key)
    body=e.npz(Path(b['parameter_path']).parent/'body_identity.npz');identity=e.js(b['contact_identity_path'])
    audit=SingleCameraAudit(root/'native',e,body,identity,layout)
    baseline=audit.snapshot(receipt['native_before']);reference=e.npz(req['v9_native_point0_path'],req['input_files'][req['v9_native_point0_path']])
    exact_snapshot(reference,baseline)
    initial=e.npz(b['initial_path']);protocol=validate_protocol(e.js(a['root_protocol']['path'],a['root_protocol']['sha256']));roots={}
    for arm in ARMS:
        lo,hi=layout['slices'][arm]
        exact(baseline[arm+'_native_q'],q[:,lo:hi],'native full q '+arm)
        exact(baseline[arm+'_native_qd'],v[:,lo:hi],'native full qd '+arm)
        exact(baseline[arm+'_native_position_targets'],q[:,lo:hi],'native full target '+arm)
        roots[arm]=evaluate(baseline[arm+'_native_root_xyzw'],initial[arm+'_root'],baseline[arm+'_native_root_velocity'],initial[arm+'_root_velocity'],protocol)
    return audit,reference,dict(q74_qd74_targets_bitwise_all32=True,V9_all27_bitwise=True,roots=roots,
        native_measurements=measured,proposals=8192,fixed_ids=list(range(32)),unselected_remainder=8160,
        policy_actions=0,physical_steps=0,CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN',global_error_bound='UNKNOWN')
