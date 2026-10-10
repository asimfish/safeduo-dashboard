"""Project proposal coordinates through source FK; never claim safe reachability."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from four_hand_geometry_v1 import FourHandGeometry

H=Path(__file__).resolve().parent
OUT=H/'full_motor_workspace_projection_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    domain=H/'full_motor_operating_domain_v1'
    assert read(domain/'INDEPENDENT_PROPOSAL_ORACLE_ACTUAL_EXIT_V1.json')['actual_outer_exit']==0
    g=FourHandGeometry()
    baseline=H/'fourhand_native_probe_v31'
    reg=read(baseline/'REGISTRATION_V1.json')
    raw=Path(read(baseline/'RESOURCE_EXECUTION_V1.json')['out'])
    params=np.load(raw/'native_parameters.npz',allow_pickle=False)
    data=np.load(domain/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz',allow_pickle=False)
    sources=dict(g.sources);rows=[];arrays={}

    def poses(arm,q):
        names=params[arm+'_joint_names'].tolist()
        if arm.startswith('F'):
            ci=params[arm+'_controlled_indices']
            state=g.provider.kin['F'].fk(torch.tensor(q[:,ci],dtype=torch.float32),g.base[arm][:3,3].tolist(),g.yaw[arm])
            flange=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
            flange[:,:3,3]=state['t_flange'].numpy();flange[:,:3,:3]=state['R_flange'].numpy()
            return g.models[arm].fk(names,q,flange)
        return g.models[arm].fk(names,q,g.base[arm])

    # Revalidate the retained source model on every actual native31 event,
    # rather than silently applying an old successful configuration.
    errors={arm:0. for arm in g.arms};checked={arm:0 for arm in g.arms}
    owners={arm:sorted({row['body'] for row in g.records[arm]}) for arm in g.arms}
    for chunk in read(raw/'physics_chunks.json'):
        file=raw/chunk['path'];assert sha(file)==chunk['sha256'];sources[str(file)]=sha(file)
        with np.load(file,allow_pickle=False) as z:
            for arm in g.arms:
                native_names=params[arm+'_body_names'].tolist()
                q=z[arm+'_native_q'][:,0]
                predicted=poses(arm,q)
                for owner in owners[arm]:
                    actual=z[arm+'_body_position_local'][:,0,native_names.index(owner)]
                    error=float(abs(predicted[owner][:,:3,3]-actual).max())
                    errors[arm]=max(errors[arm],error);checked[arm]+=len(q)
    assert max(errors.values())<=1e-5,errors
    for arm in g.arms:
        q=data[arm+'_q']
        names=params[arm+'_joint_names'].tolist()
        assert names==g.params[arm+'_joint_names'].tolist()
        # Select actual native rigid hand mounts, never calibration metadata.
        body=('left_hand_base' if arm.endswith('L') else 'right_hand_base') if arm.startswith('F') else 'wrist_3_link'
        assert body in owners[arm]
        points=[]
        for start in range(0,len(q),512):points.append(poses(arm,q[start:start+512])[body][:,:3,3])
        positions=np.concatenate(points)
        assert positions.shape==(20000,3) and np.isfinite(positions).all()
        bins=np.floor(positions/.05).astype(np.int64)
        unique=np.unique(bins,axis=0)
        arrays[arm+'_source_hand_mount_position_m']=positions
        arrays[arm+'_occupied_5cm_voxels']=unique
        rows.append(dict(arm=arm,source_hand_mount=body,proposals=20000,
                         world_component_min_m=positions.min(axis=0).tolist(),world_component_max_m=positions.max(axis=0).tolist(),
                         occupied_5cm_source_position_voxels=len(unique),native31_hand_body_position_binding_max_m=errors[arm],
                         native31_owner_event_checks=checked[arm]))
    sets=[{tuple(row) for row in arrays[arm+'_occupied_5cm_voxels']} for arm in g.arms]
    overlap=set.intersection(*sets)
    arrays['all4_sampled_position_voxel_overlap']=np.asarray(sorted(overlap),dtype=np.int64).reshape(-1,3)
    OUT.mkdir();np.savez_compressed(OUT/'FULL50_SOURCE_WORKSPACE_PROJECTION_V2.npz',**arrays)
    sources.update({str(file):sha(file) for file in [Path(__file__),raw/'native_parameters.npz',domain/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz',domain/'INDEPENDENT_PROPOSAL_ORACLE_V1.json']})
    report=dict(status='PASS_FINITE_SOURCE_HAND_FRAME_PROJECTIONS_WITH_NATIVE31_POSITION_REBINDING',
                rows=rows,sampled_all4_position_voxel_overlap_count=len(overlap),source_SHA256=sources,
                position_voxel_overlap_is_not_a_simultaneous_collision_free_grasp=True,
                native31_binding_does_not_validate_unseen_proposals=True,
                original24_native_velocity_relation_gate_remains_failed=True,
                orientation_manipulability_object_grasp_geometry_and_native_contacts_not_scored=True,
                proposals_native_executed=0,formal_holdout_count=0,fullSystem0_accepted=False)
    (OUT/'WORKSPACE_PROJECTION_V2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],errors,'sampled_overlap',len(overlap),flush=True)


if __name__=='__main__':
    main()
