"""Copied native state at actual physics boundaries; never advances physics."""
import numpy as np
ARM_KEYS=('F_L','F_R','U_L','U_R')


def array(value):
    # Copy twice across the device/NumPy seam; PhysX getter storage may be live.
    if hasattr(value,'detach'):value=value.detach().clone().cpu().numpy()
    result=np.array(value,copy=True)
    if not np.isfinite(result).all():raise ValueError('nonfinite native snapshot')
    return result


def capture_native(env, frame, boundary):
    if boundary not in ['pre_control_pre_physics','pre_physics','post_physics_pre_render']:raise ValueError('boundary')
    queue=env._evaluation_actuator_delay.queue.pending
    if len(queue)!=6:raise ValueError('exact actual FIFO6 required')
    result=dict(frame=np.array(int(frame)),boundary=np.array(boundary))
    result['environment_origins']=array(env.scene.env_origins)
    for arm in ARM_KEYS:
        view=env._arms[arm].root_physx_view
        for name,getter in [('native_q',view.get_dof_positions),('native_qd',view.get_dof_velocities),
                            ('native_root_xyzw',view.get_root_transforms),('native_root_velocity',view.get_root_velocities)]:
            result[arm+'_'+name]=array(getter())
        result[arm+'_controlled_joint_indices']=array(env._joint_idx[arm])
        result[arm+'_native_joint_names']=np.asarray(env._arms[arm].joint_names)
        result[arm+'_issued_controlled_target']=array(env._targets[arm])
        result[arm+'_pending_controlled_targets']=np.stack([array(q[arm]) for q in queue])
        result[arm+'_applied_controlled_target']=array(env._evaluation_actuator_delay.applied[arm])
        result[arm+'_pending_project_targets']=np.stack([array(q[arm]) for q in env._pending_target_history.targets])
        state=env.scene_state()
        idx=result[arm+'_controlled_joint_indices'].astype(int)
        if not np.array_equal(result[arm+'_native_q'][:,idx],array(state.q[arm])):
            raise ValueError('native controlled q differs from scored scene q')
        if not np.array_equal(result[arm+'_native_qd'][:,idx],array(state.qd[arm])):
            raise ValueError('native controlled qd differs from scored scene qd')
    return result
