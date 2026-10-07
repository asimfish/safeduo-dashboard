"""Next-run observer for complete raw native state; runtime is unvalidated here."""
import numpy as np
ARM_KEYS=('F_L','F_R','U_L','U_R')


def array(value):
    # Copy twice across the device/NumPy seam; PhysX getter storage may be live.
    if hasattr(value,'detach'):value=value.detach().clone().cpu().numpy()
    result=np.array(value,copy=True)
    if not np.isfinite(result).all():raise ValueError('nonfinite native snapshot')
    return result


def capture_native(env, frame, boundary):
    if boundary not in ['pre_physics','post_physics_pre_render']:raise ValueError('boundary')
    queue=env._evaluation_actuator_delay.queue.pending
    if len(queue)!=6:raise ValueError('exact actual FIFO6 required')
    result=dict(frame=np.array(int(frame)),boundary=np.array(boundary))
    for arm in ARM_KEYS:
        view=env._arms[arm].root_physx_view
        for name,getter in [('native_q',view.get_dof_positions),('native_qd',view.get_dof_velocities),
                            ('native_root_xyzw',view.get_root_transforms),('native_root_velocity',view.get_root_velocities)]:
            result[arm+'_'+name]=array(getter())
        result[arm+'_controlled_joint_indices']=array(env._joint_idx[arm])
        result[arm+'_issued_controlled_target']=array(env._targets[arm])
        result[arm+'_pending_controlled_targets']=np.stack([array(q[arm]) for q in queue])
    return result
