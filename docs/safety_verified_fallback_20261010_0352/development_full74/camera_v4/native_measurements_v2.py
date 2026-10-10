"""Original native readers/config; sole env change is the reviewed camera import."""
import os,sys
import numpy as np
from repair_common_v2 import require,ARMS,N,array,frozen_import_paths

def create_environment_in_existing_app(device):
    """Call only AFTER the parent's single AppLauncher. No launch in this helper."""
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and sys.dont_write_bytecode, 'native caller must use -B and PYTHONDONTWRITEBYTECODE=1')
    frozen_import_paths()
    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from qualified_views_fullsphere_v2 import qualified_camera_factory, configure_qualified_camera
    cfg = make_duo_env_cfg(num_envs=N, device=device, yaml_name='duo_env_a31_pending_guard.yaml', coordinator=True, arm_aware_obs=True, p2_obs=True)
    cfg.coordinator['terminate_on_violation'] = False
    cfg.episode_length_s, cfg.seed, cfg.enable_viz_camera = 21., 0, True

    class VisibleDuoEnv(DuoEnv):
        def _setup_scene(self):
            with qualified_camera_factory():
                super()._setup_scene()
            configure_qualified_camera(self)

    env = VisibleDuoEnv(cfg)
    require(env.scene.sensors.pop('viz_cam') is env._viz_cam, 'original shared camera identity')
    env._viz_cam.reset()
    return env

def native_parameter_readback(env):
    """Persist getter bytes for the independent reader, beyond runtime asserts."""
    getters = {'stiffness': 'get_dof_stiffnesses', 'damping': 'get_dof_dampings',
        'max_force': 'get_dof_max_forces', 'max_velocity': 'get_dof_max_velocities',
        'armature': 'get_dof_armatures', 'hard_limits': 'get_dof_limits',
        'friction': 'get_dof_friction_properties', 'drive_model': 'get_dof_drive_model_properties',
        'masses': 'get_masses', 'inertias': 'get_inertias'}
    result = {}
    for arm in ARMS:
        art, idx = env._arms[arm], env._joint_idx[arm]
        result[arm + '_native_joint_names'] = np.asarray(art.joint_names)
        result[arm + '_controlled_joint_indices'] = array(idx)
        result[arm + '_soft_limits'] = array(art.data.soft_joint_pos_limits[:, idx])
        for suffix, getter in getters.items():
            result[arm + '_' + suffix] = array(getattr(art.root_physx_view, getter)())
        for suffix, value in [('disable_gravity_cfg', art.cfg.spawn.rigid_props.disable_gravity),
            ('solver_position_iterations_cfg', art.cfg.spawn.articulation_props.solver_position_iteration_count),
            ('solver_velocity_iterations_cfg', art.cfg.spawn.articulation_props.solver_velocity_iteration_count)]:
            result[arm + '_' + suffix] = np.full(32, value, dtype=bool if suffix == 'disable_gravity_cfg' else np.int64)
    return result
