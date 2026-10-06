"""Read native arm limits and object/table geometry; not a task trial."""
import argparse,json,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--env_yaml',required=True);AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);app=AppLauncher(a).app
import isaaclab.sim as sim_utils
from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
from safeduo.safety.types import ARM_KEYS
import omni.usd
from pxr import UsdGeom,Usd

def main():
    cfg=make_duo_env_cfg(num_envs=1,device=a.device,yaml_name=a.env_yaml,coordinator=True);cfg.enable_viz_camera=False
    env=DuoEnv(cfg);result=dict(status='PASS_NATIVE_METADATA_READBACK',scope='configuration/geometry readback only, zero task trials',arms={})
    for arm in ARM_KEYS:
        art=env._arms[arm];ids=env._joint_idx[arm]
        result['arms'][arm]=dict(joint_names=[art.joint_names[i] for i in ids],hard_limits_rad=art.data.joint_pos_limits[0,ids].cpu().tolist(),soft_limits_rad=art.data.soft_joint_pos_limits[0,ids].cpu().tolist(),
            native_dof_limits_rad=art.root_physx_view.get_dof_limits()[0,ids].cpu().tolist(),q0=art.data.joint_pos[0,ids].cpu().tolist(),usd_path=cfg.robot_cfgs[arm].spawn.usd_path)
    result['native_control_dt_s']=env.step_dt;result['native_physics_dt_s']=env.cfg.sim.dt
    stage=omni.usd.get_context().get_stage();bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
    result['tables']={t:dict(min_m=list(bbox.ComputeWorldBound(stage.GetPrimAtPath(f'/World/envs/env_0/{t}/geometry/mesh')).ComputeAlignedRange().GetMin()),max_m=list(bbox.ComputeWorldBound(stage.GetPrimAtPath(f'/World/envs/env_0/{t}/geometry/mesh')).ComputeAlignedRange().GetMax())) for t in ('TableF','TableU')}
    (a.out/'limits.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);env.close()
try:main()
except BaseException:
    import traceback
    (a.out/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True);raise
finally:
    sim_utils.SimulationContext.clear_instance();app.close()
