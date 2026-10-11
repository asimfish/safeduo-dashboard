"""Preregistered three identical gravity fixtures; reporting APIs are the only factor."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

H=Path(__file__).resolve().parent
P=H/'cpu_contact_calibration_v3'

def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True
    out=Path(args.out);out.mkdir(exist_ok=False)
    reg=json.loads((P/'REGISTRATION_V1.json').read_text())
    for b in reg['bindings']:
        assert hashlib.sha256(Path(b['destination']).read_bytes()).hexdigest()==b['source_sha256']
    app=AppLauncher(args).app
    try:
        import isaaclab.sim as sim_utils
        import omni.physics.tensors as tensors
        import omni.usd
        import carb.settings
        from pxr import Usd,UsdPhysics,PhysxSchema
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=reg['dt_s'],device='cpu'))
        material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.6)
        support=sim_utils.CuboidCfg(size=(4.,4.,.1),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=material)
        support.func('/World/Support',support,translation=(0.,0.,-.05))
        cfg=sim_utils.CuboidCfg(size=(.04,.7,.08),rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.45),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=material)
        paths=['/World/WithoutReportAPI','/World/DefaultReportAPI','/World/ZeroThresholdReportAPI']
        stage=omni.usd.get_context().get_stage()
        for i,path in enumerate(paths):cfg.func(path,cfg,translation=((i-1)*.5,0.,.061))
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(paths[1]))
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(paths[2])).CreateThresholdAttr().Set(0.)
        def metadata():
            return [dict(path=path,report_api=stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxContactReportAPI),
                threshold=stage.GetPrimAtPath(path).GetAttribute('physxContactReport:threshold').Get()) for path in paths]
        before=metadata()
        filters=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Support') and p.HasAPI(UsdPhysics.CollisionAPI)]
        assert filters
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
        assert carb.settings.get_settings().get_as_bool('/physics/disableContactProcessing') is False
        sim.reset()
        view=tensors.create_simulation_view('torch');view.set_subspace_roots('/')
        assert view.device_ordinal==-1
        bodies=view.create_rigid_body_view(paths)
        # An absent reporting API cannot bind a CPU contact view. Keep the negative fixture as an unbound, explicit case.
        contact_paths=paths[1:]
        contact=view.create_rigid_contact_view(contact_paths,filter_patterns=[filters]*len(contact_paths),max_contact_data_count=256)
        assert contact.check() and list(contact.sensor_paths)==contact_paths
        after=metadata();records=[]
        for step in range(reg['physics_steps']):
            disabled=step<reg['processing_switch_step']
            carb.settings.get_settings().set_bool('/physics/disableContactProcessing',disabled)
            assert carb.settings.get_settings().get_as_bool('/physics/disableContactProcessing')==disabled
            sim.step(render=step%60==0)
            raw=contact.get_contact_data(reg['dt_s'])
            row=dict(step=step,contact_processing_disabled=np.asarray(disabled),transforms=bodies.get_transforms().detach().clone().cpu().numpy(),
                velocities=bodies.get_velocities().detach().clone().cpu().numpy(),
                net=contact.get_net_contact_forces(reg['dt_s']).detach().clone().cpu().numpy(),
                matrix=contact.get_contact_force_matrix(reg['dt_s']).detach().clone().cpu().numpy(),
                counts=raw[-2].detach().clone().cpu().numpy())
            assert np.isfinite(row['transforms']).all() and np.isfinite(row['matrix']).all()
            records.append(row)
        arrays={k:np.stack([r[k] for r in records]) for k in records[0]}
        np.savez_compressed(out/'native_fixture_stream.npz',**arrays)
        masses=bodies.get_masses().detach().clone().cpu().numpy()
        assert np.allclose(masses,.45,rtol=0,atol=1e-7)
        stage.GetRootLayer().Export(str(out/'actual_stage.usda'))
        save(out/'protocol.json',dict(status='NATIVE_DATA_COMPLETE_CLEANUP_PENDING',physics_steps=len(records),
            physics_device='cpu',before_view_reporting_metadata=before,after_view_reporting_metadata=after,
            missing_api_fixture_unbound=True,processing_switch_step=reg['processing_switch_step'],sensors=list(contact.sensor_paths),filters=[list(v) for v in contact.filter_paths],masses=masses.tolist(),
            independent_calibration_pending=True,formal_holdout_count=0,fullSystem0_accepted=False))
        bodies=None;contact=None;view=None;gc.collect();sim.clear_instance()
    finally:app.close(wait_for_replicator=False,skip_cleanup=False)

if __name__=='__main__':main()
