"""Read current camera transform and native geometry; never advance physics."""
import numpy as np
def rotation(q):
    w,x,y,z=np.asarray(q,dtype=float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def snapshot(cam,env,obj,view,phase,array):
    from pxr import Usd,UsdGeom
    prim=cam._sensor_prims[0].GetPrim()
    matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
    pos=array(cam.data.pos_w)[0];quat=array(cam.data.quat_w_opengl)[0]
    np.testing.assert_allclose(pos,view['eye'],atol=1e-5,rtol=0)
    np.testing.assert_allclose(pos,matrix[3,:3],atol=1e-5,rtol=0)
    np.testing.assert_allclose(rotation(quat),matrix[:3,:3].T,atol=1e-5,rtol=0)
    return dict(camera_pose_metadata_source='PublicCameraResetPoseBuffer_and_independent_USD_world_transform',
                camera_world_xform_row_major=matrix.tolist(),phase_after_control=phase,
                native_object_transform_xyzw=array(obj.root_physx_view.get_transforms())[0].tolist(),
                native_body_link_transforms_xyzw={a:array(env._arms[a].root_physx_view.get_link_transforms())[0].tolist() for a in ['F_L','F_R','U_L','U_R']})
