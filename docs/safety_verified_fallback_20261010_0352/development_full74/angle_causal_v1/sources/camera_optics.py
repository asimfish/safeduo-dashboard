"""Independent optical projection, extracted with source provenance in anchors.json."""
import numpy as np
from evidence_io import require
WIDTH, HEIGHT = 1280, 720

def frustum(points, camera):
    world = np.asarray(camera['actual_camera_to_world_row_matrix'], dtype=np.float64)
    k = np.asarray(camera['actual_intrinsic_matrix'], dtype=np.float64)
    near, far = camera['actual_clipping_range_m']
    require(world.shape == (4, 4) and k.shape == (3, 3) and np.isfinite(world).all()
            and np.isfinite(k).all() and 0 < near < far, 'invalid camera matrix/clipping')
    require(np.allclose(world[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0) and
            np.allclose(world[:3, :3] @ world[:3, :3].T, np.eye(3), atol=1e-6, rtol=0),
            'camera row transform is not rigid')
    require(np.linalg.det(world[:3, :3]) > 0, 'camera transform reflected')
    optics = camera['actual_usd_optics']
    require(optics['horizontal_aperture_offset'] == 0 and optics['vertical_aperture_offset'] == 0,
            'unsupported non-centered USD optics')
    expected = np.array([[WIDTH*optics['focal_length']/optics['horizontal_aperture'], 0, WIDTH/2],
                         [0, HEIGHT*optics['focal_length']/optics['vertical_aperture'], HEIGHT/2], [0, 0, 1]])
    require(np.allclose(k, expected, atol=1e-10, rtol=1e-12), 'K does not reconstruct from USD optics')
    require(np.allclose(k, camera['sdk_intrinsic_matrix'], atol=2e-4, rtol=1e-6), 'SDK K mismatch')
    require(camera['image_size'] == [WIDTH, HEIGHT], 'camera image size mismatch')
    # Independently transform native centers to camera space and evaluate each
    # inward plane as a normalized signed distance, subtracting the .06m radius.
    local = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(world)
    x, y, z = local[:, :3].T
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    margins = np.column_stack([(fx*x-cx*z)/np.hypot(fx, cx),
        (-fx*x-(WIDTH-cx)*z)/np.hypot(fx, WIDTH-cx),
        (-fy*y-cy*z)/np.hypot(fy, cy), (fy*y-(HEIGHT-cy)*z)/np.hypot(fy, HEIGHT-cy),
        -z-near, z+far]) - .06
    center = (points.min(0) + points.max(0)) / 2
    direction = world[3, :3] - center
    require(np.linalg.norm(direction[:2]) > 1e-9, 'azimuth undefined')
    return dict(pass_gate=bool(np.all(margins >= .02)), native_hand_links=len(points),
                minimum_plane_margin_m=float(margins.min()),
                minimum_by_plane_m=margins.min(0).tolist(), radius_m=.06, clearance_m=.02,
                azimuth_deg=float(np.degrees(np.arctan2(direction[1], direction[0])) % 360),
                scope='native link sphere framing only; no mesh visibility or physical safety proof')
