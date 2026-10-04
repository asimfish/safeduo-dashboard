# Camera pose metadata failure and corrective replay

v3 records direct native PhysX state, but every inspected Camera.data.pos_w
remained the zero initial position although the real RGB view changed.
`camera_pose_rejection_v3.json` preserves the rejected metadata evidence. v3 is
not an accepted actual-camera-pose calibration.

The effective Python package resolves to
`/home/liyufeng/safeduo_isaaclab/source/isaaclab/isaaclab/__init__.py`.
Its camera_cfg.py defaults `update_latest_camera_pose` to false (line77), and
camera.py only runs `_update_poses` during image update when that option is true
(line500). The similarly named `/home/liyufeng/IsaacLab` checkout examined first
is not the effective runtime package. No SDK or production source is changed.

The separately registered v4 replay reads the actual USD camera prim world
transform and its authored optical parameters after each render, preserving the
stale SDK readback in distinct fields. It checks actual position against the
intended eye and projects the selected arm group sphere centers using the actual
matrix and intrinsic calibration. A failed view aborts rather than producing a
passed receipt. It also keeps v3's direct all64 native PhysX readback invariance
checks. This establishes camera center framing, not absence of occlusion or
continuous-time physical collision safety.

Each replay uses the same development input, and contributes no new random
holdout tape. v1 startup failure, v2 cached-state evidence and v3 rejected camera
metadata remain independently preserved.
