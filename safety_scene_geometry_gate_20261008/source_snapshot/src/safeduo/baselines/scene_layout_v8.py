"""Scene layout v8 for design-time geometry (design line, 2026-09-07).

Owner update 09-07: the Franka row moved 20 cm towards the UR5 row -> row distance 1.2964 m
(v7: 1.4964). Pair spacings, base height, tables and hands are unchanged. World frame stays the
midpoint of the two rows, so both rows sit at x = +-0.6482 (see assets_src/real/scene_layout_v8.yaml,
mirrored by the env through `scene.layout_yaml`).

This module gives the design-time side (trajectory builders, offline audits, four-gate design checks)
the same layout without touching real_geometry_v7.py: `SceneLayoutV8` + `make_v8_provider`, which is
`make_v7_provider` / `make_v7r16_provider` with the layout swapped. v7 stays the default everywhere
until the owner switches (dashboard decision D8).
"""

from __future__ import annotations

from dataclasses import dataclass

from safeduo.baselines.real_geometry_v7 import (FR3_FLANGE_V4, FR3_INIT_Q_V7, FR3_JOINTS, FR3_LINKS,
                                                UR5_DFX_FLANGE, UR5_DFX_LINKS, UR_INIT_Q_V7, ArmKinematics,
                                                RealGeometryProvider, SceneLayoutV7, real_v7_specs_for_fk,
                                                real_v7r16_specs_for_fk, ur5_dfx_joint_table, v7_semantics,
                                                v7r16_semantics)

ROW_SPACING_V7 = 1.4964
ROW_SPACING_V8 = 1.2964          # = v7 - 0.20 (owner 2026-09-07)


@dataclass
class SceneLayoutV8(SceneLayoutV7):
    """assets_src/real/scene_layout_v8.yaml: rows 1.2964 m apart, everything else = v7."""
    base_x: float = ROW_SPACING_V8 / 2.0     # 0.6482


def make_v8_provider(n_envs: int, device="cpu", top_m: int = 64, r16: bool = True) -> RealGeometryProvider:
    """Design-time provider on the v8 layout (r16 finger shell by default, as the R27 recorders use)."""
    joints, internal_yaw = ur5_dfx_joint_table()
    kin = {
        "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4, 0.0, device=device),
        "U": ArmKinematics(joints, UR5_DFX_LINKS, True, UR5_DFX_FLANGE, internal_yaw, device=device),
    }
    init_q = {"F_L": FR3_INIT_Q_V7, "F_R": FR3_INIT_Q_V7, "U_L": UR_INIT_Q_V7, "U_R": UR_INIT_Q_V7}
    return RealGeometryProvider(
        n_envs, device=device, top_m=top_m, layout=SceneLayoutV8(),
        specs=real_v7r16_specs_for_fk() if r16 else real_v7_specs_for_fk(),
        kin=kin, init_q=init_q, semantics=v7r16_semantics() if r16 else v7_semantics())
