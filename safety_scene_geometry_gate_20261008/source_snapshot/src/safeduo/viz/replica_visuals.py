"""Replica visual layer for demo recordings (Agent B3, Round 119 item 2).

Purely cosmetic swap used by viz/record paths ONLY (training never imports
this): the legacy box tables stay in the physics scene (hidden), so margins,
table rows and everything the safety stack computes are bit-identical to a
plain run. What changes is what the camera sees:
  - box tables (brown Cuboids) -> visibility off, replaced by the aluminum
    -profile replica pair (assets_real/table_replica, gap=0 per Round 115
    adjudication) + white racetrack / butterfly pedestals;
  - robot spawn USDs -> *_display.usda siblings when they exist (deep-red
    JAKA with silver barrel/tubes, black F2 hands, silver flange spacer).
    v3-era stock assets (nucleus fr3/ur5e) have no display sibling and pass
    through untouched, so legacy checkpoints keep their exact visuals.

Known cosmetic caveat (documented in STATUS_B): the replica pair is the
real-machine gap=0 geometry while v3/v4 PHYSICS tables keep their frozen
layout (v4: two 0.8x1.2 boards, gap 0.5, outer edge x=+-1.05). Anything the
physics lets into the gap region will visually overlap the replica tabletop.
Do not "fix" this by moving physics tables -- that is the v5 package, gated
by supervision.
"""

from __future__ import annotations

from pathlib import Path

REPLICA_TABLE = "assets_real/table_replica/usd/table_replica_pair.usda"
PED_FR = "assets_real/table_replica/usd/pedestal_franka_visual.usda"
PED_JK = "assets_real/table_replica/usd/pedestal_jaka_visual.usda"
REPLICA_TABLE_TOP = 0.80  # pair.usda is authored for a 0.80m tabletop


def remap_display_usd(cfg) -> int:
    """Swap robot spawn USDs to *_display siblings when available.

    Call BEFORE DuoEnv(cfg). Display files sublayer the originals, so the
    physics composition is identical; only materials/subsets differ.
    """
    n = 0
    for _arm, rc in cfg.robot_cfgs.items():
        p = getattr(rc.spawn, "usd_path", "")
        if p and "://" not in p and p.startswith("/"):
            pp = Path(p)
            cand = pp.with_name(pp.stem + "_display.usda")
            if cand.is_file():
                rc.spawn.usd_path = str(cand)
                n += 1
    return n


def spawn_replica_visuals(env) -> bool:
    """Hide box-table visuals and spawn replica table + pedestals per env.

    Call AFTER DuoEnv(cfg). Visual-only: the box tables keep colliding
    invisibly; replica table collision shapes are explicitly disabled.
    """
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdGeom

    from safeduo.configs import repo_root

    root = repo_root()
    table_usd = root / REPLICA_TABLE
    if not table_usd.is_file():
        print(f"[replica_visuals] missing {table_usd}, keeping box tables", flush=True)
        return False
    stage = omni.usd.get_context().get_stage()
    tb = env.cfg.table_board
    top_z = tb["centers"][0][2] + tb["half"][2]
    z_off = top_z - REPLICA_TABLE_TOP
    # The replica USD authors colliders. Keep the overlay visual-only even
    # when the referenced asset contains PhysicsCollisionAPI shapes.
    tbl = sim_utils.UsdFileCfg(
        usd_path=str(table_usd),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    peds = {"F": sim_utils.UsdFileCfg(usd_path=str(root / PED_FR)),
            "U": sim_utils.UsdFileCfg(usd_path=str(root / PED_JK))}
    for e in range(env.num_envs):
        base = f"/World/envs/env_{e}"
        for tname in ("TableF", "TableU"):
            prim = stage.GetPrimAtPath(f"{base}/{tname}")
            if prim and prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        tbl.func(f"{base}/ReplicaTables", tbl, translation=(0.0, 0.0, z_off))
        for arm, (pos, _rot) in env.cfg.base_poses.items():
            peds[arm[0]].func(f"{base}/ReplicaPed_{arm}", peds[arm[0]],
                              translation=(pos[0], pos[1], top_z))
    print(f"[replica_visuals] replica table+pedestals spawned for {env.num_envs} env(s) "
          f"(top_z={top_z:.3f}, z_off={z_off:+.3f}); box tables hidden (visual only)",
          flush=True)
    return True
