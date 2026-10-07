"""Optional tabletop task objects for duo_env (R17 S9, 2026-08-20).

Pure-python schema helpers with NO Isaac imports, so the parsing contract is
locally testable on the Mac: duo_env.make_duo_env_cfg calls
``parse_table_objects(yaml_dict)`` and duo_env._setup_scene consumes the
normalized dicts to spawn RigidObjects.

Contract (zero-drift): assets.table_objects absent or empty list => returns
[] => duo_env spawns nothing, resets nothing -- existing scenes bit-for-bit
unchanged. Objects are demo props for the R17 "recognizable basic tasks"
videos: they are NOT in the sphere safety model and NOT in the policy
observation (both facts pinned in tests and stated in S9_object_tasks.md).

Entry schema (assets.table_objects: list of dicts):
    name              identifier, unique (prim path Obj_<name>)
    pos               [x, y, z] env-local world position (required)
    size              scalar (cube edge) or [sx, sy, sz]; default 0.05
    color             [r, g, b] 0-1; default demo red
    mass              kg; default 0.2
    static_friction / dynamic_friction    default 1.5 / 1.2 (high-friction
                      tabletop prop, stays put under incidental finger touch)

R34 (2026-09-06) mesh assets -- optional keys:
    usd               path of a sim-ready USD (RigidBody/Mass/MeshCollision
                      APIs authored in the asset); spawned with UsdFileCfg
                      instead of a cuboid. `size` then means the asset's
                      axis-aligned bounding box (grasp planner proxy), `color`
                      is ignored, `mass` overrides the asset's mass only when
                      given explicitly.
    scale             scalar or [sx, sy, sz] (default 1)
    yaw               rotation about +Z in degrees (default 0)
    catalog           name of an entry in assets.object_catalog (a yaml
                      mapping name -> {usd, size, mass, ...}); its fields fill
                      any key the entry does not set itself.
"""

from __future__ import annotations

OBJECT_DEFAULTS = {
    "size": 0.05,
    "color": (0.85, 0.20, 0.15),
    "mass": 0.2,
    "static_friction": 1.5,
    "dynamic_friction": 1.2,
}


def parse_table_objects(y: dict) -> list:
    """Normalize assets.table_objects from a duo_env yaml dict.

    Returns a list of plain dicts (name/pos/size/color/mass/frictions) with
    types coerced; raises ValueError on malformed entries so a bad yaml dies
    at cfg-build time, not mid-episode on the server.
    """
    assets = y.get("assets") or {}
    entries = assets.get("table_objects") or []
    catalog = assets.get("object_catalog") or {}
    if isinstance(catalog, str):
        catalog = load_object_catalog(catalog)
    out: list = []
    seen: set = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"table_objects[{i}]: expected mapping, got {type(e)}")
        if e.get("catalog"):
            key = str(e["catalog"])
            if key not in catalog:
                raise ValueError(f"table_objects[{i}]: catalog entry {key!r} not found")
            e = {**{k: v for k, v in catalog[key].items() if k not in ("grasp",)}, **e}
        name = str(e.get("name") or f"obj{i}")
        if not name.isidentifier():
            raise ValueError(f"table_objects[{i}]: name {name!r} not an identifier")
        if name in seen:
            raise ValueError(f"table_objects[{i}]: duplicate name {name!r}")
        seen.add(name)
        pos = e.get("pos")
        if pos is None or len(pos) != 3:
            raise ValueError(f"table_objects[{i}] ({name}): pos must be [x,y,z]")
        size = e.get("size", OBJECT_DEFAULTS["size"])
        if isinstance(size, (int, float)):
            size = (float(size),) * 3
        else:
            if len(size) != 3:
                raise ValueError(f"table_objects[{i}] ({name}): size must be "
                                 "scalar or [sx,sy,sz]")
            size = tuple(float(v) for v in size)
        if min(size) <= 0.0:
            raise ValueError(f"table_objects[{i}] ({name}): non-positive size")
        color = e.get("color", OBJECT_DEFAULTS["color"])
        if len(color) != 3:
            raise ValueError(f"table_objects[{i}] ({name}): color must be [r,g,b]")
        usd = e.get("usd")
        scale = e.get("scale", 1.0)
        if isinstance(scale, (int, float)):
            scale = (float(scale),) * 3
        else:
            if len(scale) != 3:
                raise ValueError(f"table_objects[{i}] ({name}): scale must be scalar or [sx,sy,sz]")
            scale = tuple(float(v) for v in scale)
        out.append({
            "name": name,
            "pos": tuple(float(v) for v in pos),
            "size": size,
            "color": tuple(float(v) for v in color),
            # mesh assets keep their authored mass unless the entry sets one
            "mass": ((float(e["mass"]) if "mass" in e else None) if usd
                     else float(e.get("mass", OBJECT_DEFAULTS["mass"]))),
            "static_friction": float(e.get("static_friction",
                                           OBJECT_DEFAULTS["static_friction"])),
            "dynamic_friction": float(e.get("dynamic_friction",
                                            OBJECT_DEFAULTS["dynamic_friction"])),
            "usd": (str(usd) if usd else None),
            "scale": scale,
            "yaw": float(e.get("yaw", 0.0)),
            # R35: full roll/pitch/yaw (deg) for Y-up assets; None = yaw only
            "rpy_deg": (tuple(float(v) for v in e["rpy_deg"]) if e.get("rpy_deg") else None),
            "catalog": (str(e["catalog"]) if e.get("catalog") else None),
        })
    return out


def load_object_catalog(path_or_name: str) -> dict:
    """assets.object_catalog may name a yaml file (repo configs dir or an
    absolute path): mapping name -> {usd, size, mass, grasp: {...}}. USD paths
    that are relative are resolved against the repo root."""
    import os
    import yaml

    p = path_or_name
    if not os.path.isabs(p) and not os.path.exists(p):
        from safeduo.configs import CONFIG_DIR
        p = os.path.join(str(CONFIG_DIR), p)
    with open(p) as f:
        cat = yaml.safe_load(f) or {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))                      # .../src/safeduo/envs -> repo
    for k, v in cat.items():
        if isinstance(v, dict) and v.get("usd") and not os.path.isabs(v["usd"]):
            v["usd"] = os.path.join(root, v["usd"])
    return cat
