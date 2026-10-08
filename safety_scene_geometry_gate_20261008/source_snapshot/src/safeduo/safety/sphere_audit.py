"""Instrument-geometry audit (supervision-line P0 order, 2026-08-12).

Independently witness-loads B's sphere YAMLs via from_curobo_yaml and
cross-asserts every radius against what a live SphereDistanceModule actually
holds. On any mismatch the auditor raises: the instrument refuses to measure
with unverified geometry. The returned provenance dict (paths / sha1 /
spec_version / per-link radii) goes verbatim into the parity report, so the
"spheres" label can never be hand-written again.

Pure python+torch (no Isaac): unit-testable off-server; parity_runner calls
`sphere_provenance(env._sph)` in-kit.
"""

from __future__ import annotations

import hashlib

import yaml

_ROBOT_OF_ARM = {"F_L": "fr3", "F_R": "fr3", "U_L": "ur5e", "U_R": "ur5e"}


def witness_load(root) -> tuple[dict, dict]:
    """-> (witness radii {robot: {link: [r...]}}, provenance dict)."""
    from safeduo.safety.sphere_specs import from_curobo_yaml

    witness, prov = {}, {}
    for robot in ("fr3", "ur5e"):
        p = root / "assets_src" / "spheres" / f"{robot}.yaml"
        raw = yaml.safe_load(p.read_text())
        order = list(raw["collision_spheres"].keys())  # key order = chain order, incl fr3_link0
        spec = from_curobo_yaml(str(p), order)
        witness[robot] = {ls.link: [float(r) for r in ls.radii] for ls in spec.links}
        prov[robot] = {
            "path": str(p),
            "sha1": hashlib.sha1(p.read_bytes()).hexdigest()[:12],
            "spec_version": (raw.get("meta") or {}).get("spec_version"),
            "balls": {ls.link: [round(float(r), 4) for r in ls.radii]
                      for ls in spec.links},
        }
    return witness, prov


def witness_load_real_v3(root) -> tuple[dict, dict]:
    """v4 real-asset witness: independent re-load of fr3/jaka/F2 YAMLs plus the
    code-merged flange sphere (B directive: spheres/fr3.yaml untouched, A merges
    fr3_link8 in code -- ASSEMBLY_V3 3.2; declared here so the tripwire knows
    its sanctioned provenance instead of refusing)."""
    return _witness_load_real(root, "real_v3")


def witness_load_real_v5(root) -> tuple[dict, dict]:
    """v5 bundle witness (A9-W9): same structure as real_v3 with the three
    revised sphere YAMLs (jaka_zu7_v5 + F2 v5 hands; fr3.yaml untouched)."""
    return _witness_load_real(root, "real_v5")


_REAL_SOURCES = {
    "real_v3": (("jaka_zu7", "assets_src/real/jaka_zu7.yaml"),
                ("f2_left", "assets_src/real/inspire_rh56f2_left.yaml"),
                ("f2_right", "assets_src/real/inspire_rh56f2_right.yaml")),
    "real_v5": (("jaka_zu7", "assets_src/real/jaka_zu7_v5.yaml"),
                ("f2_left", "assets_src/real/inspire_rh56f2_left_v5.yaml"),
                ("f2_right", "assets_src/real/inspire_rh56f2_right_v5.yaml")),
}


def _witness_load_real(root, layout: str) -> tuple[dict, dict]:
    from safeduo.safety.sphere_specs import (
        _FR3_FLANGE_SPHERE,
        real_v3_arm_specs,
        real_v5_arm_specs,
    )

    loader = real_v5_arm_specs if layout == "real_v5" else real_v3_arm_specs
    specs = loader(root)
    witness = {arm: {(ls.semantic_name or ls.link): [float(r) for r in ls.radii]
                     for ls in spec.links} for arm, spec in specs.items()}
    prov: dict = {"layout": layout}
    for tag, rel in (("fr3", "assets_src/spheres/fr3.yaml"),) + _REAL_SOURCES[layout]:
        p = root / rel
        raw = yaml.safe_load(p.read_text())
        prov[tag] = {"path": str(p),
                     "sha1": hashlib.sha1(p.read_bytes()).hexdigest()[:12],
                     "spec_version": (raw.get("meta") or {}).get("spec_version")}
    prov["fr3_link8_flange"] = {
        "source": "sphere_specs._FR3_FLANGE_SPHERE (code-merged, ASSEMBLY_V3 3.2)",
        "offset": list(_FR3_FLANGE_SPHERE[1][0]),
        "radius": _FR3_FLANGE_SPHERE[2][0]}
    prov["fr3_hand_transition"] = "retired (combined USD, gripper subtree removed)"
    return witness, prov


def sphere_provenance(sph, root=None, layout: str = "v0") -> dict:
    """Cross-assert a live SphereDistanceModule against the YAML witness."""
    if root is None:
        from safeduo.configs import repo_root

        root = repo_root()
    per_arm = layout in ("real_v3", "real_v5")
    if per_arm:
        witness, prov = _witness_load_real(root, layout)
    else:
        witness, prov = witness_load(root)
    loaded: dict = {}
    for i, name in enumerate(sph.qualified_names):
        arm, link = name.split("/", 1)
        loaded.setdefault(arm, {}).setdefault(link, []).append(float(sph.radii[i]))
    for arm, links in loaded.items():
        ref = witness[arm] if per_arm else witness[_ROBOT_OF_ARM[arm]]
        for link, radii in links.items():
            # tolerance: module stores float32 (0.095 -> 0.0949999988...);
            # 1e-6 m is far below any meaningful radius edit (mm scale)
            ok = (link in ref and len(radii) == len(ref[link])
                  and all(abs(a - b) < 1e-6 for a, b in zip(radii, ref[link])))
            if not ok:
                raise RuntimeError(
                    f"PARITY_SPHERE_MISMATCH {arm}/{link}: module={radii} "
                    f"yaml={ref.get(link)} -- instrument geometry does not match "
                    f"B's YAML, refusing to measure")
    prov["n_spheres"] = int(sph.n_spheres)
    prov["n_pairs"] = int(sph.n_pairs)
    prov["cross_check"] = "module radii == independent YAML witness (asserted)"
    if not per_arm:
        prov["note"] = ("inspire_hand.yaml not consumed (combined USD pending; "
                        "fr3_hand transition balls live inside fr3.yaml)")
    return prov
