"""v4 birth-pose re-engineering tool (A7-W7, supervision Round-114 ruling #2).

Goal: four-arm prep poses whose BIRTH margins survive +-0.02 rad per-joint
uniform jitter with >= 2 cm floors (self was ruled explicitly; cross/table
get the same treatment because the measured jitter violations actually come
from the pose-dependent pairs: F<->U hand-vs-hand 5.5 mm, U link2<->table
14.4 mm -- see STATUS_A W7).

Structural wrist-collar pairs are excluded from the searchable objective and
reported separately, because they are pose-INVARIANT (measured, this tool
--mode audit prints the proof numbers):
  - F fr3_link6<->fr3_link8: exactly constant 12.5 mm over full j6/j7 sweeps
    (flange puck rides the j7 axis) -> semantics adjacency exemption.
  - U link4<->hand_base collar: max 6.2 mm over 4096 random FULL-LIMIT
    poses (ceiling, not a pose problem); j6 spins CAN crush it (real mode),
    so the pair STAYS checked -- lifting it above 2 cm needs B's sphere
    radius surgery (A6-W6 note: clean fix = shrink link4/6 radii).

CPU-only (uses C5's make_v4_provider); runs on the Mac.

v5 (A9-W9): `--scene v5` swaps in the v5-bundle geometry -- scene_layout_v5
(gap=0 / 1.5 m tables at +-0.40), real_v5 spheres (jaka_zu7_v5 + F2 v5 hands),
default pose = duo_env_v5.yaml init_qpos (the v4.1 pose the env actually
spawns). The v5 provider is composed here from C's public real_geometry
pieces WITHOUT touching the C-owned file (A-line write-boundary discipline).

Usage:
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search --mode audit
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search --mode search
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search --mode verify \
      --fr3 "0.0,-0.569,0.0,-2.31,0.0,1.74,0.741" --jaka "0.0,1.35,-1.2,1.2,1.57,0.0"
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search \
      --scene v5 --mode verify
"""

from __future__ import annotations

import argparse
import json

import torch

from safeduo.baselines.real_geometry import (
    FR3_INIT_Q_V4,
    FR3_INIT_Q_V41,
    JAKA_INIT_Q_V4,
    JAKA_INIT_Q_V41,
    make_v4_provider,
    make_v5_provider,
)
from safeduo.safety.types import ARM_KEYS

# joint limits (rad): FR3 official datasheet; JAKA Zu7 from B's cleaned URDF
FR3_LIMITS = torch.tensor([
    (-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007),
    (-3.0421, -0.1518), (-2.8065, 2.8065), (0.5445, 4.5169),
    (-3.0159, 3.0159)])
JAKA_LIMITS = torch.tensor([
    (-6.2832, 6.2832), (-1.4835, 4.6251), (-3.0543, 3.0543),
    (-1.4835, 4.6251), (-6.2832, 6.2832), (-6.2832, 6.2832)])
# search bounds: delta from the current prep pose (wrists get more travel;
# shoulders less, to protect the table/workspace geometry B designed)
FR3_DELTA = torch.tensor([0.40, 0.35, 0.40, 0.45, 0.50, 0.50, 0.50])
JAKA_DELTA = torch.tensor([0.40, 0.30, 0.50, 0.60, 0.70, 0.70])
JITTER = 0.02
# targets (m): the ruling gate is 2 cm under jitter; nominal targets carry
# the jitter headroom so stage-A scoring can work on nominal margins alone.
# table is capped ~1.7 cm by the U link2 proximal end-sphere (r=0.109) over
# its own table -- quasi pose-invariant (j2 sweep moves it ~2 mm), so its
# target is "best reachable", not the 2 cm gate (measured, STATUS_A W7).
GATE = 0.02
NOM_TARGET = {"cross": 0.08, "self": 0.032, "table": 0.017}
COLLAR_U_MIN = 0.0048       # keep the wrist collar at its j6=0 optimum (~5.4mm)
EE_DEV_MAX = 0.40           # keep the prep intent recognizable (m, per arm)


def collar_masks(p) -> dict:
    qn = p.sph.qualified_names
    m_f = torch.zeros(p.n_pairs, dtype=torch.bool)
    m_u = torch.zeros(p.n_pairs, dtype=torch.bool)
    for k in range(p.n_pairs):
        if bool(p.pair_is_table[k]):
            continue
        a, b = qn[int(p.pair_sph_i[k])], qn[int(p.pair_sph_j[k])]
        pair = {a.split("/", 1)[1], b.split("/", 1)[1]}
        if a.split("/")[0] != b.split("/")[0]:
            continue
        if pair == {"fr3_link6", "fr3_link8"}:
            m_f[k] = True
        if any(x == "link4" for x in pair) and any("hand_base" in x for x in pair):
            m_u[k] = True
    return {"collar_F": m_f, "collar_U": m_u}


def q_batch(fr3: torch.Tensor, jaka: torch.Tensor) -> dict:
    """(N,7) FR3 pose + (N,6) JAKA pose -> per-arm q dict (mirrored pairs)."""
    return {"F_L": fr3, "F_R": fr3.clone(), "U_L": jaka, "U_R": jaka.clone()}


def class_floors(p, q: dict, exclude: torch.Tensor) -> dict:
    d_all, _, _ = p._all_margins(p.fk_all(q)["centers"])
    out = {}
    for name, cls in (("cross", 0), ("self", 1), ("table", 2)):
        mask = (p.pair_class == cls) & ~exclude
        out[name] = d_all[:, mask].amin(dim=1)
    return out


def ee_positions(p, q: dict) -> dict:
    fko = p.fk_all(q)
    ee, _ = p.ee_pose(fko)
    return ee


_PROVIDER_CACHE: dict = {}
# --scene 开关（A9-W9）：v4 = C5 make_v4_provider（历史行为逐位不变）；
# v5 = C6 make_v5_provider（v5 球 + gap=0 布局 + manifest d_min 口径）
_SCENE: dict = {"factory": make_v4_provider}


def provider_for(n: int):
    if n not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[n] = _SCENE["factory"](n)
    return _PROVIDER_CACHE[n]


def jitter_floors(p, fr3: torch.Tensor, jaka: torch.Tensor,
                  exclude: torch.Tensor, n_samples: int, seed: int = 0) -> dict:
    """Worst-case-over-jitter class floors for ONE candidate pose."""
    g = torch.Generator().manual_seed(seed)
    n = n_samples
    jf = fr3.expand(n, 7) + (torch.rand(n, 7, generator=g) * 2 - 1) * JITTER
    ju = jaka.expand(n, 6) + (torch.rand(n, 6, generator=g) * 2 - 1) * JITTER
    # each arm jitters INDEPENDENTLY (26 independent joints in the env)
    jf2 = fr3.expand(n, 7) + (torch.rand(n, 7, generator=g) * 2 - 1) * JITTER
    ju2 = jaka.expand(n, 6) + (torch.rand(n, 6, generator=g) * 2 - 1) * JITTER
    p_n = provider_for(n)
    q = {"F_L": jf, "F_R": jf2, "U_L": ju, "U_R": ju2}
    fl = class_floors(p_n, q, exclude_for(p_n))
    return {k: float(v.min()) for k, v in fl.items()}


_EXCLUDE_CACHE: dict = {}


def exclude_for(p) -> torch.Tensor:
    key = id(p)
    if key not in _EXCLUDE_CACHE:
        cm = collar_masks(p)
        _EXCLUDE_CACHE[key] = cm["collar_F"] | cm["collar_U"]
    return _EXCLUDE_CACHE[key]


def report_pose(fr3, jaka, n_jitter: int = 4096) -> dict:
    p1 = provider_for(1)
    q = q_batch(fr3.unsqueeze(0), jaka.unsqueeze(0))
    excl = exclude_for(p1)
    nom = {k: float(v[0]) for k, v in class_floors(p1, q, excl).items()}
    cm = collar_masks(p1)
    d_all, _, _ = p1._all_margins(p1.fk_all(q)["centers"])
    # A7-W7 之后 F 腕环对（fr3_link6|fr3_link8）已语义豁免出配对表 -> 空 mask
    # 守卫（当时 verify 跑在豁免落地前，此处为潜伏 bug，A9-W9 修复）
    collars = {k: (float(d_all[0, m].min()) if bool(m.any()) else None)
               for k, m in cm.items()}
    jit = jitter_floors(p1, fr3, jaka, excl, n_jitter)
    ee = {a: [round(float(x), 3) for x in v[0]]
          for a, v in ee_positions(p1, q).items()}
    return {"fr3": [round(float(x), 4) for x in fr3],
            "jaka": [round(float(x), 4) for x in jaka],
            "nominal_floors": {k: round(v, 4) for k, v in nom.items()},
            "jitter002_floors": {k: round(v, 4) for k, v in jit.items()},
            "collar_pairs_excluded": {k: (round(v, 4) if v is not None else None) for k, v in collars.items()},
            "gate_selfx_cross_table_ge_2cm": all(
                jit[k] >= GATE for k in ("cross", "self", "table")),
            "ee_pos": ee}


def tightest(p, q: dict, k: int = 10) -> list:
    d_all, _, _ = p._all_margins(p.fk_all(q)["centers"])
    qn = p.sph.qualified_names
    d0 = d_all[0]
    rows = []
    for idx in d0.argsort()[:k]:
        i = int(idx)
        si, sj = int(p.pair_sph_i[i]), int(p.pair_sph_j[i])
        tgt = (f"TABLE{int(p.pair_tab[i])}" if bool(p.pair_is_table[i])
               else qn[sj])
        rows.append((round(float(d0[i]), 4), qn[si], tgt))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "search", "verify"],
                    default="audit")
    ap.add_argument("--scene", choices=["v4", "v5"], default="v4",
                    help="v5 = B2 bundle 场景（v5 球 + gap=0 布局），A9-W9")
    ap.add_argument("--fr3", type=str, default="")
    ap.add_argument("--jaka", type=str, default="")
    ap.add_argument("--n_a", type=int, default=8192)
    ap.add_argument("--n_b", type=int, default=48)
    ap.add_argument("--jit_b", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scene == "v5":
        _SCENE["factory"] = make_v5_provider
        fr3_def, jaka_def = FR3_INIT_Q_V41, JAKA_INIT_Q_V41
    else:
        fr3_def, jaka_def = FR3_INIT_Q_V4, JAKA_INIT_Q_V4

    fr3_0 = torch.tensor(
        [float(x) for x in args.fr3.split(",")] if args.fr3 else fr3_def)
    jaka_0 = torch.tensor(
        [float(x) for x in args.jaka.split(",")] if args.jaka else jaka_def)

    if args.mode == "audit":
        p1 = provider_for(1)
        q = q_batch(fr3_0.unsqueeze(0), jaka_0.unsqueeze(0))
        print("TIGHTEST", json.dumps(tightest(p1, q), ensure_ascii=False))
        print("AUDIT " + json.dumps(report_pose(fr3_0, jaka_0)))
        return
    if args.mode == "verify":
        rep = report_pose(fr3_0, jaka_0, n_jitter=8192)
        p1 = provider_for(1)
        rep["tightest"] = tightest(
            p1, q_batch(fr3_0.unsqueeze(0), jaka_0.unsqueeze(0)))
        print("VERIFY " + json.dumps(rep))
        return

    # ---- stage A: nominal-margin screening over random deltas -------------
    torch.manual_seed(args.seed)
    n = args.n_a
    p = provider_for(n)
    excl = exclude_for(p)
    cm = collar_masks(p)
    df = (torch.rand(n, 7) * 2 - 1) * FR3_DELTA
    du = (torch.rand(n, 6) * 2 - 1) * JAKA_DELTA
    du[:, 5] = 0.0                 # pin JAKA j6: its 0.0 is the collar optimum
    df[0], du[0] = 0.0, 0.0        # keep the incumbent in the race
    fr3 = (fr3_0 + df).clamp(FR3_LIMITS[:, 0], FR3_LIMITS[:, 1])
    jaka = (jaka_0 + du).clamp(JAKA_LIMITS[:, 0], JAKA_LIMITS[:, 1])
    q = q_batch(fr3, jaka)
    fl = class_floors(p, q, excl)
    d_all, _, _ = p._all_margins(p.fk_all(q)["centers"])
    collar_u = d_all[:, cm["collar_U"]].amin(dim=1)
    ee = ee_positions(p, q)
    ee0 = ee_positions(provider_for(1),
                       q_batch(fr3_0.unsqueeze(0), jaka_0.unsqueeze(0)))
    dev = torch.zeros(n)
    for a in ARM_KEYS:
        dev = torch.maximum(dev, (ee[a] - ee0[a][0]).norm(dim=-1))
    # shortfalls saturate at 0 (no bonus for overshoot -> prefer small dev)
    short = torch.stack([
        (fl["cross"] - NOM_TARGET["cross"]).clamp_max(0.0),
        (fl["self"] - NOM_TARGET["self"]).clamp_max(0.0),
        (fl["table"] - NOM_TARGET["table"]).clamp_max(0.0),
        (collar_u - COLLAR_U_MIN).clamp_max(0.0) * 4.0]).min(dim=0).values
    score = short - 0.10 * (dev - 0.15).clamp_min(0.0)
    score[dev > EE_DEV_MAX] = -1e9
    top = score.argsort(descending=True)[: args.n_b]
    print(f"stageA best score {float(score[top[0]]):.4f} "
          f"(incumbent {float(score[0]):.4f})", flush=True)

    # ---- stage B: jitter-robust evaluation of the shortlist ---------------
    best, best_key = None, None
    for rank, idx in enumerate(top.tolist()):
        if float(collar_u[idx]) < COLLAR_U_MIN:
            continue
        jit = jitter_floors(p, fr3[idx], jaka[idx], excl, args.jit_b,
                            seed=args.seed + 1)
        worst = min(min(jit["cross"], jit["self"]) - GATE,
                    jit["table"] - NOM_TARGET["table"])
        key = (round(min(worst, 0.004), 4), -round(float(dev[idx]), 2))
        if best_key is None or key > best_key:
            best_key, best = key, idx
            print(f"  B[{rank}] idx {idx} jitter floors "
                  f"{ {k: round(v, 4) for k, v in jit.items()} } "
                  f"collarU {float(collar_u[idx]):.4f} "
                  f"dev {float(dev[idx]):.3f}", flush=True)
    fr3_b, jaka_b = fr3[best].clone(), jaka[best].clone()

    # ---- stage C: coordinate polish (small steps, keep if jitter improves) -
    p1 = provider_for(1)

    def jworst(f, u):
        jf = jitter_floors(p, f, u, excl, args.jit_b, seed=args.seed + 2)
        da, _, _ = p1._all_margins(
            p1.fk_all(q_batch(f.unsqueeze(0), u.unsqueeze(0)))["centers"])
        cu = float(da[0, cm["collar_U"]].amin())
        v = min(min(jf["cross"], jf["self"]) - GATE,
                jf["table"] - NOM_TARGET["table"],
                (cu - COLLAR_U_MIN) * 4.0)
        return v, jf

    cur, curf = jworst(fr3_b, jaka_b)
    for step in (0.08, 0.04):
        for jnt in range(13):
            if jnt == 12:
                continue           # JAKA j6 stays pinned
            for sgn in (1.0, -1.0):
                f2, u2 = fr3_b.clone(), jaka_b.clone()
                if jnt < 7:
                    f2[jnt] = (f2[jnt] + sgn * step).clamp(
                        FR3_LIMITS[jnt, 0], FR3_LIMITS[jnt, 1])
                else:
                    u2[jnt - 7] = (u2[jnt - 7] + sgn * step).clamp(
                        JAKA_LIMITS[jnt - 7, 0], JAKA_LIMITS[jnt - 7, 1])
                v, vf = jworst(f2, u2)
                if v > cur + 1e-4:
                    fr3_b, jaka_b, cur, curf = f2, u2, v, vf
    print(f"stageC polished worst-floor score {cur:.4f}", flush=True)

    rep = report_pose(fr3_b, jaka_b, n_jitter=8192)
    p1 = provider_for(1)
    rep["tightest"] = tightest(
        p1, q_batch(fr3_b.unsqueeze(0), jaka_b.unsqueeze(0)))
    print("SEARCH_RESULT " + json.dumps(rep))


if __name__ == "__main__":
    main()
