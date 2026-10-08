"""S4: v7 出生位形重搜(R14 验收项 "birth_pose_search --scene v7 重搜后回填").

v4/v5 工具(birth_pose_search.py)的 v7 移植, 独立成文件的原因: 红线要求
不动 v5/v6 任何行为, 而旧工具的 --scene 开关/JAKA 限位/收藏对逻辑都是
v5 语义. 几何底座 = v7_offline_geometry(与 Isaac 冒烟四桶逐位对齐,
见该模块 --selftest).

v7 要求(任务书 2026-08-19):
  ① 静息全通道 margin>0 且余量 >=15mm(四桶口径 cross/self_F/self_U/table);
  ② 四臂 EE 在各自可达工作区内、朝向共享带(|x|<0.20);
  ③ +-0.05 rad 每关节均匀扰动(四臂独立抽样) 50 次不产生出生违规(margin<0);
  ④ 对称性: 同排双臂同一组关节值(init_qpos schema 单 franka/ur 块保证).

搜索结构与 v4 工具同款三段: A 名义边距海选 -> B 抖动稳健短名单 ->
C 坐标精修; EE 位置/朝向作硬过滤(海选即剔除背对共享带/怼天怼地的位形).

用法:
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search_v7 --mode audit
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search_v7 --mode search
  PYTHONPATH=src .venv/bin/python -m safeduo.envs.birth_pose_search_v7 --mode verify \
      --fr3 "..7 值.." --ur "..6 值.."
"""

from __future__ import annotations

import argparse
import json

import torch

from safeduo.envs.v7_offline_geometry import (
    class_floors_v7,
    make_v7_provider,
    tightest_v7,
    ur5_v7_limits,
    v7_init_q,
)

FR3_LIMITS = torch.tensor([
    (-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007),
    (-3.0421, -0.1518), (-2.8065, 2.8065), (0.5445, 4.5169),
    (-3.0159, 3.0159)])
# UR URDF 限位是 +-2pi(elbow +-pi); 搜索空间收到 +-pi 防绕圈等价解
UR_LIMITS = ur5_v7_limits().clamp(-torch.pi, torch.pi)

# 出生位候选的每关节搜索半径(围绕现任; 腕大肩小, 保住"备菜位"意图)
FR3_DELTA = torch.tensor([0.40, 0.35, 0.40, 0.45, 0.50, 0.50, 0.60])
UR_DELTA = torch.tensor([0.45, 0.40, 0.50, 0.70, 0.70, 0.80])
JITTER = 0.05                 # 任务书 ③
N_ACCEPT = 50                 # 任务书 ③ 抽样数(验收用; 搜索用更大 n 保稳健)
REST_GATE = 0.015             # 任务书 ① 静息全通道 >=15mm
# 名义目标(带抖动余量): 0.05 rad 抖动最坏可吃掉 ~2-3cm 远端边距
NOM_TARGET = {"cross": 0.10, "self_F": 0.030, "self_U": 0.030, "table_ex": 0.035}
# 抖动下的通道保底(worst-over-jitter): cross 沿用 v4 裁定 2cm, 其余 +5mm
JIT_GATE = {"cross": 0.02, "self_F": 0.005, "self_U": 0.005, "table_ex": 0.005}
EE_DEV_MAX = 0.50             # 相对现任 EE 的最大漂移(m, 任一臂)


def q_batch(fr3: torch.Tensor, ur: torch.Tensor) -> dict:
    return {"F_L": fr3, "F_R": fr3.clone(), "U_L": ur, "U_R": ur.clone()}


def ee_filter(p, q: dict) -> tuple:
    """EE 位置/朝向硬过滤(任务书 ②). 返回 (ok(N,), 诊断 dict).

    - 位置: F EE x in [0.10, 0.55](基座前方朝共享带), U EE x in [-0.55, -0.10];
      z in [0.85, 1.39](桌面 +5cm ~ AABB 顶); |y-基座y| <= 0.40;
      离本臂基座 <= 0.78 m(可达内, 不满伸);
    - 朝向: 手轴(法兰 +z, F2/DFX 手都沿它伸出)不得背对共享带 --
      F 要求轴 x 分量 <= 0.15, U 要求 >= -0.15(允许朝下备菜, 禁止反指自家).
    """
    fko = p.fk_all(q)
    ee, rot = p.ee_pose(fko)
    n = q["F_L"].shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    diag = {}
    for a in ("F_L", "F_R", "U_L", "U_R"):
        base = torch.tensor(p.layout.base_pose(a)[0])
        pos, ax = ee[a], rot[a][:, :, 2]
        if a.startswith("F"):
            ok &= (pos[:, 0] >= 0.10) & (pos[:, 0] <= 0.55) & (ax[:, 0] <= 0.15)
        else:
            ok &= (pos[:, 0] <= -0.10) & (pos[:, 0] >= -0.55) & (ax[:, 0] >= -0.15)
        ok &= (pos[:, 2] >= 0.85) & (pos[:, 2] <= 1.39)
        ok &= (pos[:, 1] - base[1]).abs() <= 0.40
        ok &= (pos - base).norm(dim=-1) <= 0.78
        diag[a] = {"ee": [round(float(x), 3) for x in pos[0]],
                   "hand_axis": [round(float(x), 3) for x in ax[0]],
                   "reach_from_base": round(float((pos - base).norm(dim=-1)[0]), 3)}
    return ok, diag


def floors_batch(p, fr3: torch.Tensor, ur: torch.Tensor,
                 chunk: int = 1024) -> dict:
    """(N,7)/(N,6) 同批四桶 floor, 分块防 cross 对距离场爆内存."""
    outs = []
    for i in range(0, fr3.shape[0], chunk):
        outs.append(class_floors_v7(p, q_batch(fr3[i:i + chunk], ur[i:i + chunk])))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def jitter_floors(p, fr3: torch.Tensor, ur: torch.Tensor, n: int,
                  seed: int = 0) -> dict:
    """单候选: n 组 +-JITTER 扰动(四臂独立)的 worst-over-jitter 四桶."""
    g = torch.Generator().manual_seed(seed)

    def jit(base, dof, lim):
        return (base.expand(n, dof)
                + (torch.rand(n, dof, generator=g) * 2 - 1) * JITTER
                ).clamp(lim[:, 0], lim[:, 1])

    q = {"F_L": jit(fr3, 7, FR3_LIMITS), "F_R": jit(fr3, 7, FR3_LIMITS),
         "U_L": jit(ur, 6, UR_LIMITS), "U_R": jit(ur, 6, UR_LIMITS)}
    fl = class_floors_v7(p, q)
    return {k: float(v.min()) for k, v in fl.items()}


def report_pose(p, fr3: torch.Tensor, ur: torch.Tensor,
                n_jitter: int = 8192) -> dict:
    q1 = q_batch(fr3.unsqueeze(0), ur.unsqueeze(0))
    nom = {k: float(v[0]) for k, v in class_floors_v7(p, q1).items()}
    jit = jitter_floors(p, fr3, ur, n_jitter, seed=1)
    jit50 = jitter_floors(p, fr3, ur, N_ACCEPT, seed=2026)
    _, ee_diag = ee_filter(p, q1)
    return {
        "fr3": [round(float(x), 4) for x in fr3],
        "ur": [round(float(x), 4) for x in ur],
        "nominal_floors_mm": {k: round(v * 1000, 1) for k, v in nom.items()},
        "gate_rest_ge_15mm_ex_struct": all(
            nom[k] >= REST_GATE for k in NOM_TARGET),
        "table_struct_mm": round(nom["table_struct"] * 1000, 2),
        f"jitter{JITTER}_floors_mm_n{n_jitter}":
            {k: round(v * 1000, 1) for k, v in jit.items()},
        f"jitter{JITTER}_floors_mm_n{N_ACCEPT}_seed2026":
            {k: round(v * 1000, 1) for k, v in jit50.items()},
        "gate_jitter_no_violation": all(v > 0 for v in jit.values()),
        "gate_jitter50_no_violation": all(v > 0 for v in jit50.values()),
        "ee": ee_diag,
        "tightest": [(round(m * 1000, 1), a, b)
                     for m, a, b in tightest_v7(p, q1, k=8)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "search", "verify"],
                    default="audit")
    ap.add_argument("--fr3", type=str, default="")
    ap.add_argument("--ur", type=str, default="")
    ap.add_argument("--n_a", type=int, default=16384)
    ap.add_argument("--n_b", type=int, default=48)
    ap.add_argument("--jit_b", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = make_v7_provider(1)
    fr3_def, ur_def = v7_init_q()
    fr3_0 = torch.tensor([float(x) for x in args.fr3.split(",")]) \
        if args.fr3 else fr3_def
    ur_0 = torch.tensor([float(x) for x in args.ur.split(",")]) \
        if args.ur else ur_def

    if args.mode in ("audit", "verify"):
        nj = 8192 if args.mode == "verify" else 2048
        print(args.mode.upper() + " " +
              json.dumps(report_pose(p, fr3_0, ur_0, n_jitter=nj),
                         ensure_ascii=False))
        return

    # ---- stage A: 名义边距海选 --------------------------------------------
    torch.manual_seed(args.seed)
    n = args.n_a
    df = (torch.rand(n, 7) * 2 - 1) * FR3_DELTA
    du = (torch.rand(n, 6) * 2 - 1) * UR_DELTA
    df[0], du[0] = 0.0, 0.0            # 现任保留参赛
    fr3 = (fr3_0 + df).clamp(FR3_LIMITS[:, 0], FR3_LIMITS[:, 1])
    ur = (ur_0 + du).clamp(UR_LIMITS[:, 0], UR_LIMITS[:, 1])
    fl = floors_batch(p, fr3, ur)
    ok_ee, _ = ee_filter(p, q_batch(fr3, ur))
    # EE 漂移(相对现任)软惩罚: 保住备菜位意图
    ee0, _ = p.ee_pose(p.fk_all(q_batch(fr3_0.unsqueeze(0), ur_0.unsqueeze(0))))
    ee, _ = p.ee_pose(p.fk_all(q_batch(fr3, ur)))
    dev = torch.zeros(n)
    for a in ("F_L", "F_R", "U_L", "U_R"):
        dev = torch.maximum(dev, (ee[a] - ee0[a][0]).norm(dim=-1))
    short = torch.stack([
        (fl[k] - NOM_TARGET[k]).clamp_max(0.0) for k in NOM_TARGET
    ]).min(dim=0).values
    score = short - 0.05 * (dev - 0.20).clamp_min(0.0)
    score[~ok_ee] = -1e9
    score[dev > EE_DEV_MAX] = -1e9
    top = score.argsort(descending=True)[: args.n_b]
    print(f"stageA 过 EE 过滤 {int(ok_ee.sum())}/{n}, "
          f"best score {float(score[top[0]]):.4f} "
          f"(incumbent {float(score[0]):.4f})", flush=True)

    # ---- stage B: 抖动稳健短名单 ------------------------------------------
    def jkey(f, u, seed):
        jf = jitter_floors(p, f, u, args.jit_b, seed=seed)
        return min(jf[k] - JIT_GATE[k] for k in JIT_GATE), jf

    best, best_key, best_jf = None, None, None
    for rank, idx in enumerate(top.tolist()):
        v, jf = jkey(fr3[idx], ur[idx], args.seed + 1)
        key = (round(min(v, 0.005), 4), -round(float(dev[idx]), 2))
        if best_key is None or key > best_key:
            best_key, best, best_jf = key, idx, jf
            print(f"  B[{rank}] idx {idx} jitter floors(mm) "
                  f"{ {k: round(x * 1000, 1) for k, x in jf.items()} } "
                  f"dev {float(dev[idx]):.3f}", flush=True)
    fr3_b, ur_b = fr3[best].clone(), ur[best].clone()

    # ---- stage C: 坐标精修 -------------------------------------------------
    cur, _ = jkey(fr3_b, ur_b, args.seed + 2)
    for step in (0.08, 0.04, 0.02):
        for jnt in range(13):
            for sgn in (1.0, -1.0):
                f2, u2 = fr3_b.clone(), ur_b.clone()
                if jnt < 7:
                    f2[jnt] = (f2[jnt] + sgn * step).clamp(
                        FR3_LIMITS[jnt, 0], FR3_LIMITS[jnt, 1])
                else:
                    u2[jnt - 7] = (u2[jnt - 7] + sgn * step).clamp(
                        UR_LIMITS[jnt - 7, 0], UR_LIMITS[jnt - 7, 1])
                okc, _ = ee_filter(p, q_batch(f2.unsqueeze(0), u2.unsqueeze(0)))
                if not bool(okc[0]):
                    continue
                v, _ = jkey(f2, u2, args.seed + 2)
                if v > cur + 1e-4:
                    fr3_b, ur_b, cur = f2, u2, v
    print(f"stageC polished worst-floor key {cur:.4f}", flush=True)

    # ---- stage D: facing polish ------------------------------------------
    # requirement (2): hands should face the shared band. greedy pass over
    # the wrist-ish joints maximizing facing = (-axF_x) + (+axU_x), keeping
    # the jitter-robust key above max(cur-0.002, 0.010) and EE filter green.
    def facing(f, u):
        _, rot = p.ee_pose(p.fk_all(q_batch(f.unsqueeze(0), u.unsqueeze(0))))
        return float(-rot["F_L"][0, 0, 2] + rot["U_L"][0, 0, 2])

    key_floor = max(cur - 0.002, 0.010)
    fcur = facing(fr3_b, ur_b)
    for step in (0.15, 0.08, 0.04):
        for jnt in (0, 4, 5, 6, 7, 10, 11, 12):   # F j1/j5-j7, U pan/w1-w3
            for sgn in (1.0, -1.0):
                f2, u2 = fr3_b.clone(), ur_b.clone()
                if jnt < 7:
                    f2[jnt] = (f2[jnt] + sgn * step).clamp(
                        FR3_LIMITS[jnt, 0], FR3_LIMITS[jnt, 1])
                else:
                    u2[jnt - 7] = (u2[jnt - 7] + sgn * step).clamp(
                        UR_LIMITS[jnt - 7, 0], UR_LIMITS[jnt - 7, 1])
                okc, _ = ee_filter(p, q_batch(f2.unsqueeze(0), u2.unsqueeze(0)))
                if not bool(okc[0]):
                    continue
                f_new = facing(f2, u2)
                if f_new <= fcur + 1e-3:
                    continue
                v, _ = jkey(f2, u2, args.seed + 2)
                if v >= key_floor:
                    fr3_b, ur_b, fcur = f2, u2, f_new
                    if v > cur:
                        cur = v
    print(f"stageD facing polished: facing_sum {fcur:.3f} key {cur:.4f}",
          flush=True)

    print("SEARCH_RESULT " +
          json.dumps(report_pose(p, fr3_b, ur_b), ensure_ascii=False))


if __name__ == "__main__":
    main()
