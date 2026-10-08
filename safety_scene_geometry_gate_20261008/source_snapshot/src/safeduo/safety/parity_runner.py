"""A3 正式对拍 v2：10^5 随机构型，球距离 vs ContactSensor（B 正式球分解）。

协议 v2（2026-08-12，Agent A2）：v1 的 gt_hit=100% 系 GT 通道污染——
fr3_hand 经 fixed-joint 挂到 fr3_link7，两者碰撞体互穿，PhysX 报 ~9.7e4 N
永久焊接伪力（探针 parity_probe.py 实证：静止位姿唯二受力 body，双 F 臂同现；
U 臂/基座均无力）。净接触力 GT 无法做逐对归因，故 v2 将焊接对涉及的
F 臂 {fr3_link7, fr3_link8, fr3_hand, fr3_hand_tcp} 移出 GT body 集。

覆盖性（牛顿第三定律）：F 手/link7 撞【对方机器人臂杆 / F 另一臂 link1-6 /
本臂非邻接杆】时，对方 body 在 GT 集内仍受力可测。残留 GT 盲区（如实上报）：
  1. F 手/link7 <-> F 手/link7（双 Franka 末端互碰）
  2. F 手/link7 <-> 桌面（近桌豁免语义下本也放行低速贴桌）
盲区内漏检由 tests/test_sphere_distance.py 手工姿态用例兜底；升级路径 =
filter_prim_paths_expr 逐对力矩阵（M1 接）。

口径：
  预测碰撞 = SphereDistOut.violation（margin<0 非豁免对；官方资产无 /hand/
             子树，near_table 条件豁免对为空集，pred_policy ≡ pred_raw，
             两者都记录并在 JSON 里核对）
  真实碰撞 = 任一【GT 集内带球 body】净接触力 > force_eps
  漏检率 = miss / gt_hit（G0 < 0.1%），误报率 = false_pos / pred_hit（G0 < 5%）
归因：miss 按 GT body 直方图（谁受力而球没报）；FP 按违规球对直方图
（@B 迭代半径的 top 贡献对），全向量化累计。

用法：python -m safeduo.safety.parity_runner --num_envs 256 --configs 100000 --headless
输出：~/safeduo/artifacts/parity/a3_contact_<date>_v2.json

协议 v3（2026-08-12，监管线对账指令）——测量口径对齐安全语义，几何不背协议的锅：
  背景：B 球分解 v2 大改（shoulder -30%/link5 +球/link0 补形）后两门数字零响应；
  eps5 的 miss_margin_hist 显示 65% 漏检发生在最近 judged 对还差 >=1.5cm 处
  （众数 2-3cm）——接触发生在【豁免对】上（邻接 link4/6、显式豁免 5-7/5-hand、
  焊接剔除体），GT 按 body 记力而球侧按设计不看，被记成"漏检"。
  1. GT 记账豁免对齐：每个 GT 命中 env 取受力最大 body，用全球集（含豁免对
     球）+ 桌 SDF 推断最近接触伙伴（"受力方向/最近 body 推断配对"），伙伴对
     若在 judged 集 -> chargeable（可追责）；豁免邻接/焊接体/table_skip ->
     设计豁免，不记入可追责漏检。gate_miss_v3 用 chargeable 口径，
     raw 数字并列保留（不是改门槛，是把责任对齐到检测器职责范围）。
  2. FP 穿透深度直方图：FP 时球面最深穿透的分布——峰在 0-2cm = 保守球壳的
     near-touch 设计属性（网格真距查询不可用，穿透深度是可用代理，如实标注）。
  3. --protocol rollout：状态分布从均匀随机teleport换成运行工况滚动
     （coordinator env + l2_mix 流 + L2 兜底在环 + alpha=1 直通 = 部署语义的
     保守上界），FP 门在该分布上判定（论文口径"部署工况包络内"）。
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--configs", type=int, default=100000)
parser.add_argument("--force_eps", type=float, default=1.0)
parser.add_argument("--yaml", type=str, default="duo_env.yaml",
                    help="env profile; duo_env_v4.yaml = v4 real-asset scene (A6)")
parser.add_argument("--protocol", choices=["teleport", "rollout"], default="teleport",
                    help="teleport=均匀随机构型（miss 门）；rollout=运行工况滚动（FP 门）")
parser.add_argument("--followthrough_steps", type=int, default=5,
                    help="rollout 去删失：球报警后跟踪 K 步看力是否发展（v3.1），"
                         "并停用违规终止防止重置掐断因果链；0=旧行为")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.sphere_audit import sphere_provenance  # noqa: E402
from safeduo.safety.sphere_distance import TABLE_NAMES  # noqa: E402
from safeduo.safety.types import ARM_KEYS  # noqa: E402

# fixed-joint 焊接伪力体（探针实证 9.66e4 N @ rest）；按名剔除，U 臂天然不含
WELD_EXCLUDE = {"fr3_link7", "fr3_link8", "fr3_hand", "fr3_hand_tcp"}

# 漏检时"球面还差多远才报警"的分桶（m）；对照 B 的 +10mm 期望：
# 质量落在 >0.01 桶 = 加径 1cm 也吃不掉的欠覆盖
MISS_MARGIN_EDGES = [0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 1.0]
# FP 时球面最深穿透深度分桶（m，取 |min margin|）；峰在 0-2cm = near-touch 签名
FP_PEN_EDGES = [0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 1.0]


class PartnerAttributor:
    """GT 命中的最近接触伙伴推断（协议 v3 记账核心）。

    受力最大 body 的球集 vs 全场球集（含豁免对）+ 桌 SDF，最近者为伙伴；
    伙伴关系按 judged 集分类。焊接体 link8/hand_tcp 无球，其接触会归到
    最近带球体（link7/hand）——已知局限，量级由 weld 桶整体覆盖。
    """

    def __init__(self, env, sph, dev):
        self.sph, self.dev = sph, dev
        self.arm_index = {a: i for i, a in enumerate(ARM_KEYS)}
        S = sph.n_spheres
        self.body_of_sphere = torch.zeros(S, dtype=torch.long, device=dev)
        for arm in ARM_KEYS:
            sl = sph._arm_slices[arm]
            self.body_of_sphere[sl] = sph._body_idx[arm]
        self.arm_of_sphere = sph.arm_id
        self.qnames = list(sph.qualified_names)
        self.radii = sph.radii
        judged, judged_table = set(), set()
        pt = sph.pair_table
        for k in range(int(sph._slice_table.start)):
            judged.add(frozenset((self.qnames[int(pt[k, 0])],
                                  self.qnames[int(pt[k, 1])])))
        for k in range(int(sph._slice_table.start), int(sph.n_pairs)):
            judged_table.add((self.qnames[int(pt[k, 0])],
                              TABLE_NAMES[int(pt[k, 1])]))
        self.judged, self.judged_table = judged, judged_table
        tb = env.cfg.table_board
        self.tb_c = torch.tensor(tb["centers"], device=dev, dtype=torch.float32)
        self.tb_h = torch.tensor(tb["half"], device=dev, dtype=torch.float32)
        self.origins = env.scene.env_origins

    def __call__(self, e: int, arm_star: str, b_star: int, gt_link: str):
        """-> (gt_qname, partner_name, relation, gap_m)。"""
        sph = self.sph
        gt_q = f"{arm_star}/{gt_link}"
        mine = (self.arm_of_sphere == self.arm_index[arm_star]) \
            & (self.body_of_sphere == int(b_star))
        mi = mine.nonzero(as_tuple=True)[0]
        cs = sph.last_centers[e]                                   # (S,3) world
        d = (cs[mi].unsqueeze(1) - cs.unsqueeze(0)).norm(dim=-1) \
            - self.radii[mi].unsqueeze(1) - self.radii.unsqueeze(0)
        d[:, mine] = 1e9
        gap_s = float(d.min())
        j = int(d.min(dim=0).values.argmin())
        # 桌 SDF（env-local）
        p = cs[mi] - self.origins[e]
        q = (p.unsqueeze(1) - self.tb_c.unsqueeze(0)).abs() - self.tb_h
        sdf = q.clamp(min=0).norm(dim=-1) + q.amax(dim=-1).clamp(max=0) \
            - self.radii[mi].unsqueeze(1)                          # (m,2)
        gap_t = float(sdf.min())
        if gap_t < gap_s:
            t = int(sdf.min(dim=0).values.argmin())
            partner = TABLE_NAMES[t]
            rel = "judged_table" if (gt_q, partner) in self.judged_table \
                else "exempt_table_skip"
            return gt_q, partner, rel, gap_t
        partner = self.qnames[j]
        if frozenset((gt_q, partner)) in self.judged:
            return gt_q, partner, "judged", gap_s
        p_link = partner.split("/", 1)[1]
        if p_link in WELD_EXCLUDE:
            return gt_q, partner, "exempt_weld_partner", gap_s
        if partner.split("/", 1)[0] == arm_star:
            return gt_q, partner, "exempt_same_arm_adjacent", gap_s
        return gt_q, partner, "unclassified", gap_s   # cross 全 judged，理论不可达


def pair_name(sph, k: int) -> str:
    i, j = int(sph.pair_table[k, 0]), int(sph.pair_table[k, 1])
    if k >= sph._slice_table.start:
        return f"{sph.qualified_names[i]}|{TABLE_NAMES[j]}"
    return f"{sph.qualified_names[i]}|{sph.qualified_names[j]}"


def main():
    n = args.num_envs
    rollout = args.protocol == "rollout"
    ft = args.followthrough_steps if rollout else 0
    # rollout = coordinator 模式（l2_mix 流 + L2 兜底在环），teleport = raw
    cfg = make_duo_env_cfg(num_envs=n, coordinator=rollout, yaml_name=args.yaml,
                           device=getattr(args, "device", None) or "cuda:0")
    cfg.enable_contact_gt = True
    if ft > 0:
        # v3.1 去删失：违规不终止（否则报警瞬间即重置，力永远来不及发展，
        # 42/53 "FP" 实为被删失的提前预警——2026-08-12 rollout eps1 教训）
        cfg.coordinator["terminate_on_violation"] = False
    env = DuoEnv(cfg)
    env.reset()
    dev = env.device
    sph = env._sph
    layout = ("real_v5" if "v5" in args.yaml
              else "real_v3" if "v4" in args.yaml else "v0")
    prov = sphere_provenance(env._sph, layout=layout)  # 几何不实 -> 直接 raise，拒绝出数
    robots = (("fr3", "jaka_zu7", "f2_left", "f2_right")
              if layout in ("real_v3", "real_v5") else ("fr3", "ur5e"))
    print("PARITY_SPHERES " + json.dumps(
        {r: {"spec_version": prov[r]["spec_version"], "sha1": prov[r]["sha1"]}
         for r in robots} | {"n_spheres": prov["n_spheres"],
                             "n_pairs": prov["n_pairs"],
                             "layout": layout,
                             "protocol": args.protocol}), flush=True)
    attributor = PartnerAttributor(env, sph, dev)
    passthrough = None
    if rollout:
        # alpha=1 直通 + p=0：兜底在环的部署语义保守上界（未过滤指令流）
        passthrough = torch.zeros(n, 5, device=dev)
        passthrough[:, :4] = 1.0

    # gt_idx 必须是 ContactSensor 张量的索引空间（v0 与 articulation 索引巧合
    # 一致；v4 组合 USD 的手 body 在 f2_hand/ 下一层，sensor 单层正则匹配不到
    # -> 名字映射 + 缺失记录。GPU 上索引越界会化身 fabric device assert，
    # CPU 才现 IndexError 原形——A6 五连发排障结论）
    gt_idx, gt_names, excluded = {}, {}, []
    for arm in ARM_KEYS:
        names = list(env._arms[arm].body_names)
        sensor_of = {nm: i for i, nm in enumerate(env._contact[arm].body_names)}
        keep_sensor, keep_names = [], []
        for b in torch.unique(sph._body_idx[arm]).tolist():
            nm = names[int(b)]
            if nm in WELD_EXCLUDE:
                excluded.append(f"{arm}/{nm}")
            elif nm not in sensor_of:
                excluded.append(f"{arm}/{nm}(no-sensor)")
            else:
                keep_sensor.append(sensor_of[nm])
                keep_names.append(nm)
        gt_idx[arm] = torch.tensor(keep_sensor, dtype=torch.long, device=dev)
        gt_names[arm] = keep_names
    print(f"PARITY_GT_EXCLUDED {excluded}", flush=True)

    limits = {}
    for arm in ARM_KEYS:
        art, jid = env._arms[arm], env._joint_idx[arm]
        lim = art.data.soft_joint_pos_limits[0, jid]
        limits[arm] = (lim[:, 0], lim[:, 1])
    gen = torch.Generator(device=dev).manual_seed(7)

    steps = (args.configs + n - 1) // n
    stats = {"configs": 0, "gt_hit": 0, "pred_hit": 0, "pred_raw_hit": 0,
             "miss": 0, "false_pos": 0, "gt_chargeable": 0, "miss_chargeable": 0}
    fp_pair_hist = torch.zeros(sph.n_pairs, dtype=torch.long, device=dev)
    miss_body_hist = {arm: torch.zeros(len(gt_names[arm]), dtype=torch.long, device=dev)
                      for arm in ARM_KEYS}
    # B round-3 工单：漏检时全场最小球面 margin 的分布（还差多远才会报警）
    mm_edges = torch.tensor(MISS_MARGIN_EDGES, device=dev)
    miss_margin_hist = torch.zeros(len(MISS_MARGIN_EDGES) - 1,
                                   dtype=torch.long, device=dev)
    # 可追责漏检的缺口分布（B 的可行动直方图：真覆盖缺口有多深）
    chargeable_gap_hist = torch.zeros(len(MISS_MARGIN_EDGES) - 1,
                                      dtype=torch.long, device=dev)
    fp_edges = torch.tensor(FP_PEN_EDGES, device=dev)
    fp_pen_hist = torch.zeros(len(FP_PEN_EDGES) - 1, dtype=torch.long, device=dev)
    miss_partner_counts: dict = {}
    fp_samples = []
    # v3.1 跟踪窗（rollout 去删失）：球报警 rising edge 开窗，K 步内力到 =
    # 提前预警（TP，记提前量），到期无力 = 确证 FP，窗内遇 episode 重置 = 删失
    ft_stats = {"alarms": 0, "tp_instant": 0, "tp_early": 0,
                "fp_confirmed": 0, "censored_by_reset": 0}
    delay_hist = torch.zeros(max(ft, 1) + 1, dtype=torch.long, device=dev)
    prev_pred = torch.zeros(n, dtype=torch.bool, device=dev)
    pend_until = torch.full((n,), -1, dtype=torch.long, device=dev)
    pend_t0 = torch.zeros(n, dtype=torch.long, device=dev)
    pend_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    for it in range(steps):
        if rollout:
            env.step(passthrough)
        else:
            for arm in ARM_KEYS:
                art, jid = env._arms[arm], env._joint_idx[arm]
                lo, hi = limits[arm]
                u = torch.rand(n, len(jid), device=dev, generator=gen)
                q = art.data.default_joint_pos.clone()
                q[:, jid] = lo + u * (hi - lo)
                art.write_joint_state_to_sim(q, torch.zeros_like(q))
                art.set_joint_position_target(q)
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        out = env.compute_dist(need_full=True)
        pred = out.violation
        pred_raw = (out.dists < 0.0).any(dim=1)
        fb_per_arm = {}
        forces = []
        for arm in ARM_KEYS:
            f = env._contact[arm].data.net_forces_w        # (N, B, 3)
            fb = f[:, gt_idx[arm]].norm(dim=-1)            # (N, K)
            fb_per_arm[arm] = fb
            forces.append(fb.amax(dim=-1))
        gt = torch.stack(forces, dim=-1).amax(dim=-1) > args.force_eps
        miss_m = gt & ~pred
        fp_m = pred & ~gt
        stats["configs"] += n
        stats["gt_hit"] += int(gt.sum())
        stats["pred_hit"] += int(pred.sum())
        stats["pred_raw_hit"] += int(pred_raw.sum())
        stats["miss"] += int(miss_m.sum())
        stats["false_pos"] += int(fp_m.sum())
        # 归因直方图（全向量化）
        fp_pair_hist += ((out.dists < 0.0) & fp_m.unsqueeze(1)).sum(dim=0)
        for arm in ARM_KEYS:
            miss_body_hist[arm] += ((fb_per_arm[arm] > args.force_eps)
                                    & miss_m.unsqueeze(1)).sum(dim=0)
        if miss_m.any():
            dmin = out.dists[miss_m].amin(dim=1).clamp(min=0.0, max=0.999)
            b = torch.bucketize(dmin, mm_edges[1:-1])
            miss_margin_hist += torch.bincount(b, minlength=miss_margin_hist.numel())
        if fp_m.any():
            pen = (-out.dists[fp_m].amin(dim=1)).clamp(min=0.0, max=0.999)
            b = torch.bucketize(pen, fp_edges[1:-1])
            fp_pen_hist += torch.bincount(b, minlength=fp_pen_hist.numel())
        if ft > 0:
            just_reset = env.episode_length_buf == 0
            active = pend_until >= 0
            pend_hit |= active & gt
            to_close = active & (pend_hit | (it >= pend_until) | just_reset)
            if to_close.any():
                cens = to_close & just_reset & ~pend_hit
                tp = to_close & pend_hit
                fpc = to_close & ~pend_hit & ~just_reset
                ft_stats["censored_by_reset"] += int(cens.sum())
                ft_stats["tp_early"] += int(tp.sum())
                ft_stats["fp_confirmed"] += int(fpc.sum())
                if tp.any():
                    d = (it - pend_t0[tp]).clamp(0, ft)
                    delay_hist += torch.bincount(d, minlength=delay_hist.numel())
                pend_until[to_close] = -1
            rising = pred & ~prev_pred & (pend_until < 0) & ~just_reset
            inst = rising & gt          # 报警瞬间力已在 -> 提前量 0
            ft_stats["alarms"] += int(rising.sum())
            ft_stats["tp_instant"] += int(inst.sum())
            delay_hist[0] += int(inst.sum())
            opens = rising & ~gt
            pend_until[opens] = it + ft
            pend_t0[opens] = it
            pend_hit[opens] = False
            prev_pred = pred.clone()
        # 协议 v3：逐 GT 命中 env 推断最近接触伙伴 -> 可追责/豁免分账
        if gt.any():
            arm_vals, arm_pos = [], []
            for arm in ARM_KEYS:
                v, p = fb_per_arm[arm].max(dim=1)
                arm_vals.append(v)
                arm_pos.append(p)
            arm_vals = torch.stack(arm_vals, dim=-1)               # (N,4)
            best_arm = arm_vals.argmax(dim=-1)                     # (N,)
            miss_cpu = set(miss_m.nonzero(as_tuple=True)[0].tolist())
            for e in gt.nonzero(as_tuple=True)[0].tolist():
                ai = int(best_arm[e])
                arm = ARM_KEYS[ai]
                k = int(arm_pos[ai][e])
                gt_q, partner, rel, gap = attributor(
                    e, arm, int(gt_idx[arm][k]), gt_names[arm][k])
                chargeable = rel in ("judged", "judged_table")
                stats["gt_chargeable"] += int(chargeable)
                if e in miss_cpu:
                    key = f"{gt_q} ~ {partner} [{rel}]"
                    miss_partner_counts[key] = miss_partner_counts.get(key, 0) + 1
                    if chargeable:
                        stats["miss_chargeable"] += 1
                        g = min(max(gap, 0.0), 0.999)
                        gi = int(torch.bucketize(torch.tensor(g, device=dev),
                                                 mm_edges[1:-1]))
                        chargeable_gap_hist[gi] += 1
        if len(fp_samples) < 200:
            for e in fp_m.nonzero(as_tuple=True)[0][:8].tolist():
                mm = {k: round(float(v[e]), 4) for k, v in out.min_margin.items()}
                cols = (out.dists[e] < 0.0).nonzero(as_tuple=True)[0][:4]
                fp_samples.append({"iter": it, "env": e, "min_margin": mm,
                                   "pairs": [pair_name(sph, int(k)) for k in cols]})
        if it % 50 == 0:
            print(f"PARITY_PROGRESS {stats}", flush=True)
    stats["miss_rate"] = stats["miss"] / max(stats["gt_hit"], 1)
    stats["false_rate"] = stats["false_pos"] / max(stats["pred_hit"], 1)
    stats["gate_miss"] = "PASS" if stats["miss_rate"] < 1e-3 else "FAIL"
    stats["gate_false"] = "PASS" if stats["false_rate"] < 0.05 else "FAIL"
    # v3 可追责口径：GT 命中与漏检都只对 judged 对记账（豁免接触=检测器职责外）
    stats["miss_rate_chargeable"] = (stats["miss_chargeable"]
                                     / max(stats["gt_chargeable"], 1))
    stats["gate_miss_v3"] = ("PASS" if stats["miss_rate_chargeable"] < 1e-3
                             else "FAIL")
    if ft > 0:
        adjudicated = (ft_stats["tp_instant"] + ft_stats["tp_early"]
                       + ft_stats["fp_confirmed"])
        stats["ft"] = ft_stats
        stats["false_rate_followthrough"] = (ft_stats["fp_confirmed"]
                                             / max(adjudicated, 1))
        stats["gate_false_v31"] = ("PASS" if
                                   stats["false_rate_followthrough"] < 0.05
                                   else "FAIL")
        stats["alarm_delay_hist_steps"] = delay_hist.tolist()

    top_fp = torch.argsort(fp_pair_hist, descending=True)[:20]
    fp_top_pairs = [{"pair": pair_name(sph, int(k)), "count": int(fp_pair_hist[k])}
                    for k in top_fp if int(fp_pair_hist[k]) > 0]
    miss_top_bodies = []
    for arm in ARM_KEYS:
        h = miss_body_hist[arm]
        for b in torch.argsort(h, descending=True)[:5]:
            if int(h[b]) > 0:
                miss_top_bodies.append({"body": f"{arm}/{gt_names[arm][int(b)]}",
                                        "count": int(h[b])})
    miss_top_bodies.sort(key=lambda r: -r["count"])

    out_dir = Path.home() / "safeduo" / "artifacts" / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    def _hist_table(hist, edges):
        return [{"bin_m": f"[{edges[i]:.3f},{edges[i + 1]:.3f})",
                 "count": int(hist[i])}
                for i in range(hist.numel()) if int(hist[i]) > 0]

    miss_partner_top = sorted(miss_partner_counts.items(),
                              key=lambda kv: -kv[1])[:20]
    payload = {"date": str(date.today()),
               "protocol": {"version": 3, "state_distribution": args.protocol,
                            "miss_accounting": "chargeable = 最近伙伴推断为 judged 对"
                                               "（豁免邻接/焊接体/table_skip 接触不追责）",
                            "fp_note": "FP 门以 rollout 分布为准（部署工况包络）；"
                                       "fp_pen_hist 为 near-touch 签名的可用代理"},
               "force_eps": args.force_eps, "spheres": prov,
               "gt_excluded_bodies": excluded,
               "coverage_notes": "F hand/link7 vs {F hand/link7, table} 不在 GT 内"
                                 "（焊接伪力剔除的代价）；其余对经对方受力体覆盖",
               "stats": stats, "fp_top_pairs": fp_top_pairs,
               "miss_top_bodies": miss_top_bodies[:20],
               "miss_margin_hist": _hist_table(miss_margin_hist, MISS_MARGIN_EDGES),
               "chargeable_gap_hist": _hist_table(chargeable_gap_hist,
                                                  MISS_MARGIN_EDGES),
               "fp_pen_hist": _hist_table(fp_pen_hist, FP_PEN_EDGES),
               "miss_partner_top": [{"pair": k, "count": v}
                                    for k, v in miss_partner_top],
               "false_positive_samples": fp_samples}
    # 场景布局进文件名：v4(real_v3) 运行与 v0 时代产物同日同协议会撞名
    # 覆盖登记证据（A6 六发事故：A3 的 v3_teleport_eps1 被覆盖后从本地恢复）
    scene_tag = ("v5scene_" if "v5" in args.yaml
                 else "v4scene_" if "v4" in args.yaml else "")
    fname = (f"a3_contact_{date.today().strftime('%Y%m%d')}"
             f"_v3_{scene_tag}{args.protocol}_eps{args.force_eps:g}.json")
    (out_dir / fname).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("PARITY_FINAL " + json.dumps(stats), flush=True)
    print("PARITY_FP_TOP " + json.dumps(fp_top_pairs[:8], ensure_ascii=False), flush=True)
    print("PARITY_MISS_TOP " + json.dumps(miss_top_bodies[:8], ensure_ascii=False), flush=True)
    print("PARITY_MISS_PARTNERS " + json.dumps(
        [{"pair": k, "count": v} for k, v in miss_partner_top[:10]],
        ensure_ascii=False), flush=True)
    print("PARITY_FILE " + str(out_dir / fname), flush=True)


if __name__ == "__main__":
    main()
    app.close()
