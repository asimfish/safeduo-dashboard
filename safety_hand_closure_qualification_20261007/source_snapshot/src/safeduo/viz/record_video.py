"""V2 实时查看器雏形（A6 雏形，W2 交付：一段带 overlay 的视频）。

内容：4 env 场景相机渲染 + 每臂 EE 处 commanded(黄)/executed(绿) 双箭头
（EE 雅可比把关节 delta 投到笛卡尔）+ 基座 alpha 着色圆点（绿=1 红=0）
+ 顶部 p 让行条 + 最小跨机 margin 文本。alpha/p 默认脚本振荡驱动；
--checkpoint 传 train_ppo checkpoint 即真策略驱动（obs->actor 均值，
与 block1_harness 的 PolicyDriver 同路径）；--scenario 强制全部 env
跑单一 L2 冲突族（如 head_on_crossing / handover_approach）出定性片。

用法：python -m safeduo.viz.record_video --steps 300 --headless
真策略：... --checkpoint artifacts/runs/<run>/model_N.pt --scenario head_on_crossing
输出：~/safeduo/artifacts/viz/v2_proto_<date>.mp4（真策略为 v2_policy_<scenario>_<date>.mp4）
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=4)  # CPU 物理路径建议 1（规避 fabric assert 的替代方案）
parser.add_argument("--checkpoint", type=str, default="",
                    help="train_ppo checkpoint；给了就用真策略驱动 alpha/p（默认脚本振荡）")
parser.add_argument("--scenario", type=str, default="",
                    help="强制全部 env 跑该 L2 冲突族（l2_env_source SCENARIOS 键名）")
parser.add_argument("--alpha1", action="store_true",
                    help="alpha 恒 1/p 恒 0（纯直通+damper 兜底演示，不振荡）")
parser.add_argument("--fabric", action="store_true",
                    help="开 fabric（GPU 物理位姿→渲染同步必需；fabric 关会导致渲染里机器人冻结。"
                         "崩溃规避：不要 CUDA_VISIBLE_DEVICES，用 --device cuda:N 选卡）")
parser.add_argument("--env_yaml", type=str, default="duo_env.yaml",
                    help="env 配置 yaml；v4 组合 USD 场景（四臂带 F2 手）传 duo_env_v4.yaml")
parser.add_argument("--plain_scene", action="store_true",
                    help="保留旧白盒桌/原版机器人 USD（跳过 B3 复刻视觉层）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

# fabric 关 + updateToUsd 关（Isaac Lab 性能默认）= 物理位姿到不了渲染 stage，
# 画面永远停在出生姿态。CPU 物理管线支持 updateToUsd，录像强制打开。
carb.settings.get_settings().set_bool("/physics/updateToUsd", True)

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS  # noqa: E402

W, H = 1280, 720  # 与 duo_env.enable_viz_camera 的 CameraCfg 一致


class Projector:
    def __init__(self, cam: Camera, eye: np.ndarray, target: np.ndarray):
        # 外参直接从 eye/target 构造（-Z 前 +Y 上），不再依赖相机 data 的
        # 四元数约定（历史上 quat_w_world/+X 前被当 -Z 前用，点位全错）
        self.K = cam.data.intrinsic_matrices[0].cpu().numpy()
        eye = np.asarray(eye, dtype=np.float64)
        fwd = np.asarray(target, dtype=np.float64) - eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        R = np.stack([right, up2, -fwd], axis=1)
        self.R_t = R.T
        self.t = eye

    def px(self, p_w: np.ndarray):
        pc = self.R_t @ (p_w - self.t)          # 相机系：-Z 前
        z = -pc[2]
        if z <= 0.05:
            return None
        u = self.K[0, 0] * (pc[0] / z) + self.K[0, 2]
        v = self.K[1, 1] * (-pc[1] / z) + self.K[1, 2]
        return (float(u), float(v))


def arrow(draw, p0, p1, color, width=4):
    if p0 is None or p1 is None:
        return
    draw.line([p0, p1], fill=color, width=width)
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(vx, vy)
    if n < 1e-3:
        return
    vx, vy = vx / n, vy / n
    for s in (1, -1):
        hx = p1[0] - 12 * vx + s * 6 * vy
        hy = p1[1] - 12 * vy - s * 6 * vx
        draw.line([p1, (hx, hy)], fill=color, width=width)


def main():
    # device 必须跟 AppLauncher 对齐：渲染作业用 --device cuda:1 选卡时
    # （CVD 会触发 fabric interop device-assert，不能用），cfg 若停在默认
    # cuda:0 会造成 PhysX 数据(cuda:1) 与索引张量(cuda:0) 跨卡 RuntimeError
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device,
                           yaml_name=args.env_yaml, coordinator=True)
    cfg.scene.env_spacing = 3.5
    cfg.enable_viz_camera = True
    # omni.physx.fabric 在"带相机 + 本机多 kit 折腾后"非确定性 device-assert
    # （2026-08-12 凌晨 6/7 次启动中毒）；但 GPU 物理下关 fabric 会断掉
    # 物理位姿→渲染同步：画面里机器人永远停在出生姿态（2026-08-13 定案，
    # 历史所有"手臂不动"视频的根因）。默认仍关（保留 A 的规避），录真视频
    # 必须 --fabric 且用 --device cuda:N 选卡（CVD 才是 assert 触发面）。
    cfg.sim.use_fabric = bool(args.fabric)
    if not args.plain_scene:
        # B3 复刻视觉层（Round 119）：机器人换 *_display 兄弟件（红 JAKA+银筒段
        # +黑 F2 手；v3 官方资产无兄弟件自动跳过），白盒桌隐藏、铝型材复刻桌+
        # 垫板顶上。纯视觉：物理桌隐形照撞，margin/安全栈逐位不变。
        from safeduo.viz.replica_visuals import remap_display_usd
        n_remap = remap_display_usd(cfg)
        print(f"REPLICA_VISUALS display remap: {n_remap}/4 arms", flush=True)
    env = DuoEnv(cfg)
    if not args.plain_scene:
        from safeduo.viz.replica_visuals import spawn_replica_visuals
        spawn_replica_visuals(env)
    cam = env._viz_cam  # 相机在 _setup_scene 建（play 前），此时已完成初始化
    origins = env.scene.env_origins.cpu().numpy()
    center = origins.mean(axis=0) + np.array([0.0, 0.0, 0.8])
    # 单 env 拉近取景（动作可辨认）；多 env 用远机位覆盖全场
    eye = center + (np.array([1.9, -3.4, 2.0]) if env.num_envs == 1
                    else np.array([3.2, -6.5, 4.2]))
    if args.scenario:
        # 强制单一冲突族：改 mix 权重后 set_progress 重算，reset 时按新权重
        # 重采 assignment（不动 env 代码，与训练路径同一 ConflictMixSource）
        src = env._delta_src
        assert args.scenario in src.names[1:], \
            f"unknown scenario {args.scenario}, choices={src.names[1:]}"
        src.cfg["mix"] = {"l1": 0.0, "l2": 1.0}
        src.cfg["scenarios"] = {args.scenario: 1.0}
        src.cfg["stages"] = []
        src.set_progress(0.0)
    policy = None
    if args.checkpoint:
        from safeduo.eval.block1_harness import ActorMLP  # 纯 torch，可安全导入
        policy = ActorMLP.from_checkpoint(args.checkpoint).to(env.device)
    obs, _ = env.reset()
    # 用 API 原生 look-at（set_world_poses_from_view）。历代"白屏视频"的真正病根：
    # 手写 look_at 产出 opengl 约定四元数，set_world_poses 默认按 ROS 约定解读
    # -> 相机翻转朝天只剩 dome 光；fabric/CPU 物理均为替罪羊（2026-08-12 A4 定案）
    cam.set_world_poses_from_view(
        torch.tensor(eye, device=env.device, dtype=torch.float32).unsqueeze(0),
        torch.tensor(center, device=env.device, dtype=torch.float32).unsqueeze(0))
    out_dir = Path.home() / "safeduo" / "artifacts" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        tag = args.scenario or "mix"
        out = out_dir / f"v2_policy_{tag}_{date.today().strftime('%Y%m%d')}.mp4"
    elif args.alpha1:
        tag = args.scenario or "mix"
        out = out_dir / f"v2_passthrough_{tag}_{date.today().strftime('%Y%m%d')}.mp4"
    else:
        out = out_dir / f"v2_proto_{date.today().strftime('%Y%m%d')}.mp4"
    import imageio.v2 as imageio
    # 逐帧落盘：被 timeout/崩溃收割也能留下可播放的部分视频（攒内存最后一次
    # 写的旧方案在 kit 挂死/超时下颗粒无收），且每 100 帧打进度
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=7)
    n_written = 0
    proj = None
    ee_jac = {}
    viol_total = 0          # A8-W8 HUD: env-0 cumulative violation steps
    for t in range(args.steps):
        ne = env.num_envs
        if policy is not None:
            # 真策略：obs -> actor 均值（确定性），与 block1_harness PolicyDriver 同口径
            with torch.no_grad():
                act = policy(obs["policy"].to(env.device)).clamp(-1.0, 1.0)
            alpha = (act[:, :4] + 1.0) * 0.5
            p = act[:, 4]
        elif args.alpha1:
            # 直通演示：delta 全量放行，靠 damper 兜底拦停（arms 大幅动作）
            alpha = torch.ones(ne, 4, device=env.device)
            p = torch.zeros(ne, device=env.device)
            act = torch.cat([alpha * 2 - 1, p.unsqueeze(-1)], dim=-1)
        else:
            # 脚本化 alpha/p：各臂错相位振荡 + p 全程扫摆
            ph = 2 * math.pi * t / 150
            alpha = torch.tensor([[0.5 + 0.5 * math.sin(ph + i * math.pi / 2)
                                   for i in range(4)]], device=env.device).repeat(ne, 1)
            p = torch.full((ne,), math.sin(ph / 2), device=env.device)
            act = torch.cat([alpha * 2 - 1, p.unsqueeze(-1)], dim=-1)
        obs, _, _, _, _ = env.step(act)
        # 手工注册的相机不会触发 step 内 sim.render()（rtx_sensors 标志只在
        # InteractiveScene 构造期设置）：不显式渲染+强制刷新的话 annotator
        # 永远停在初始化帧 -> 历史所有视频机器人冻结的根因（2026-08-13）
        env.sim.render()
        cam.update(env.step_dt, force_recompute=True)
        if proj is None:
            proj = Projector(cam, eye, center)
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb[..., :3])
        draw = ImageDraw.Draw(img)
        c = env._step_cache
        state = env.scene_state()
        for e in range(env.num_envs):
            for ai, arm in enumerate(ARM_KEYS):
                art = env._arms[arm]
                jid = env._joint_idx[arm]
                # EE 雅可比：把 cmd/exec 关节 delta 投到笛卡尔（借 provider 的球心雅可比）
                body = torch.full((env.num_envs, 1), env._ee_idx[arm], device=env.device,
                                  dtype=torch.long)
                jc = env._provider._sphere_jacobian(arm, body,
                                                    torch.zeros(env.num_envs, 1, 3, device=env.device))
                v_cmd = torch.einsum("cd,d->c", jc[e, 0],
                                     c["cmd"].delta_q[arm][e]).cpu().numpy()
                v_exe = torch.einsum("cd,d->c", jc[e, 0],
                                     c["exec"].delta_q[arm][e]).cpu().numpy()
                ee = (state.ee_pos[arm][e].cpu().numpy() + origins[e])
                scale = 6.0
                p0 = proj.px(ee)
                arrow(draw, p0, proj.px(ee + v_cmd * scale), (255, 210, 40))
                arrow(draw, p0, proj.px(ee + v_exe * scale), (60, 230, 90))
                # 基座 alpha 着色
                bp = np.array(env.cfg.base_poses[arm][0]) + origins[e]
                pb = proj.px(bp + np.array([0, 0, 0.25]))
                if pb:
                    a = float(alpha[e, ai])
                    col = (int(255 * (1 - a)), int(220 * a), 40)
                    r = 7
                    draw.ellipse([pb[0] - r, pb[1] - r, pb[0] + r, pb[1] + r], fill=col)
        # p 让行条 + 文本
        pv = float(p[0])
        cx = W // 2
        draw.rectangle([cx - 200, 20, cx + 200, 44], outline=(255, 255, 255), width=2)
        px0, px1 = sorted((cx, cx + int(pv * 198)))  # p<0 时坐标反转，PIL 要求 x1>=x0
        draw.rectangle([px0, 22, px1, 42],
                       fill=(80, 160, 255) if pv >= 0 else (255, 140, 80))
        mm = float(env._last_out.min_margin["cross"].min())
        draw.text((cx - 195, 50), f"p={pv:+.2f} (+1: U yields)  min_cross={mm:.3f}m  t={t}",
                  fill=(255, 255, 255))
        # A8-W8 HUD second line: per-class margins + alpha values + violation
        # counter (env 0), red banner while a violation step is active
        mm_by = {k: float(env._last_out.min_margin[k][0])
                 for k in ("table", "self_F", "self_U")}
        viol_now = bool(c["violation"][0])
        viol_total += int(viol_now)
        al0 = [round(float(alpha[0, i]), 2) for i in range(4)]
        draw.text((cx - 195, 66),
                  f"selfF={mm_by['self_F']:+.3f} selfU={mm_by['self_U']:+.3f} "
                  f"table={mm_by['table']:+.3f}  alpha={al0}  viol={viol_total}",
                  fill=(255, 255, 255))
        if viol_now:
            draw.rectangle([2, 2, W - 3, H - 3], outline=(255, 40, 40), width=6)
            draw.text((cx - 60, 84), "VIOLATION", fill=(255, 40, 40))
        draw.text((20, 20), "cmd", fill=(255, 210, 40))
        draw.text((20, 40), "exec", fill=(60, 230, 90))
        writer.append_data(np.asarray(img))
        n_written += 1
        if n_written % 100 == 0:
            cn = torch.cat([c["cmd"].delta_q[a][0] for a in ARM_KEYS]).norm()
            en = torch.cat([c["exec"].delta_q[a][0] for a in ARM_KEYS]).norm()
            print(f"VIDEO_PROGRESS {n_written}/{args.steps} "
                  f"cmd={cn:.4f} exec={en:.4f} "
                  f"alpha={[round(float(v), 2) for v in alpha[0]]} p={float(p[0]):+.2f} "
                  f"mm_cross={mm:.3f}", flush=True)
    writer.close()
    print(f"VIDEO_DONE {out} frames={n_written}", flush=True)


if __name__ == "__main__":
    main()
    # A8-W8: hard-exit instead of app.close() -- kit exit hangs for 10-25 min
    # after VIDEO_DONE (known disease, same fix as record_s0_demo / probes);
    # the mp4 writer is already closed inside main().
    import os
    os._exit(0)
