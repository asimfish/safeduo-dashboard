"""任务轨迹规划器接口（R23, 2026-08-28）：默认 IK+插值，cuRobo 留接口。

为什么：owner 建议 S9 抓取修复用 cuRobo（plan_grasp 能生成抓取手势与
偏移）。本仓库 v5/v7 技能库其实已有 cuRobo 服务端管线（skill_plan_server
/ skill_plan_server_v7：design 产 waypoints json → 服务器 curobo_t1 venv
plan_cspace 连关节段 → compose 缝合），但 S9 任务是短行程 + 逐步硬门槛
margin 审计，IK+cosine 插值已经够用，且两张 5090 正在跑关键训练（a24/
a24b），现在不做 GPU 规划。所以把"waypoints → 分段关节路径"的产段者抽成
接口：默认实现返回 None（compose_skill 走内置 cosine-ease 回退，行为与
现状逐位一致）；cuRobo 实现是带完整落地说明的存根，GPU 空闲后接上即可。

远端环境事实（2026-08-28 查证）：
  - conda env `safeduo`（训练/录像用）：没有 cuRobo，只有 torch 2.7+warp；
  - venv `~/venvs/curobo_t1`：nvidia-curobo 0.8.0（editable 安装自
    ~/safeduo/assets_src/third_party/curobo）—— v5/T1 时代装好的，可直接用。
"""

from __future__ import annotations

import numpy as np


class ArmTrajPlanner:
    """把 design 阶段的 waypoints doc 变成 compose_skill 的 segments dict。

    协议与 compose_skill(doc, segments) 对齐：segments["<arm>_seg<k>"] =
    (M, dof) 稠密关节路径，缺段 = 该段走 cosine-ease 回退。
    """

    name = "base"

    def segments_for(self, doc: dict) -> "dict | None":
        raise NotImplementedError

    def plan_grasp(self, *args, **kwargs):
        raise NotImplementedError


class IkInterpPlanner(ArmTrajPlanner):
    """默认实现：不产分段 —— compose_skill 对 mode=plan 且无段的相位自动
    走 cosine-ease 关节插值（与 S9 现状完全一致；margin 逐步审计兜底）。"""

    name = "ik"

    def segments_for(self, doc: dict) -> "dict | None":
        return None


class CuRoboPlanner(ArmTrajPlanner):
    """cuRobo 实现存根（TODO R23 后续：GPU 空闲后接入）。

    两条落地路径（按工作量从小到大）：

    A. 复用现成的 v7 服务端分段管线（零新代码，只换调用方式）：
       1) 本地/服务器 CPU 跑 design（本模块消费方 task_record_s9 的
          build 会把 waypoints_<skill>.json 落在 --out 目录）；
       2) 服务器上（GPU 空闲时）：
            source ~/safeduo_setup/env.sh
            source ~/venvs/curobo_t1/bin/activate
            CUDA_VISIBLE_DEVICES=<空闲卡> timeout 540 python -m \
                safeduo.delta.skill_plan_server_v7 --wp-dir <out目录>
          产出 segments_<skill>.npz；
       3) compose 时把该 npz 喂给 compose_skill（task_record_s9 build
          --segments <dir> 即可）。
       注意：v7 服务端的 U 臂用 UR5_only.urdf 臂-only 模型、F 臂用
       franka.yml，手都不在规划模型里 —— 手感知碰撞仍靠 compose 后的
       逐步 sphere 审计（与 13 族技能库同一口径）。

    B. plan_grasp 原生抓取规划（owner 建议的最终形态）：
       curobo 0.8.0 的 MotionGen.plan_grasp(goal_poses, ...) 接收抓取位姿
       集（grasp_gen.box_antipodal_grasps 的输出直接够格：flange 位姿 +
       pre-grasp 偏移就是它要的 approach offset），返回 approach+grasp 两
       段轨迹。需要给 F 臂配 fr3+F2 的 curobo robot yml（现 franka.yml 无
       F2 手）；建议照 assets_src/third_party/curobo 里的 franka 配置改
       collision_spheres 指到 assets_real/spheres 的 F2 球包。

    若 curobo_t1 venv 丢失，重装命令（勿在训练期间跑，编译要吃满 CPU/GPU）：
        python -m venv ~/venvs/curobo_t1 && source ~/venvs/curobo_t1/bin/activate
        pip install torch --index-url https://download.pytorch.org/whl/cu128
        cd ~/safeduo/assets_src/third_party/curobo && pip install -e . --no-build-isolation
    """

    name = "curobo"

    def segments_for(self, doc: dict) -> "dict | None":
        raise NotImplementedError(
            "cuRobo 分段规划是服务器 GPU 作业：先 design 产 waypoints json，"
            "再在 GPU 空闲时跑 skill_plan_server_v7（命令见 CuRoboPlanner "
            "docstring / /tmp/r23_grasp_fix_design.md），把产出的 "
            "segments_<skill>.npz 通过 --segments 传给 compose。")


def make_planner(name: str) -> ArmTrajPlanner:
    """按名字取规划器；默认 ik（现状行为）。"""
    if name == "ik":
        return IkInterpPlanner()
    if name == "curobo":
        return CuRoboPlanner()
    raise ValueError(f"unknown planner {name!r} (choices: ik, curobo)")


def load_segments_dir(seg_dir, skill: str) -> "dict | None":
    """从目录读 segments_<skill>.npz（cuRobo 服务端作业的产物）。"""
    from pathlib import Path

    if not seg_dir:
        return None
    sp = Path(seg_dir) / f"segments_{skill}.npz"
    if not sp.exists():
        return None
    return dict(np.load(sp))
