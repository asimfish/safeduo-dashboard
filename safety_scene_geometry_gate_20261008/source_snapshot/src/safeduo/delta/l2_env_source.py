"""Conflict-enriched env delta source: the L2 scenario library packaged so
duo_env can consume it as a third `delta_source` mode (`l2_mix`).

Motivation (STATUS_SERVER 08-12 03:20, A2 finding #2): under the arbitrated
+-0.70 base layout the L1 random stream almost never produces cross-robot
conflicts inside a 10 s episode (eval tube_fraction ~= 0), so the coordinator
was effectively training on self/table avoidance only. This module feeds the
scripted conflict scenario library (delta/l2_scenarios.py) into training with
a configurable L1:L2 curriculum mixture.

Pieces:
- `RealScenePoses`     scene truth loaded from B's assets_src/scene_layout.yaml
                       + configs/duo_env.yaml (base_x_abs override, Round-12
                       arbitration). NOT the stale mirrored defaults in
                       baselines/real_geometry.SceneLayout.
- `EnvEEBackend`       batched analytic FK/EE-jacobians for the real 26-DoF
                       geometry (FR3/UR5e chains from baselines/real_geometry),
                       one FK per arm per step shared by every scenario, with
                       a one-shot UR base-yaw auto-calibration against the
                       env-reported EE positions (the URDF-vs-USD internal yaw
                       ambiguity flagged in STATUS_C W2 #2).
- `CachedEEMapper`     IntentMapper that maps EE-space scenario intents to
                       joint deltas through the cached damped pseudo-inverse.
- `ConflictMixSource`  the DeltaSource duo_env instantiates: per-env
                       assignment to L1 or one of the 8 L2 scenario families,
                       resampled on episode reset from curriculum weights.

Config schema: configs/delta_curriculum.yaml (C2-W3 proposal; @A2 wires
`coordinator.delta_source: l2_mix` in duo_env to construct this class).

Interface contract identical to L1RandomDelta: reset(env_ids, generator) /
sample(state) -> DeltaCmd; pure torch, device-agnostic, Mac-importable.
"""

from __future__ import annotations

import math

import torch

from safeduo.delta._contract_stub import ARM_KEYS, DeltaCmd, DeltaSource, SceneState
from safeduo.delta.l1_random import IntentMapper, L1Params, L1RandomDelta
from safeduo.delta.l2_scenarios import SCENARIOS, WorkspaceSpec, build_scenario


# --------------------------------------------------------------------------
# scene truth
# --------------------------------------------------------------------------

class RealScenePoses:
    """Per-arm base pose + shared-workspace geometry from the layout YAMLs.

    Reads the same files duo_env reads (scene_layout.yaml via duo_env.yaml's
    `scene.layout_yaml`, with `scene.base_x_abs` overriding |x|), so the FK
    frames match what Isaac actually spawns. B convention: F table at +x,
    F bases yaw=pi (facing -x); U at -x, yaw=0 (facing +x).
    """

    def __init__(self, env_yaml: str = "duo_env.yaml"):
        from safeduo.configs import load_config, repo_root

        import yaml as _yaml

        y = load_config(env_yaml)
        with open(repo_root() / y["scene"]["layout_yaml"]) as f:
            layout = _yaml.safe_load(f)
        base_x = y["scene"].get("base_x_abs")
        self.base_pos: dict[str, tuple] = {}
        self.base_yaw: dict[str, float] = {}
        for arm in ARM_KEYS:
            a = layout["robots"]["arms"][arm]
            pos = [float(v) for v in a["base_pos"]]
            if base_x is not None:
                pos[0] = float(base_x) if pos[0] > 0 else -float(base_x)
            self.base_pos[arm] = tuple(pos)
            self.base_yaw[arm] = float(a["base_yaw"])
        ws = layout["workspace"]
        self.ws_center = tuple(float(v) for v in ws["center"])
        self.ws_lo = tuple(float(v) for v in ws["aabb_min"])
        self.ws_hi = tuple(float(v) for v in ws["aabb_max"])
        self.table_top_z = float(layout["tables"]["table_F"]["top_z"])
        # F-side sign along x decides which half-box belongs to which robot
        self._f_sign = 1.0 if self.base_pos["F_L"][0] > 0 else -1.0

    def workspace_spec(self, margin: float = 0.05,
                       f_sign: "float | None" = None) -> WorkspaceSpec:
        """Shared-workspace boxes for the scenario library, split at x=0.

        box_F = the half of B's workspace AABB on the F side; scenarios use
        these to place sweep paths / evasion clamps inside each robot's
        reachable share. z floor is lifted `margin` above the table top so
        scripted targets do not start inside the table.

        f_sign overrides which x half-space belongs to F -- pass the sign of
        the F base x in the *consumer's* frame when rolling scenarios in a
        world whose convention differs from B's (e.g. baselines/real_geometry
        expresses the same scene rotated pi about z, F at -x). The AABB is
        x/y-symmetric so only the split flips.
        """
        lo, hi = self.ws_lo, self.ws_hi
        z_lo = max(lo[2], self.table_top_z + margin)
        sign = self._f_sign if f_sign is None else f_sign
        f_lo_x, f_hi_x = (margin, hi[0]) if sign > 0 else (lo[0], -margin)
        u_lo_x, u_hi_x = (lo[0], -margin) if sign > 0 else (margin, hi[0])
        return WorkspaceSpec(
            center=self.ws_center,
            table_z=self.table_top_z,
            box_F=((f_lo_x, lo[1], z_lo), (f_hi_x, hi[1], hi[2])),
            box_U=((u_lo_x, lo[1], z_lo), (u_hi_x, hi[1], hi[2])),
        )


# --------------------------------------------------------------------------
# batched real-geometry EE backend
# --------------------------------------------------------------------------

class EnvEEBackend:
    """One FK per arm per step -> flange position + point jacobian cache.

    UR yaw ambiguity: the UR URDF carries an internal base_link->
    base_link_inertia yaw of pi and it is unconfirmed whether Isaac's
    ur5e.usd bakes it (STATUS_C W2 @A item #2). On the first sample() with a
    live state we compute the analytic flange for both conventions and keep
    whichever matches the env-reported EE positions; the residuals are kept
    in `calib_report` so the training log can surface a convention drift.
    """

    _YAW_CANDIDATES = (0.0, math.pi)

    def __init__(self, poses: RealScenePoses, device: "str | torch.device" = "cpu"):
        from safeduo.baselines.real_geometry import (
            FR3_FLANGE,
            FR3_JOINTS,
            FR3_LINKS,
            UR5E_FLANGE,
            UR5E_JOINTS,
            UR5E_LINKS,
            ArmKinematics,
        )

        self.poses = poses
        self.device = torch.device(device)
        self._mk_kin = {
            "F": lambda yaw: ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE,
                                           yaw, device=self.device),
            "U": lambda yaw: ArmKinematics(UR5E_JOINTS, UR5E_LINKS, False, UR5E_FLANGE,
                                           yaw, device=self.device),
        }
        # F has no internal-yaw ambiguity; U starts at 0.0 until calibrated
        self.kin = {"F": self._mk_kin["F"](0.0), "U": self._mk_kin["U"](0.0)}
        self.calibrated = False
        self.calib_report: dict = {}
        self._cache_token = -1
        self._cache: dict = {}

    def _flange_and_jac(self, arm: str, q: torch.Tensor,
                        kin=None) -> tuple[torch.Tensor, torch.Tensor]:
        kin = kin or self.kin[arm[0]]
        pos, yaw = self.poses.base_pos[arm], self.poses.base_yaw[arm]
        fko = kin.fk(q, pos, yaw)
        p = fko["t_flange"]
        fidx = torch.tensor([kin.dof], dtype=torch.long, device=q.device)
        J = kin.point_jacobian(fko, p.unsqueeze(1), fidx)[:, 0]      # (N, 3, dof)
        return p, J

    def calibrate(self, state: SceneState) -> None:
        """Pick the UR internal-yaw convention that matches the env's EE."""
        report = {}
        best_yaw, best_err = 0.0, float("inf")
        for yaw in self._YAW_CANDIDATES:
            kin = self._mk_kin["U"](yaw)
            errs = []
            for arm in ("U_L", "U_R"):
                p, _ = self._flange_and_jac(arm, state.q[arm], kin=kin)
                errs.append((p - state.ee_pos[arm]).norm(dim=-1).mean().item())
            err = sum(errs) / len(errs)
            report[f"ur_yaw_{yaw:.2f}"] = err
            if err < best_err:
                best_yaw, best_err = yaw, err
        self.kin["U"] = self._mk_kin["U"](best_yaw)
        for arm in ("F_L", "F_R"):
            p, _ = self._flange_and_jac(arm, state.q[arm])
            report[f"{arm}_resid"] = (p - state.ee_pos[arm]).norm(dim=-1).mean().item()
        report["ur_yaw_chosen"] = best_yaw
        report["ur_resid"] = best_err
        # flange-vs-reported-body offsets (hand mount etc.) land here too; a
        # residual this large means the *convention* is off, not just an offset
        report["suspect"] = bool(best_err > 0.15
                                 or max(report[f"{a}_resid"] for a in ("F_L", "F_R")) > 0.15)
        self.calib_report = report
        self.calibrated = True

    def refresh(self, state: SceneState, token: int) -> None:
        """Recompute the per-arm jacobian cache once per env step."""
        if token == self._cache_token:
            return
        if not self.calibrated:
            self.calibrate(state)
        for arm in ARM_KEYS:
            p, J = self._flange_and_jac(arm, state.q[arm])
            JJt = J @ J.transpose(-1, -2)
            eye = torch.eye(3, device=J.device, dtype=J.dtype).expand_as(JJt)
            self._cache[arm] = {
                "J": J,
                "JJt_inv": torch.linalg.inv(JJt + 1e-2 * eye),
            }
        self._cache_token = token

    def jac(self, arm: str) -> dict:
        return self._cache[arm]


class CachedEEMapper(IntentMapper):
    """JacobianMapper equivalent that reuses the backend's per-step cache.

    Semantics match l1_random.JacobianMapper: v_ee points at the target,
    capped at `speed` (m/s), damped pseudo-inverse to joint space. The
    per-step (JJt + lambda I)^-1 is shared by all 8 scenarios instead of
    re-solving per scenario call.
    """

    def __init__(self, backend: EnvEEBackend, ee_speed: float = 0.25):
        self.backend = backend
        self.ee_speed = ee_speed

    def map(self, arm, state, tgt, dt, speed=None):
        err = tgt - state.ee_pos[arm]
        dist = err.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cap = self.ee_speed if speed is None else torch.as_tensor(
            speed, dtype=err.dtype, device=err.device).reshape(-1, 1)
        v_ee = err / dist * torch.minimum(dist / dt, cap * torch.ones_like(dist))
        c = self.backend.jac(arm)
        qd = c["J"].transpose(-1, -2) @ (c["JJt_inv"] @ v_ee.unsqueeze(-1))
        return qd.squeeze(-1) * dt

    def sample_waypoint(self, arm, state, generator):
        lo = torch.tensor(self.backend.poses.ws_lo, device=state.device)
        hi = torch.tensor(self.backend.poses.ws_hi, device=state.device)
        n = state.ee_pos[arm].shape[0]
        u = torch.rand(n, 3, device=state.device, generator=generator)
        return lo + u * (hi - lo)


# --------------------------------------------------------------------------
# curriculum mixture source
# --------------------------------------------------------------------------

DEFAULT_CURRICULUM = {
    "mix": {"l1": 0.5, "l2": 0.5},
    "scenarios": {name: 1.0 for name in SCENARIOS},
    "split": "train",
    "n_variants": 100,
    "l1_params": {},
    "ee_speed_default": 0.25,
    "stages": None,
    # ---- T2/T3 extensions (2026-08-13, all default-off: behavior identical
    #      to the pre-T2 source unless a curriculum yaml opts in) ----------
    # "workspace": the L1 sub-source roams EE-space waypoints over the full
    # shared workspace (delta.l1_workspace.WorkspaceRoamMapper, sharing this
    # mix's FK backend); "joint" keeps the stock JointSpaceMapper.
    "l1_mapper": "joint",
    # amp curriculum (the spec S1-T3 "amp 课程进 l1 漫游", training side):
    # list of {until_frac, amp_max} consumed by set_progress -- updates BOTH
    # the mixture output clamp and the L1 sub-source's own amp cap. None =
    # constant amp_max from the constructor (pre-T2 behavior).
    "amp_stages": None,
    # optional skill-replay family in the mixture (spec S1-T1/T2 as training
    # traffic): {"dir": <npz dir>, "tier": 0-3, "weight": w, "loop": true}.
    # Weight is relative to mix.l1/mix.l2 before normalization; stages may
    # override it with a "skill" key.
    "skills": None,
    # scene geometry family for the FK backend + roam boxes (R15 v7 wiring,
    # 2026-08-20): "v5" keeps EnvEEBackend + stock WorkspaceRoamMapper
    # (pre-v7 behavior, bit-exact); "v7" swaps in EnvEEBackendV7 (URDF UR5
    # chain) + make_roam_mapper_v7 boxes. env_yaml must point at the matching
    # duo_env yaml (duo_env.py forwards coordinator.delta_env_yaml).
    "geometry": "v5",
    # R25 l1_full 覆盖族(2026-08-28,默认 None = 与既有配方逐位一致):
    # {"weight": w, ...其余键为 L1FullParams 字段覆盖}。w 与 mix.l1 /
    # mix.l2 / skills.weight 同池归一化;stages 可用 "l1_full" 键逐段改
    # 权重。v7 几何专属(delta/l1_coverage.py,盒由 full_boxes_v7 标定);
    # 该族自持 FK 后端(需要旋转雅可比缓存,共享后端没有),每步多一次
    # 26-DoF 解析 FK——纯 CPU 张量,量级远小于物理步,opt-in 才付费。
    "l1_full": None,
}


def load_curriculum_cfg(name: str = "delta_curriculum.yaml") -> dict:
    from safeduo.configs import load_config

    cfg = dict(DEFAULT_CURRICULUM)
    cfg.update(load_config(name) or {})
    return cfg


class ConflictMixSource(DeltaSource):
    """Per-env mixture of L1 random + the 8 scripted L2 conflict families.

    Assignment (which sub-source drives which env) is resampled per episode
    reset from the current curriculum weights; every sub-source stays batched
    over all N envs and the outputs are gathered by assignment, so the cost
    is one cheap tensor pass per family plus one shared FK per arm per step.

    `set_progress(frac)` moves through `stages` (list of {until_frac, l1, l2,
    scenarios?}); without stages the flat `mix` weights apply. New weights
    take effect for envs at their next reset (in-flight episodes keep their
    scenario -- no mid-episode switching).
    """

    L1_INDEX = 0

    def __init__(self, n_envs: int, cfg: "dict | None" = None,
                 device: "str | torch.device" = "cpu",
                 mapper: "IntentMapper | None" = None,
                 ws: "WorkspaceSpec | None" = None,
                 dof_of: "dict | None" = None,
                 amp_max: float = 0.015,
                 env_yaml: str = "duo_env.yaml"):
        self.n = n_envs
        self.cfg = {**DEFAULT_CURRICULUM, **(cfg or {})}
        self.device = torch.device(device)
        self.amp_max = float(amp_max)
        self.gen: "torch.Generator | None" = None
        self.backend: "EnvEEBackend | None" = None
        geometry = str(self.cfg.get("geometry", "v5"))
        if mapper is None:
            poses = RealScenePoses(env_yaml)
            if geometry == "v7":
                from safeduo.delta.l1_workspace_v7 import EnvEEBackendV7

                self.backend = EnvEEBackendV7(poses, device=device)
            else:
                self.backend = EnvEEBackend(poses, device=device)
            mapper = CachedEEMapper(self.backend,
                                    ee_speed=float(self.cfg["ee_speed_default"]))
            ws = ws or poses.workspace_spec()
        assert ws is not None, "ws required when injecting a custom mapper"
        l1p = L1Params(**{**{"amp_max": self.amp_max}, **self.cfg["l1_params"]})
        l1_mapper = None
        if str(self.cfg.get("l1_mapper", "joint")) == "workspace":
            # workspace-spanning roam for the L1 family (T2-C): share this
            # mix's backend so the per-step FK refresh happens exactly once
            assert self.backend is not None, \
                "l1_mapper=workspace needs the real-geometry backend " \
                "(cannot combine with an injected toy mapper)"
            if geometry == "v7":
                from safeduo.delta.l1_workspace_v7 import make_roam_mapper_v7

                l1_mapper = make_roam_mapper_v7(
                    self.backend, ee_speed=float(self.cfg["ee_speed_default"]))
            else:
                from safeduo.delta.l1_workspace import WorkspaceRoamMapper

                l1_mapper = WorkspaceRoamMapper(
                    self.backend, ee_speed=float(self.cfg["ee_speed_default"]))
        self.names = ["l1"] + list(SCENARIOS.keys())
        self._l1_src = L1RandomDelta(n_envs, params=l1p, mapper=l1_mapper,
                                     device=device, dof_of=dof_of)
        self.sources: list[DeltaSource] = [self._l1_src]
        for name in SCENARIOS:
            self.sources.append(build_scenario(
                name, n_envs, mapper, split=self.cfg["split"],
                n_variants=int(self.cfg["n_variants"]), ws=ws,
                device=device, dof_of=dof_of))
        self._skill_weight = 0.0
        if self.cfg.get("skills"):
            sk = dict(self.cfg["skills"])
            from safeduo.delta.skill_replay import (
                SkillNoiseParams,
                SkillReplayDelta,
                load_library,
            )

            self.sources.append(SkillReplayDelta(
                n_envs, load_library(sk["dir"]),
                params=SkillNoiseParams.tier(int(sk.get("tier", 1))),
                device=device, dof_of=dof_of,
                loop=bool(sk.get("loop", True))))
            self.names.append("skill")
            self._skill_weight = float(sk.get("weight", 0.0))
        # R25 l1_full 覆盖族(键缺省 = 本分支整体不生效,零漂移)。放在
        # skill 之后追加:既有配方里 skill 的家族下标(1+n_scen)不动。
        self._l1full_weight = 0.0
        self._l1full_src = None
        if self.cfg.get("l1_full"):
            assert geometry == "v7", \
                "l1_full 族只标定了 v7 几何(delta_curriculum yaml 置 geometry: v7)"
            assert dof_of is None, \
                "l1_full 族用真实 26-DoF FK,不支持 toy dof_of 注入"
            lf = dict(self.cfg["l1_full"])
            from safeduo.delta.l1_coverage import make_l1_full_v7

            self._l1full_weight = float(lf.pop("weight", 0.0))
            self._l1full_src = make_l1_full_v7(
                n_envs, amp_max=self.amp_max, device=device,
                cfg=lf, env_yaml=env_yaml)
            self.sources.append(self._l1full_src)
            self.names.append("l1_full")
        self.assignment = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self._progress = 0.0
        self._weights = self._weights_for(self._progress)
        self._apply_amp(self._amp_for(self._progress))
        self._step_token = 0

    # ---- curriculum weights ----

    def _weights_for(self, frac: float) -> torch.Tensor:
        cfg = self.cfg
        mix, scen_w = cfg["mix"], cfg["scenarios"]
        skill_w = self._skill_weight
        l1full_w = self._l1full_weight
        if cfg.get("stages"):
            for st in cfg["stages"]:
                if frac <= float(st["until_frac"]) + 1e-9:
                    break
            else:
                st = cfg["stages"][-1]
            mix = {"l1": st["l1"], "l2": st["l2"]}
            scen_w = st.get("scenarios", scen_w)
            skill_w = float(st.get("skill", skill_w))
            l1full_w = float(st.get("l1_full", l1full_w))
        n_scen = len(SCENARIOS)
        w = torch.zeros(len(self.sources))
        w[self.L1_INDEX] = float(mix["l1"])
        sw = torch.tensor([float(scen_w.get(n, 0.0))
                           for n in self.names[1:1 + n_scen]])
        if sw.sum() > 0:
            w[1:1 + n_scen] = float(mix["l2"]) * sw / sw.sum()
        # 追加族按名字定位(R25 起 skill 后可能还有 l1_full;既有配方里
        # "skill" 恰在 1+n_scen 位,按名索引与旧的按长度判断逐位等价)
        if "skill" in self.names:
            w[self.names.index("skill")] = skill_w
        if "l1_full" in self.names:
            w[self.names.index("l1_full")] = l1full_w
        total = w.sum()
        assert total > 0, "curriculum weights are all zero"
        return (w / total).to(self.device)

    def _amp_for(self, frac: float) -> "float | None":
        """amp_stages lookup (None = no amp curriculum configured)."""
        stages = self.cfg.get("amp_stages")
        if not stages:
            return None
        for st in stages:
            if frac <= float(st["until_frac"]) + 1e-9:
                return float(st["amp_max"])
        return float(stages[-1]["amp_max"])

    def _apply_amp(self, amp: "float | None") -> None:
        """Move BOTH the mixture clamp and the L1 sub-source's own cap: the
        sub-source clamps its output at p.amp_max before the mixture gather,
        so raising only the outer clamp would leave l1 stuck at the ctor amp.
        (The skill family keeps its own SkillNoiseParams.amp_max=0.06 hard
        cap -- skill replay is bounded traffic by design, spec S1-T2.)"""
        if amp is None:
            return
        self.amp_max = float(amp)
        self._l1_src.p.amp_max = float(amp)
        if self._l1full_src is not None:   # R25 覆盖族与 l1 同步吃 amp 课程
            self._l1full_src.p.amp_max = float(amp)

    def set_progress(self, frac: float) -> None:
        """Training progress in [0,1]; A2's train loop calls this per iter.
        Advances the family mixture (stages) AND the amp curriculum
        (amp_stages, the spec S1-T3 training-side wiring)."""
        self._progress = max(0.0, min(1.0, float(frac)))
        self._weights = self._weights_for(self._progress)
        self._apply_amp(self._amp_for(self._progress))

    @property
    def mix_fractions(self) -> dict:
        """Current assignment histogram (for logging: is L2 actually live?)."""
        counts = torch.bincount(self.assignment, minlength=len(self.sources)).float()
        return {n: (c / self.n).item() for n, c in zip(self.names, counts)}

    # ---- DeltaSource contract ----

    def reset(self, env_ids: torch.Tensor, generator: "torch.Generator | None" = None) -> None:
        if generator is not None:
            self.gen = generator
        ids = env_ids.to(self.device)
        if ids.numel():
            draw = torch.multinomial(
                self._weights.expand(ids.numel(), -1), 1,
                replacement=True, generator=self.gen).squeeze(-1)
            self.assignment[ids] = draw
        for src in self.sources:
            src.reset(env_ids, generator)

    def sample(self, state: SceneState) -> DeltaCmd:
        self._step_token += 1
        if self.backend is not None:
            self.backend.refresh(state, self._step_token)
        outs = [src.sample(state) for src in self.sources]
        result = {}
        for arm in ARM_KEYS:
            stack = torch.stack([o.delta_q[arm] for o in outs], dim=0)  # (K, N, dof)
            idx = self.assignment.view(1, -1, 1).expand(1, -1, stack.shape[-1])
            d = stack.gather(0, idx).squeeze(0)
            result[arm] = d.clamp(-self.amp_max, self.amp_max)
        return DeltaCmd(delta_q=result)
