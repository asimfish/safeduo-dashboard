"""Per-skill task-completion criteria in EE (flange) space -- R5 first half (T4).

The skill-replay evaluation so far only measures SAFETY (violation steps);
owner 2026-08-16 requires the orchestrated-trajectory data to demonstrably
COMPLETE its task ("???????????"). Master report section 7/9 R5
acceptance: per-skill EE arrival criteria + success rate into block1.

Criterion definition (per skill, parameters in configs/skill_success.yaml):

1. **Ordered waypoint arrival**: the waypoints_<skill>.json design artifact
   stores the joint-space waypoint of every arm at every phase boundary.
   Those are FK'd (v5 scene caliber, same kinematics/layout family the
   library's hard-gate audit used: FR3+JAKA ArmKinematics + SceneLayoutV5
   base poses, flange = EE) into per-arm EE key waypoints. Success requires
   every arm to pass within ``radius_m`` of each of its key waypoints in
   order. Order tolerance: waypoint k may be matched up to ``order_tol_s``
   BEFORE the match time of waypoint k-1 (greedy sequential matching with a
   backward window), so benign local reorderings under time-warp noise do
   not fail the run. Consecutive waypoints closer than ``dedupe_eps_m`` in
   EE space are collapsed (hold phases produce duplicated boundaries);
   static arms therefore reduce to a single trivially-hit waypoint.

2. **Handover meet** (skills with ``handover_pairs``): the giver/receiver EE
   pair must come within ``handover_dist_m`` at some point of the run (the
   transfer window is where the trajectory's pair distance dips, so the
   minimum over the run IS the transfer-window reading); for multi-meet
   chains (tool_pass_chain) the pairs must dip below threshold in listed
   order (first-crossing times non-decreasing, same order tolerance).

FK caliber note: criteria are DEFINED and CALIBRATED in this file's v5
flange FK. The endurance wire-in feeds executed joint states through the
same FK (EETraceTracker), so offline calibration and online judgment share
one caliber; Isaac's own EE body frame is never mixed in.

Offline verification entry point (Mac, pure torch, no Isaac):

    PYTHONPATH=src python -m safeduo.eval.skill_success \
        --skill-dir artifacts/skill_trajs --tiers 0 1 2 3 --seeds 25

replays every skill through SkillReplayDelta at each noise tier, integrates
the emitted deltas from the birth pose (exactly what a passthrough env
executes), and prints/writes the per-skill success table. Acceptance line:
tier-0 must be 13/13.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import yaml

from safeduo.baselines.real_geometry import (
    FR3_FLANGE_V4,
    FR3_JOINTS,
    FR3_LINKS,
    JAKA_AXES,
    JAKA_FLANGE,
    JAKA_JOINTS,
    JAKA_LINKS,
    ArmKinematics,
    SceneLayoutV5,
)
from safeduo.safety.types import ARM_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "src" / "safeduo" / "configs" / "skill_success.yaml"
DEFAULT_SKILL_DIR = REPO_ROOT / "artifacts" / "skill_trajs"


# --------------------------------------------------------------------------
# v5 EE (flange) forward kinematics
# --------------------------------------------------------------------------

class V5SkillFK:
    """Flange-position FK on the v5 scene (the library's audit caliber).

    Same kinematic tables and base placement make_v5_provider uses
    (real_geometry constants; FR3_FLANGE_V4 = true fr3_link8 frame, JAKA
    tool frame = link6), without the sphere machinery -- skill success only
    needs EE positions.
    """

    def __init__(self, device: "str | torch.device" = "cpu"):
        self.device = torch.device(device)
        self.layout = SceneLayoutV5()
        self.kin = {
            "F": ArmKinematics(FR3_JOINTS, FR3_LINKS, True, FR3_FLANGE_V4,
                               0.0, device=device),
            "U": ArmKinematics(JAKA_JOINTS, JAKA_LINKS, True, JAKA_FLANGE,
                               0.0, device=device, axes=JAKA_AXES),
        }

    def ee_pos(self, q_by_arm: dict) -> dict:
        """{arm: (..., dof)} joint positions -> {arm: (..., 3)} flange pos.

        Leading dims are free (batched FK: (N,), (T, N), ...); env-local
        frame (base poses from SceneLayoutV5, no env_origins involved).
        """
        out = {}
        for arm, q in q_by_arm.items():
            q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
            lead = q.shape[:-1]
            q2 = q.reshape(-1, q.shape[-1])
            pos, yaw = self.layout.base_pose(arm)
            fko = self.kin[arm[0]].fk(q2, pos, yaw)
            out[arm] = fko["t_flange"].reshape(*lead, 3)
        return out


# --------------------------------------------------------------------------
# the judge
# --------------------------------------------------------------------------

def load_success_config(path: "str | Path | None" = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if not cfg.get("skills"):
        raise ValueError(f"skill_success config {path} lists no skills")
    return cfg


class SkillSuccessJudge:
    """Loads waypoints_<skill>.json + skill_success.yaml, judges trajectories.

    Precomputes per skill: per-arm EE key waypoints (FK of the phase-boundary
    joint waypoints, consecutive near-duplicates collapsed) and the handover
    pair list. evaluate()/evaluate_ee() judge ONE env's trajectory and return
    {success, per_waypoint_hits, miss_reason, ...}.
    """

    def __init__(self, skill_dir: "str | Path" = DEFAULT_SKILL_DIR,
                 config_path: "str | Path | None" = None,
                 fk: "V5SkillFK | None" = None,
                 device: "str | torch.device" = "cpu"):
        self.fk = fk or V5SkillFK(device)
        cfg = load_success_config(config_path)
        self.defaults = dict(cfg.get("defaults") or {})
        self.skill_dir = Path(skill_dir)
        self.skills: dict = {}
        for name, scfg in cfg["skills"].items():
            merged = {**self.defaults, **(scfg or {})}
            wp_path = self.skill_dir / f"waypoints_{name}.json"
            wp = json.loads(wp_path.read_text())
            q_wp = {a: torch.tensor(wp["waypoints"][a], dtype=torch.float32)
                    for a in ARM_KEYS}
            ee_wp = self.fk.ee_pos(q_wp)                      # arm -> (W, 3)
            eps = float(merged.get("dedupe_eps_m", 0.01))
            kept, kept_idx = {}, {}
            for a in ARM_KEYS:
                idx = [0]
                for k in range(1, ee_wp[a].shape[0]):
                    if float((ee_wp[a][k] - ee_wp[a][idx[-1]]).norm()) > eps:
                        idx.append(k)
                kept_idx[a] = idx
                kept[a] = ee_wp[a][idx]
            self.skills[name] = {
                "cfg": merged,
                "dt": float(wp["dt"]),
                "phases": [p["name"] for p in wp["phases"]],
                "ee_wp": kept,                                # arm -> (Wk, 3)
                "wp_idx": kept_idx,                           # arm -> [orig k]
                "q_wp": q_wp,                                 # joint truth
            }

    # ---- single-env judgment -------------------------------------------------

    def _boundary_label(self, name: str, orig_idx: int) -> str:
        phases = self.skills[name]["phases"]
        return "start" if orig_idx == 0 else f"end_of_{phases[orig_idx - 1]}"

    def evaluate_ee(self, ee_traj_by_arm: dict, skill_name: str,
                    dt: "float | None" = None) -> dict:
        """{arm: (T, 3)} executed EE trace of ONE env -> judgment dict."""
        if skill_name not in self.skills:
            return {"skill": skill_name, "success": False,
                    "per_waypoint_hits": {}, "waypoint_frac": 0.0,
                    "miss_reason": f"skill {skill_name!r} not in config"}
        sk = self.skills[skill_name]
        cfg = sk["cfg"]
        dt = float(dt if dt is not None else sk["dt"])
        radius = float(cfg["radius_m"])
        tol = int(round(float(cfg.get("order_tol_s", 1.0)) / dt))
        hits, min_dists = {}, {}
        miss_reason = None
        n_hit = n_total = 0
        for a in ARM_KEYS:
            ee = torch.as_tensor(ee_traj_by_arm[a], dtype=torch.float32)
            d = torch.cdist(ee, sk["ee_wp"][a])               # (T, Wk)
            hit_mask = d <= radius
            arm_hits, cursor = [], 0
            for w in range(d.shape[1]):
                lo = max(0, cursor - tol)
                cand = hit_mask[lo:, w].nonzero()
                n_total += 1
                if cand.numel():
                    t = lo + int(cand[0])
                    arm_hits.append(t)
                    cursor = max(cursor, t + 1)
                    n_hit += 1
                else:
                    arm_hits.append(-1)
                    if miss_reason is None:
                        orig = sk["wp_idx"][a][w]
                        miss_reason = (
                            f"{a} wp{orig} ({self._boundary_label(skill_name, orig)}) "
                            f"unreached: min EE dist "
                            f"{float(d[lo:, w].min()):.3f} m > radius {radius} m")
            hits[a] = arm_hits
            min_dists[a] = [round(float(d[:, w].min()), 4)
                            for w in range(d.shape[1])]
        result = {
            "skill": skill_name,
            "per_waypoint_hits": hits,
            "waypoint_min_dist": min_dists,
            "waypoint_frac": (n_hit / n_total) if n_total else 0.0,
        }
        # handover meet criterion
        pairs = cfg.get("handover_pairs") or []
        if pairs:
            hd = cfg["handover_dist_m"]
            # scalar or per-pair list (tool_pass_chain's two meets differ
            # geometrically: pair idle distances 0.345 vs 0.825 m)
            h_dists = list(hd) if isinstance(hd, (list, tuple)) \
                else [float(hd)] * len(pairs)
            assert len(h_dists) == len(pairs), \
                f"handover_dist_m list length {len(h_dists)} != {len(pairs)} pairs"
            first_below, mins = [], {}
            for (a, b), h_dist in zip(pairs, h_dists):
                ea = torch.as_tensor(ee_traj_by_arm[a], dtype=torch.float32)
                eb = torch.as_tensor(ee_traj_by_arm[b], dtype=torch.float32)
                dd = (ea - eb).norm(dim=-1)                   # (T,)
                mins[f"{a}-{b}"] = round(float(dd.min()), 4)
                below = (dd <= h_dist).nonzero()
                if below.numel():
                    first_below.append(int(below[0]))
                else:
                    first_below.append(-1)
                    if miss_reason is None:
                        miss_reason = (
                            f"handover pair {a}-{b} never met: min EE dist "
                            f"{float(dd.min()):.3f} m > {h_dist} m")
            if miss_reason is None:
                for i in range(1, len(first_below)):
                    if first_below[i] < first_below[i - 1] - tol:
                        a, b = pairs[i]
                        miss_reason = (
                            f"handover chain out of order: pair {a}-{b} met at "
                            f"step {first_below[i]} before pair "
                            f"{pairs[i-1][0]}-{pairs[i-1][1]} "
                            f"(step {first_below[i-1]})")
                        break
            result["handover_min_dist"] = mins
        result["success"] = miss_reason is None
        result["miss_reason"] = miss_reason
        return result

    def evaluate(self, joint_traj_by_arm: dict, skill_name: str,
                 dt: "float | None" = None) -> dict:
        """{arm: (T, dof)} executed joint trace of ONE env -> judgment dict."""
        ee = self.fk.ee_pos({a: joint_traj_by_arm[a] for a in ARM_KEYS})
        return self.evaluate_ee(ee, skill_name, dt=dt)


_DEFAULT_JUDGE: "SkillSuccessJudge | None" = None


def default_judge() -> SkillSuccessJudge:
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        _DEFAULT_JUDGE = SkillSuccessJudge()
    return _DEFAULT_JUDGE


def evaluate_skill_success(joint_traj_by_arm: dict, skill_name: str,
                           judge: "SkillSuccessJudge | None" = None,
                           dt: "float | None" = None) -> dict:
    """Core API (spec R5): per-arm executed joint trajectory -> success dict.

    joint_traj_by_arm: {arm: (T, dof)} for ONE env/run. Returns at least
    {success: bool, per_waypoint_hits: {arm: [step|-1 ...]},
    miss_reason: str|None} plus diagnostics (waypoint_frac,
    waypoint_min_dist, handover_min_dist).
    """
    return (judge or default_judge()).evaluate(joint_traj_by_arm, skill_name,
                                               dt=dt)


# --------------------------------------------------------------------------
# endurance wire-in: per-step executed-EE collection + row columns
# --------------------------------------------------------------------------

class EETraceTracker:
    """Collects executed EE traces inside an eval loop (endurance run_window).

    record(q_by_arm) once per post-step with the env's CURRENT joint state
    ({arm: (N, dof)}); EE comes from the same v5 FK the criteria were
    calibrated in and is buffered on CPU ((T, N, 3) per arm ~ 43 MB for a
    600 s x 25 env window -- same order as the margin buffers).
    """

    def __init__(self, fk: "V5SkillFK | None" = None):
        self.fk = fk or V5SkillFK()
        self._buf: dict = {a: [] for a in ARM_KEYS}

    def record(self, q_by_arm: dict) -> None:
        ee = self.fk.ee_pos({a: torch.as_tensor(q_by_arm[a]).detach().cpu()
                             for a in ARM_KEYS})
        for a in ARM_KEYS:
            self._buf[a].append(ee[a])

    def ee_traces(self) -> dict:
        """{arm: (T, N, 3)} collected so far."""
        return {a: torch.stack(v) for a, v in self._buf.items()}


def skill_success_columns(tracker: EETraceTracker, families: list,
                          judge: "SkillSuccessJudge | None" = None,
                          dt: float = 1.0 / 60.0) -> dict:
    """Per-env task-success columns for endurance run rows (only-add contract).

    families: per-env skill name (endurance source_families). Returns
    {task_success: [bool], task_waypoint_frac: [float],
    task_miss_reason: [str]} ready for run_rows_from_margins extra_per_env.
    """
    judge = judge or default_judge()
    traces = tracker.ee_traces()
    n = traces[ARM_KEYS[0]].shape[1]
    assert len(families) == n, f"{len(families)} families != {n} envs"
    cols: dict = {"task_success": [], "task_waypoint_frac": [],
                  "task_miss_reason": []}
    for e in range(n):
        ee = {a: traces[a][:, e] for a in ARM_KEYS}
        r = judge.evaluate_ee(ee, families[e], dt=dt)
        cols["task_success"].append(bool(r["success"]))
        cols["task_waypoint_frac"].append(float(round(r["waypoint_frac"], 5)))
        cols["task_miss_reason"].append(r["miss_reason"] or "")
    return cols


# --------------------------------------------------------------------------
# offline verification: replay integration + tier table
# --------------------------------------------------------------------------

def integrate_replay(traj, tier: int, n_seeds: int, seed: int,
                     amp_max: float = 0.06, no_crop: bool = False) -> dict:
    """Integrate a SkillReplayDelta stream from the birth pose (passthrough).

    Exactly what a no-safety env executes: q += emitted delta each step.
    Steps budget covers the slowest tier playback (speed_range[0] shrunk by
    the OU wobble band); after the (possibly end_frac-cropped) trajectory
    ends the source holds zeros, so extra steps are harmless.
    ``no_crop=True`` zeroes start_frac/end_frac (attribution arm: isolates
    the joint-noise random walk from the start/end crop truncation).
    Returns {arm: (steps+1, N, dof)}.
    """
    from safeduo.delta.skill_replay import SkillNoiseParams, SkillReplayDelta
    from safeduo.safety.types import SceneState

    p = SkillNoiseParams.tier(tier)
    p.amp_max = float(amp_max)
    if no_crop:
        p.start_frac = 0.0
        p.end_frac = 0.0
    src = SkillReplayDelta(n_seeds, [traj], params=p, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    src.reset(torch.arange(n_seeds), gen)
    if tier == 0:
        steps = traj.n_steps
    else:
        slack = max(p.speed_range[0] * (1.0 - 2.0 * p.time_jitter), 0.25)
        steps = int(math.ceil(traj.n_steps / slack)) + 2
    q = {a: torch.as_tensor(traj.q[a][0]).float()
         .expand(n_seeds, -1).clone() for a in ARM_KEYS}
    qs = {a: [q[a].clone()] for a in ARM_KEYS}
    state = SceneState(q=q, dt=traj.dt)
    for _ in range(steps):
        cmd = src.sample(state)
        for a in ARM_KEYS:
            q[a] = q[a] + cmd.delta_q[a]
            qs[a].append(q[a].clone())
    return {a: torch.stack(v) for a, v in qs.items()}


def tier_table(judge: SkillSuccessJudge, skill_dir: "str | Path",
               tiers: "list[int]" = (0, 1, 2, 3), n_seeds: int = 25,
               seed: int = 20260816, amp_max: float = 0.06,
               no_crop: bool = False) -> dict:
    """{skill: {tier: {n, k_success, rate, waypoint_frac_mean,
    miss_reasons}}} -- the offline calibration/verification table."""
    from safeduo.delta.skill_replay import load_manifest_library

    lib = load_manifest_library(Path(skill_dir) / "skills_manifest.json")
    out: dict = {}
    for traj in lib:
        name = str(traj.meta["skill"])
        out[name] = {}
        for tier in tiers:
            qtr = integrate_replay(traj, tier, n_seeds, seed + tier,
                                   amp_max=amp_max, no_crop=no_crop)
            ee = judge.fk.ee_pos(qtr)                # arm -> (T+1, N, 3)
            k = 0
            fracs, reasons = [], {}
            for e in range(n_seeds):
                r = judge.evaluate_ee({a: ee[a][:, e] for a in ARM_KEYS},
                                      name, dt=traj.dt)
                k += int(r["success"])
                fracs.append(r["waypoint_frac"])
                if r["miss_reason"]:
                    key = r["miss_reason"].split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
            out[name][tier] = {
                "n": n_seeds, "k_success": k, "rate": k / n_seeds,
                "waypoint_frac_mean": round(sum(fracs) / len(fracs), 4),
                "miss_reasons": reasons,
            }
    return out


def render_tier_table(table: dict) -> str:
    tiers = sorted(next(iter(table.values())).keys())
    lines = ["| skill | " + " | ".join(f"tier{t}" for t in tiers)
             + " | wp_frac t" + "/t".join(str(t) for t in tiers) + " |",
             "|---|" + "---|" * (len(tiers) + 1)]
    for name, row in table.items():
        cells = [f"{row[t]['k_success']}/{row[t]['n']}" for t in tiers]
        fracs = "/".join(f"{row[t]['waypoint_frac_mean']:.2f}" for t in tiers)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {fracs} |")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=str, default=str(DEFAULT_SKILL_DIR))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tiers", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--amp-max", type=float, default=0.06)
    parser.add_argument("--no-crop", action="store_true",
                        help="zero start_frac/end_frac (attribution arm: "
                             "noise-only, no start/end truncation)")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    judge = SkillSuccessJudge(skill_dir=args.skill_dir,
                              config_path=args.config)
    table = tier_table(judge, args.skill_dir, tiers=args.tiers,
                       n_seeds=args.seeds, seed=args.seed,
                       amp_max=args.amp_max, no_crop=args.no_crop)
    md = render_tier_table(table)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"schema": "skill_task_success_tiers_v1", "seeds": args.seeds,
             "seed_base": args.seed, "amp_max": args.amp_max,
             "no_crop": args.no_crop, "table": table}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
