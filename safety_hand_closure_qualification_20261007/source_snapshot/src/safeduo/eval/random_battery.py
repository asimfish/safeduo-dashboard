"""Category-5 random-operator battery: the safety evidence run.

The four-gate battery (clutch_eval) keeps every per-step trace on the CPU, so
it is bound to ~25 envs x 60 s.  The step latency of this pipeline is ~0.4 s
regardless of the env count (training: 4096 envs at ~1e4 env-steps/s), so
episode volume only comes from batching envs.  This battery therefore keeps
GPU-side running counters per env and never stores traces:

  * flows      l1_full      octant-coverage roaming, 6D intent, 3 speed tiers (R25)
               directed_all all 8 scripted conflict families (l2_scenarios)
               mixed        half the envs plain OU roaming, half scripted conflicts
  * per env    steps, actually-safe steps (clutch_eval SAFE_LINES), official
               violation steps (official sphere-module flags), cross violation steps, near-contact
               steps (cross < near_mm), min margins per class, per-arm locked steps,
               per-arm false-brake steps (locked while actually safe)
  * exposure   per env and per arm PAIR (6 pairs) the number of steps the pair's
               own min sphere margin dropped below d_warn -- the owner's
               "every arm must have had the chance to collide" audit; an episode
               with a pair that never came within d_warn is reported separately
  * outcome    episode (= env-window) violation rate 0/N with Clopper-Pearson 95 %
               upper bound, near-contact rate, false-brake rate, per-flow breakdown
  * calibers   official (sphere-module flags, R29 structural per-link rows counted)
               AND non_structural (R29 rows excluded -- the caliber the analytic
               backstop is designed against), plus per-class top violating link
               pairs for attribution (RowViolationAudit)

Usage (bjxy_5090, GPU):
  python -m safeduo.eval.random_battery --headless --ckpt artifacts/runs/a27_v7_r18_s46_coop/model_last.pt \
      --env-yaml duo_env_v7_r18_stack_fix.yaml --num-envs 1024 --duration-s 600 --seeds 1 2 3 \
      --flows l1_full directed_all --amps 0.03 0.06 --theta 0.5:0.2 --retreat-passthrough \
      --out artifacts/random_battery/a27_stack_fix_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from safeduo.safety.sphere_distance import TABLE_NAMES

ARM_KEYS = ("F_L", "F_R", "U_L", "U_R")
ARM_PAIRS = tuple(itertools.combinations(range(4), 2))          # 6 arm pairs
PAIR_NAMES = tuple(f"{ARM_KEYS[i]}-{ARM_KEYS[j]}" for i, j in ARM_PAIRS)
CLASS_KEYS = ("cross", "self_F", "self_U", "table")


def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float | None:
    """One-sided upper confidence bound for a binomial proportion."""
    if n <= 0:
        return None
    if x >= n:
        return 1.0
    alpha = 1.0 - conf

    def log_binom_cdf(p: float) -> float:
        # log P[X <= x] for X ~ Bin(n, p)
        terms = []
        for k in range(0, x + 1):
            lc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            terms.append(lc + k * math.log(p) + (n - k) * math.log1p(-p))
        m = max(terms)
        return m + math.log(sum(math.exp(t - m) for t in terms))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or mid >= 1.0:
            break
        if math.exp(log_binom_cdf(mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


class PairMarginProbe:
    """Per-step min sphere margin per arm pair (N, 6) from the sphere module's
    cached centres (same readout family as clutch_eval.CrossArgminTracker)."""

    def __init__(self, sph) -> None:
        self._sph = sph
        # Same-robot arm pairs live in the self channel. Table rows and
        # within-arm self pairs must not contribute to inter-arm exposure.
        pairs = torch.cat([sph.pairs_cross, sph.pairs_self], dim=0)
        pair_arms = sph.pair_arms[:len(pairs)]
        inter_arm = pair_arms[:, 0] != pair_arms[:, 1]
        self._pairs = pairs[inter_arm]
        pair_arms = pair_arms[inter_arm]
        self._radius_sum = (sph.radii[self._pairs[:, 0]] + sph.radii[self._pairs[:, 1]])
        masks = []
        for i, j in ARM_PAIRS:
            m = ((pair_arms[:, 0] == i) & (pair_arms[:, 1] == j)) | ((pair_arms[:, 0] == j) & (pair_arms[:, 1] == i))
            if not bool(m.any()):
                raise ValueError(f"sphere module has no pairs for {ARM_KEYS[i]}-{ARM_KEYS[j]}")
            masks.append(m)
        self._masks = torch.stack(masks)                                # (6, P)

    def pair_min(self) -> torch.Tensor:
        centers = self._sph.last_centers
        ci = centers[:, self._pairs[:, 0]]
        cj = centers[:, self._pairs[:, 1]]
        margins = (ci - cj).norm(dim=-1) - self._radius_sum.unsqueeze(0)   # (N, P)
        out = []
        for k in range(len(ARM_PAIRS)):
            m = self._masks[k]
            out.append(margins[:, m].amin(dim=1, keepdim=True))
        return torch.cat(out, dim=1)                                    # (N, 6)


ROW_CLASS_KEYS = ("cross", "self_F", "self_U")                         # link-pair channels (no table)


class RowViolationAudit:
    """Per-row (sphere pair) violation accounting on the cross + self channels.

    Two calibers per class for the same step:
      official     any kept row with surface distance < 0 -- identical to the
                   sphere module's min_margin[k] < 0 (R29 structural rows included);
      non_struct   the R29 structural per-link rows are excluded: rows whose
                   per-row d_min override differs from the class lock line
                   (semantics yaml), i.e. link pairs that sit inside the lock
                   line by construction and that the analytic backstop is
                   allowed to ignore (backstop.exempt_structural_rows). This is
                   the caliber the backstop is designed against; a violation
                   here is a genuine unplanned link-pair overlap.
    Plus per-row violation step counts for attribution (which link pairs).
    Table rows keep the official exempt-aware flag (table_flags_from_module)."""

    def __init__(self, sph) -> None:
        self._sph = sph
        pc, ps = len(sph.pairs_cross), len(sph.pairs_self)
        dev = sph.radii.device
        self._pairs = torch.cat([sph.pairs_cross, sph.pairs_self], dim=0)
        self._radius_sum = sph.radii[self._pairs[:, 0]] + sph.radii[self._pairs[:, 1]]
        cls = torch.cat([torch.zeros(pc, dtype=torch.long, device=dev),
                         1 + sph.self_robot.to(dev).long()])                 # 0 cross, 1 self_F, 2 self_U
        self.cls = cls
        self.dmin = sph.pair_dmin[:pc + ps]
        src = sph.sem.d_min if sph.sem is not None else sph.dmin_default
        class_dmin = torch.tensor([float(src["cross"]), float(src["self"]), float(src["self"])], device=dev)
        self.struct = (self.dmin - class_dmin[cls]).abs() > 1e-9            # (P,) R29 structural rows
        self.names = [f"{sph.qualified_names[i]}|{sph.qualified_names[j]}" for i, j in self._pairs.tolist()]
        self.masks = torch.stack([cls == k for k in range(3)])               # (3, P)
        self.n_rows = pc + ps
        # table rows: the per-link table d_min overrides (v6 structural hover links, e.g. the
        # shoulder sphere that sits inside the table lock line by construction) are the R29 rows
        # that actually exist in the v7 envs; the near_table VIOLATION exemption stays applied.
        self.dmin_table = sph.pair_dmin[sph._slice_table]
        self.struct_table = (self.dmin_table - float(src["table"])).abs() > 1e-9    # (P_t,)
        self.table_names = [f"{sph.qualified_names[i]}|{TABLE_NAMES[t]}" for i, t in sph.pairs_table.tolist()]
        self.n_table_rows = int(len(sph.pairs_table))

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, P) surface distances and (N, P) violation flags for the last step."""
        centers = self._sph.last_centers
        d = (centers[:, self._pairs[:, 0]] - centers[:, self._pairs[:, 1]]).norm(dim=-1) - self._radius_sum
        return d, d < 0.0

    def step_table(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, P_t) table margins and official (exempt-aware) per-row violation flags."""
        d = self._sph.last_table_margin
        ex = self._sph.last_table_viol_exempt
        return d, (d < 0.0) & ~ex

    @staticmethod
    def _aggregate(names, steps, struct, dmin, idx, k):
        """Sum violation steps over sphere rows that share a link pair (several spheres per link
        give several rows with the same qualified name); keep the k largest link pairs."""
        agg: dict = {}
        for i in idx.tolist():
            s = int(steps[i])
            if s <= 0:
                continue
            e = agg.setdefault(names[i], {"pair": names[i], "steps": 0, "rows": 0,
                                          "structural": bool(struct[i]), "d_min_m": round(float(dmin[i]), 4)})
            e["steps"] += s
            e["rows"] += 1
        return sorted(agg.values(), key=lambda e: -e["steps"])[:k]

    def top_rows(self, row_steps: torch.Tensor, k: int = 8, table_steps: torch.Tensor | None = None) -> dict:
        """Per class the k link pairs with the most violation steps (rows aggregated per link pair)."""
        out = {}
        steps_cpu = row_steps.detach().cpu()
        for c, key in enumerate(ROW_CLASS_KEYS):
            idx = torch.nonzero(self.masks[c].cpu(), as_tuple=False).flatten()
            out[key] = self._aggregate(self.names, steps_cpu, self.struct.cpu(), self.dmin.cpu(), idx, k) if idx.numel() else []
        if table_steps is not None and table_steps.numel():
            ts = table_steps.detach().cpu()
            out["table"] = self._aggregate(self.table_names, ts, self.struct_table.cpu(), self.dmin_table.cpu(),
                                           torch.arange(ts.numel()), k)
        return out


def make_flow(flow: str, env, amp: float, env_yaml: str):
    if flow == "l1_full":
        from safeduo.delta.l1_coverage import make_l1_full_v7

        return make_l1_full_v7(env.num_envs, amp_max=float(amp), device=env.device, env_yaml=env_yaml)
    from safeduo.delta.l2_env_source import ConflictMixSource
    from safeduo.delta.l2_scenarios import SCENARIOS

    mix = {"l1": 0.0, "l2": 1.0} if flow == "directed_all" else {"l1": 0.5, "l2": 0.5}
    cfg = {"mix": mix, "scenarios": {name: 1.0 for name in SCENARIOS},
           "split": "eval", "n_variants": 200, "stages": None, "geometry": "v7"}
    return ConflictMixSource(env.num_envs, cfg, device=env.device, amp_max=float(amp), env_yaml=env_yaml)


def run_window(env, driver, steps: int, safe_lines: dict, d_warn: float, near_m: float,
               log_every: int = 600, observer=None) -> dict:
    from safeduo.eval.endurance_eval import table_flags_from_module

    N, dev = env.num_envs, env.device
    probe = PairMarginProbe(env._sph)
    audit = RowViolationAudit(env._sph)
    z = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=dev)          # noqa: E731
    acc = {
        "steps": 0,
        "safe": z(N), "viol_any": z(N), "viol_cross": z(N), "near": z(N),
        "viol_cls": {k: z(N) for k in CLASS_KEYS},
        "locked": z(N, 4), "false_brake": z(N, 4), "pair_warn": z(N, 6),
        "min_cls": {k: torch.full((N,), 10.0, device=dev) for k in CLASS_KEYS},
        "min_pair": torch.full((N, 6), 10.0, device=dev),
        # second caliber: R29 structural rows excluded (table stays official)
        "viol_ns_any": z(N), "viol_ns_cls": {k: z(N) for k in CLASS_KEYS},
        "min_ns_cls": {k: torch.full((N,), 10.0, device=dev) for k in CLASS_KEYS},
        "row_viol": z(audit.n_rows), "row_viol_table": z(audit.n_table_rows), "audit_mismatch": z(N),
        "audit": audit,
    }
    ns_masks = audit.masks & ~audit.struct.unsqueeze(0)                          # (3, P) non-structural rows
    ns_table = ~audit.struct_table                                               # (P_t,)
    obs, _ = env.reset()
    driver.reset(torch.arange(N, device=dev))
    if observer is not None:
        observer.start(env)
    t0 = time.time()
    for t in range(steps):
        action = driver.act(env, obs)
        if observer is not None and hasattr(observer, 'before_step'):
            observer.before_step(env, t)
        obs, _, term, trunc, _ = env.step(action)
        if bool((term | trunc).any()):
            raise RuntimeError(f"env reset inside a battery window at step {t}")
        if observer is not None:
            observer.step(env, t)
        out = env._last_out
        mm = out.min_margin
        tv, _ = table_flags_from_module(env._sph)
        safe = torch.ones(N, dtype=torch.bool, device=dev)
        for k, line in safe_lines.items():
            safe &= mm[k] > line
        safe &= ~tv
        acc["safe"] += safe.long()
        acc["viol_any"] += out.violation.long() if getattr(out, "violation", None) is not None \
            else ((mm["cross"] < 0) | tv).long()
        acc["viol_cross"] += (mm["cross"] < 0).long()
        for k in ("cross", "self_F", "self_U"):
            acc["viol_cls"][k] += (mm[k] < 0).long()
        acc["viol_cls"]["table"] += tv.long()
        acc["near"] += (mm["cross"] < near_m).long()
        for k in CLASS_KEYS:
            acc["min_cls"][k] = torch.minimum(acc["min_cls"][k], mm[k])
        # row audit: official recomputed from centres (cross-check) + non-structural caliber
        d_rows, viol_rows = audit.step()
        acc["row_viol"] += viol_rows.sum(dim=0)
        official_rc = viol_rows.any(dim=1) | tv
        official_env = out.violation if getattr(out, "violation", None) is not None else ((mm["cross"] < 0) | tv)
        acc["audit_mismatch"] += (official_rc != official_env).long()
        d_tab, viol_tab = audit.step_table()                                     # official per-row table flags
        acc["row_viol_table"] += viol_tab.sum(dim=0)
        if bool(ns_table.any()):
            tv_ns = viol_tab[:, ns_table].any(dim=1)
            acc["min_ns_cls"]["table"] = torch.minimum(acc["min_ns_cls"]["table"], d_tab[:, ns_table].amin(dim=1))
        else:
            tv_ns = torch.zeros_like(tv)
        ns_any = tv_ns.clone()
        for c, k in enumerate(ROW_CLASS_KEYS):
            m = ns_masks[c]
            if bool(m.any()):
                v = viol_rows[:, m].any(dim=1)
                acc["viol_ns_cls"][k] += v.long()
                ns_any |= v
                acc["min_ns_cls"][k] = torch.minimum(acc["min_ns_cls"][k], d_rows[:, m].amin(dim=1))
        acc["viol_ns_cls"]["table"] += tv_ns.long()
        acc["viol_ns_any"] += ns_any.long()
        eng = getattr(driver, "engaged", None)
        if eng is None:
            alpha = env._step_cache["alpha"]
            eng = alpha > 1.0 - 1e-6
        locked = ~eng.bool()
        acc["locked"] += locked.long()
        acc["false_brake"] += (locked & safe.unsqueeze(1)).long()
        pm = probe.pair_min()
        acc["pair_warn"] += (pm < d_warn).long()
        acc["min_pair"] = torch.minimum(acc["min_pair"], pm)
        acc["steps"] = t + 1
        if t % log_every == 0:
            rate = (t + 1) / max(time.time() - t0, 1e-9)
            print(f"[battery] step {t}/{steps} ({rate:.2f} steps/s = {rate * N:.0f} env-steps/s, "
                  f"eta {(steps - t) / max(rate, 1e-9):.0f}s) viol_eps={int((acc['viol_any'] > 0).sum())}",
                  flush=True)
    acc["wall_s"] = time.time() - t0
    return acc


def summarize_window(acc: dict, meta: dict, k_expose: int) -> dict:
    N = acc["safe"].shape[0]
    T = acc["steps"]
    viol_eps = int((acc["viol_any"] > 0).sum())
    cross_eps = int((acc["viol_cross"] > 0).sum())
    exposed_all = (acc["pair_warn"] >= k_expose).all(dim=1)               # every pair had >= k warn steps
    n_exposed = int(exposed_all.sum())
    viol_eps_exposed = int(((acc["viol_any"] > 0) & exposed_all).sum())
    safe_arm_steps = float(acc["safe"].sum()) * 4
    row = {
        **meta,
        "num_envs": N, "steps": T, "wall_s": round(acc["wall_s"], 1),
        "env_steps_per_s": round(N * T / max(acc["wall_s"], 1e-9), 1),
        "episodes": N,
        "violation_episodes": viol_eps,
        "violation_episode_rate": viol_eps / N,
        "violation_cp95_upper": clopper_pearson_upper(viol_eps, N),
        "cross_violation_episodes": cross_eps,
        "violation_episodes_by_class": {k: int((acc["viol_cls"][k] > 0).sum()) for k in CLASS_KEYS},
        "violation_steps_by_class": {k: int(acc["viol_cls"][k].sum()) for k in CLASS_KEYS},
        "violation_step_rate": float(acc["viol_any"].sum()) / (N * T),
        "near_contact_step_rate": float(acc["near"].sum()) / (N * T),
        "safe_step_frac": float(acc["safe"].sum()) / (N * T),
        "false_brake_arm_rate": float(acc["false_brake"].sum()) / max(safe_arm_steps, 1.0),
        "locked_frac_per_arm": {a: float(acc["locked"][:, i].sum()) / (N * T) for i, a in enumerate(ARM_KEYS)},
        "min_margin_cls_m": {k: round(float(acc["min_cls"][k].min()), 4) for k in CLASS_KEYS},
        "exposure": {
            "k_warn_steps": k_expose,
            "episodes_all_pairs_exposed": n_exposed,
            "frac_all_pairs_exposed": n_exposed / N,
            "pair_exposed_frac": {PAIR_NAMES[p]: float((acc["pair_warn"][:, p] >= k_expose).float().mean())
                                  for p in range(6)},
            "pair_exposed_episodes": {PAIR_NAMES[p]: int((acc["pair_warn"][:, p] >= k_expose).sum())
                                       for p in range(6)},
            "pair_min_margin_m": {PAIR_NAMES[p]: round(float(acc["min_pair"][:, p].min()), 4) for p in range(6)},
            "violation_episodes_among_exposed": viol_eps_exposed,
            "violation_cp95_upper_exposed": clopper_pearson_upper(viol_eps_exposed, n_exposed),
        },
    }
    if "viol_ns_any" in acc:                                              # second caliber (R29 rows excluded)
        ns_eps = int((acc["viol_ns_any"] > 0).sum())
        audit = acc["audit"]
        row["non_structural"] = {
            "definition": "rows whose d_min == class lock line only (R29 structural per-link rows excluded, "
                          "cross/self AND table); near_table VIOLATION exemption still applied",
            "structural_rows": int(audit.struct.sum()), "rows_total": int(audit.n_rows),
            "structural_table_rows": int(audit.struct_table.sum()), "table_rows_total": int(audit.n_table_rows),
            "violation_episodes": ns_eps,
            "violation_episode_rate": ns_eps / N,
            "violation_cp95_upper": clopper_pearson_upper(ns_eps, N),
            "violation_episodes_by_class": {k: int((acc["viol_ns_cls"][k] > 0).sum()) for k in CLASS_KEYS},
            "violation_steps_by_class": {k: int(acc["viol_ns_cls"][k].sum()) for k in CLASS_KEYS},
            "violation_step_rate": float(acc["viol_ns_any"].sum()) / (N * T),
            "min_margin_cls_m": {k: round(float(acc["min_ns_cls"][k].min()), 4) for k in CLASS_KEYS},
            "official_recompute_mismatch_steps": int(acc["audit_mismatch"].sum()),
            "top_rows": audit.top_rows(acc["row_viol"], table_steps=acc.get("row_viol_table")),
        }
    return row


def aggregate(rows: list) -> dict:
    n = sum(r["episodes"] for r in rows)
    v = sum(r["violation_episodes"] for r in rows)
    ne = sum(r["exposure"]["episodes_all_pairs_exposed"] for r in rows)
    ve = sum(r["exposure"]["violation_episodes_among_exposed"] for r in rows)
    tot_steps = sum(r["episodes"] * r["steps"] for r in rows)
    pair_exposed = {
        name: sum(r["exposure"].get("pair_exposed_episodes", {}).get(name, 0)
                  for r in rows)
        for name in PAIR_NAMES
    }
    ns_rows = [r for r in rows if "non_structural" in r]
    ns = None
    if ns_rows:
        vn = sum(r["non_structural"]["violation_episodes"] for r in ns_rows)
        nn = sum(r["episodes"] for r in ns_rows)
        ns = {"episodes": nn, "violation_episodes": vn, "violation_episode_rate": vn / max(nn, 1),
              "violation_cp95_upper": clopper_pearson_upper(vn, nn),
              "by_flow": {f: {"episodes": sum(r["episodes"] for r in ns_rows if r["flow"] == f),
                              "violation_episodes": sum(r["non_structural"]["violation_episodes"]
                                                        for r in ns_rows if r["flow"] == f)}
                          for f in sorted({r["flow"] for r in ns_rows})}}
    return {
        "windows": len(rows), "episodes": n, "violation_episodes": v,
        "violation_episode_rate": v / max(n, 1), "violation_cp95_upper": clopper_pearson_upper(v, n),
        "non_structural": ns,
        "episodes_all_pairs_exposed": ne, "violation_episodes_among_exposed": ve,
        "violation_cp95_upper_exposed": clopper_pearson_upper(ve, ne),
        "pair_exposure": {
            "episodes_exposed_by_pair": pair_exposed,
            "episodes_total": n,
            "all_pairs_seen_across_windows": all(v > 0 for v in pair_exposed.values()),
            "pair_exposed_fraction": {k: v / max(n, 1) for k, v in pair_exposed.items()},
        },
        "operator_hours": round(tot_steps * rows[0]["dt"] / 3600.0, 2) if rows else 0.0,
        "false_brake_arm_rate_mean": sum(r["false_brake_arm_rate"] for r in rows) / max(len(rows), 1),
        "near_contact_step_rate_mean": sum(r["near_contact_step_rate"] for r in rows) / max(len(rows), 1),
        "by_flow": {f: {"episodes": sum(r["episodes"] for r in rows if r["flow"] == f),
                        "violation_episodes": sum(r["violation_episodes"] for r in rows if r["flow"] == f)}
                    for f in sorted({r["flow"] for r in rows})},
    }


def main():
    parser = argparse.ArgumentParser(description="SafeDuo category-5 random-operator battery")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--env-yaml", default="duo_env_v7_r18_stack_fix.yaml")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--flows", nargs="+", default=["l1_full", "directed_all"],
                        choices=("l1_full", "directed_all", "mixed"))
    parser.add_argument("--amps", nargs="+", type=float, default=[0.03, 0.06])
    parser.add_argument("--theta", default="0.5:0.2", help="clutch hysteresis hi:lo; 'off' = reference driver")
    parser.add_argument("--retreat-passthrough", action="store_true")
    parser.add_argument("--retreat-dwell", type=int, default=10)
    parser.add_argument("--d-warn-mm", type=float, default=80.0,
                        help="exposure line: the v7 cross d_warn / four-gate trigger margin (80 mm)")
    parser.add_argument("--near-mm", type=float, default=5.0, help="near-contact line on the cross channel")
    parser.add_argument("--k-expose", type=int, default=3, help="warn steps per pair for 'exposed'")
    parser.add_argument("--log-every", type=int, default=600)
    parser.add_argument("--out", required=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.eval.block1_harness import PolicyDriver
    from safeduo.eval.clutch import wrap_clutch
    from safeduo.eval.clutch_eval import SAFE_LINES
    from safeduo.eval.endurance_eval import ckpt_arm_aware, ckpt_p2_obs

    arm_aware = ckpt_arm_aware(args.ckpt)
    p2 = ckpt_p2_obs(args.ckpt)
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device, yaml_name=args.env_yaml,
                           coordinator=True, arm_aware_obs=arm_aware, p2_obs=p2)
    cfg.coordinator["terminate_on_violation"] = False
    if args.retreat_passthrough:
        cfg.coordinator["retreat_passthrough"] = True
        cfg.coordinator["retreat_dwell_steps"] = int(args.retreat_dwell)
    ctrl_dt = cfg.sim.dt * cfg.decimation
    cfg.episode_length_s = float(args.duration_s) + 5.0
    cfg.seed = int(args.seeds[0])
    env = DuoEnv(cfg)
    steps = int(round(args.duration_s / ctrl_dt))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clutch_on = args.theta != "off"
    if clutch_on:
        hi, lo = (float(v) for v in args.theta.split(":"))
    base = PolicyDriver(args.ckpt, device=str(env.device))
    safe_lines = dict(SAFE_LINES)
    rows = []
    w = 0
    for seed in args.seeds:
        for flow in args.flows:
            for amp in args.amps:
                w += 1
                env._gen.manual_seed(seed * 1000 + w)
                env._pending_cmd = None
                env._delta_src = make_flow(flow, env, amp, args.env_yaml)
                driver = wrap_clutch(base, clutch_on, hi if clutch_on else 0.0, lo if clutch_on else 0.0,
                                     env.num_envs, env.device)
                print(f"[battery] === window {w}: seed={seed} flow={flow} amp={amp} "
                      f"envs={env.num_envs} steps={steps} ===", flush=True)
                acc = run_window(env, driver, steps, safe_lines, args.d_warn_mm * 1e-3, args.near_mm * 1e-3,
                                 log_every=args.log_every)
                meta = {"window": w, "seed": seed, "flow": flow, "amp": amp, "dt": ctrl_dt,
                        "theta": args.theta, "ckpt": args.ckpt, "env_yaml": args.env_yaml,
                        "retreat_passthrough": args.retreat_passthrough}
                row = summarize_window(acc, meta, args.k_expose)
                rows.append(row)
                with open(out_dir / f"window_{w:03d}.json", "w") as f:
                    json.dump(row, f, indent=1)
                ns = row.get("non_structural", {})
                print(f"[battery] window {w} done: viol_eps={row['violation_episodes']}/{row['episodes']} "
                      f"(CP95 <= {row['violation_cp95_upper']:.4f}) "
                      f"non_struct_eps={ns.get('violation_episodes')} mismatch={ns.get('official_recompute_mismatch_steps')} "
                      f"exposed={row['exposure']['frac_all_pairs_exposed']:.2f} "
                      f"near={row['near_contact_step_rate']:.4f} false_brake={row['false_brake_arm_rate']:.3f} "
                      f"min_cross={row['min_margin_cls_m']['cross']} wall={row['wall_s']}s", flush=True)
                summary = {"args": vars(args) if not hasattr(args, "__dict__") else
                           {k: v for k, v in vars(args).items() if not k.startswith("_")},
                           "aggregate": aggregate(rows), "windows": rows}
                with open(out_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=1, default=str)
    agg = aggregate(rows)
    ns = agg.get("non_structural") or {}
    print(f"RANDOM_BATTERY_SUMMARY episodes={agg['episodes']} violation_episodes={agg['violation_episodes']} "
          f"cp95_upper={agg['violation_cp95_upper']:.5f} non_struct_violation_episodes={ns.get('violation_episodes')} "
          f"exposed={agg['episodes_all_pairs_exposed']} "
          f"operator_hours={agg['operator_hours']} false_brake={agg['false_brake_arm_rate_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
