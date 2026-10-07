"""rows.J sign/scale audit probe (A6-W6, adjudication for the support line's
damper energy-store investigation, STATUS_SERVER 08-12 19:05).

Negative-result map: under passthrough on the L1 domain every EXTERNAL
intervention (target leash / dmin lift / total-offset accounting) makes
violations WORSE. Two competing hypotheses:
  (a) rows.J sign/scale convention mismatch vs the backstop's G=-J usage;
  (b) energy store: the damper constrains only the per-step increment u while
      the PD chases the persistent-target backlog (q_target - q), releasing
      stored motion across steps -> any static per-step accounting is wrong.

This probe records, for every valid sphere-sphere active row at every step:
  {d, dmin, cap, h_F, h_U, Gu_F, Gu_U, Gb_F, Gb_U, J_dq, dd}
where  Gu_r = (-J_r) @ u_r          (damper's constrained quantity, exec delta)
       Gb_r = (-J_r) @ backlog_r    (PD backlog incl. this step's increment)
       J_dq = sum_r J_r @ (q1-q0)_r (prediction of dd if J is correct)
       dd   = d(t+1) - d(t)         (measured margin change, same global pair)

Verdicts computed in-script:
  A. slope/pearson of dd ~ J_dq   -> J semantics: ~+1 correct, ~-1 sign flip,
                                     other stable slope = scale error.
  B. |corr(dd, -Gu)| vs |corr(dd, -Gb)| -> which variable actually drives the
                                     margin: the increment or the PD backlog.
  C. compliance frac(Gu_r <= h_r + tol) on involved rows -> projector honors
                                     its own constraint (implementation ok).

Usage (server):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  PYTHONPATH=src python -m safeduo.safety.damper_audit \
    --num_envs 128 --episodes 2 --delta_source l1 \
    --out artifacts/analysis/damper_audit --headless
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--yaml", type=str, default="duo_env.yaml",
                    help="env profile; duo_env_v4.yaml = v4 real-asset scene")
parser.add_argument("--delta_source", type=str, default="l1")
parser.add_argument("--d_record", type=float, default=0.12,
                    help="record only rows with margin below this (volume cap)")
parser.add_argument("--backlog_aware", action="store_true",
                    help="enable the backstop backlog-aware budget (A/B arm)")
parser.add_argument("--contact_gt", action="store_true",
                    help="enable ContactSensor GT and pair contact onset vs "
                         "sphere margins (contact-before-alarm adjudication)")
parser.add_argument("--dmin_self", type=float, default=0.0,
                    help=">0: override semantics d_min['self'] (hot-fix A/B)")
parser.add_argument("--dmin_cross", type=float, default=0.0,
                    help=">0: override semantics d_min['cross']")
parser.add_argument("--gamma", type=float, default=0.0,
                    help=">0: override backstop gamma (retune arm)")
parser.add_argument("--out", type=str, default="artifacts/analysis/damper_audit")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.baselines.base import stack_robot  # noqa: E402
from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS, CLASS_CROSS  # noqa: E402


@torch.no_grad()
def main() -> None:
    cfg = make_duo_env_cfg(num_envs=args.num_envs, coordinator=True,
                           yaml_name=args.yaml)
    cfg.seed = args.seed
    cfg.coordinator["delta_source"] = args.delta_source
    if args.backlog_aware:
        cfg.safety_cfg["backstop"]["backlog_aware"] = True
    if args.contact_gt:
        cfg.enable_contact_gt = True
    ov = {}
    if args.dmin_self > 0:
        ov["self"] = args.dmin_self
    if args.dmin_cross > 0:
        ov["cross"] = args.dmin_cross
    if ov:
        cfg.safety_cfg["d_min_override"] = ov
    if args.gamma > 0:
        cfg.safety_cfg["backstop"]["gamma"] = args.gamma
    env = DuoEnv(cfg)
    n = env.num_envs
    dev = env.device
    m = env._sph
    prov = env._provider
    gamma = env._backstop.cfg.gamma
    dt = float(env.cfg.sim.dt * env.cfg.decimation)
    steps = int(env.max_episode_length)
    s_table = m._slice_table.start
    radii = m.radii

    action = torch.zeros(n, 5, device=dev)
    action[:, :4] = 1.0  # passthrough: alpha=1, p=0

    # contact-before-alarm adjudication (STATUS_SERVER 20:30): pair PhysX
    # ContactSensor onsets with the sphere-layer margins at that moment.
    # Weld-polluted bodies excluded (fr3_hand fixed-joint pseudo-force 96kN,
    # A2 protocol-v2 workaround; gripper/tool/flange remnants likewise).
    contact = None
    if args.contact_gt:
        import re
        bad = re.compile(r"hand|finger|tcp|link8|gripper|tool|flange", re.I)
        contact = {}
        for arm in ARM_KEYS:
            sensor = env._contact[arm]
            keep_m = torch.tensor([not bad.search(nm) for nm in sensor.body_names],
                                  dtype=torch.bool, device=dev)
            contact[arm] = (sensor, keep_m)
            print(f"CONTACT_BODIES {arm} kept="
                  f"{[nm for nm in sensor.body_names if not bad.search(nm)]}",
                  flush=True)
        # per-arm force channels recorded every step; onset detection runs
        # OFFLINE with per-arm adaptive floors (the fr3_hand weld pseudo-force
        # keeps F-arm net force permanently >1N -- a fixed global threshold
        # masks every real onset, see R1 instrument bug in STATUS_A W6)
        force_rows: list = []
        margin_rows: list = []
        reset_rows: list = []

    def pair_d(gidx: torch.Tensor) -> torch.Tensor:
        """Sphere-sphere margin for global pair ids, from current centers."""
        centers = m.last_centers
        pt = m.pair_table[gidx.clamp_min(0)]                      # (N,M,2)
        nn, mm = gidx.shape
        ci = centers.gather(1, pt[..., 0].unsqueeze(-1).expand(nn, mm, 3))
        cj = centers.gather(1, pt[..., 1].unsqueeze(-1).expand(nn, mm, 3))
        return (ci - cj).norm(dim=-1) - radii[pt[..., 0]] - radii[pt[..., 1]]

    cols = ("d", "dmin", "cap", "cls", "h_F", "h_U", "Gu_F", "Gu_U",
            "Gb_F", "Gb_U", "J_dq", "dd", "inv_F", "inv_U")
    recs: dict[str, list] = {c: [] for c in cols}
    sanity_logged = False
    # episode-slot violation accounting, same caliber as eval_coordinator
    # (support-line passthrough L1 baseline 27.8%): slot = steps-long window,
    # violated if any step in it raises the violation flag
    slot_viol = torch.zeros(args.episodes, n, dtype=torch.bool, device=dev)

    obs, _ = env.reset(seed=args.seed)
    for t in range(args.episodes * steps):
        slot_viol[t // steps] |= env._last_out.violation
        out = env._last_out
        rows = prov.rows_from(out, env._body_pos_cache)
        q0 = {a: env._arms[a].data.joint_pos[:, env._joint_idx[a]].clone()
              for a in ARM_KEYS}
        gidx = out.active_idx.clone()
        d0 = rows.d.clone()
        dmin0 = out.active_dmin.clone()
        cls0 = rows.cls.clone()
        keep = rows.valid & (gidx >= 0) & (gidx < s_table) \
            & (d0 < args.d_record)
        # mid-step resets make the post-step state garbage for those envs:
        # pre-step violations terminate now, episode-boundary envs truncate now
        keep &= ~out.violation.unsqueeze(-1)
        keep &= (env.episode_length_buf < steps - 2).unsqueeze(-1)
        if not sanity_logged and keep.any():
            diff = (pair_d(gidx) - d0)[keep].abs().max().item()
            print(f"AUDIT_SANITY pair_d vs rows.d max|diff|={diff:.2e}",
                  flush=True)
            assert diff < 1e-4, "pair_d recompute mismatch"
            sanity_logged = True
        J = {r: rows.J[r].clone() for r in ("F", "U")}
        inv = {r: (rows.arm_mask[..., [0, 1] if r == "F" else [2, 3]]
                   .any(-1)) for r in ("F", "U")}

        obs, _, _, _, _ = env.step(action)

        if contact is not None:
            farm = torch.zeros(n, 4, device=dev)
            for ai, arm in enumerate(ARM_KEYS):
                sensor, keep_m = contact[arm]
                f = sensor.data.net_forces_w.norm(dim=-1)        # (N, B)
                f = torch.where(keep_m.unsqueeze(0), f, torch.zeros_like(f))
                farm[:, ai] = f.amax(dim=-1)
            mm = env._last_out.min_margin                        # post-step margins
            force_rows.append(farm.cpu())
            margin_rows.append(torch.stack(
                [mm["cross"], mm["self_F"], mm["self_U"], mm["table"]],
                dim=-1).cpu())
            reset_rows.append((env.episode_length_buf == 0).cpu())

        exec_cmd = env._step_cache["exec"]
        u = {r: stack_robot(exec_cmd.delta_q, r) for r in ("F", "U")}
        q1 = {a: env._arms[a].data.joint_pos[:, env._joint_idx[a]]
              for a in ARM_KEYS}
        dq = {a: q1[a] - q0[a] for a in ARM_KEYS}
        dq_r = {r: stack_robot(dq, r) for r in ("F", "U")}
        backlog = {a: env._targets[a] - q0[a] for a in ARM_KEYS}
        bl_r = {r: stack_robot(backlog, r) for r in ("F", "U")}

        cap = gamma * (d0 - dmin0) * dt
        is_cross = cls0 == CLASS_CROSS
        # p=0 passthrough: cross split = 0.5*cap when cap>=0 else cap
        h_split = torch.where(cap >= 0, 0.5 * cap, cap)
        h = {r: torch.where(is_cross, h_split, cap) for r in ("F", "U")}

        def jdot(r: str, vec: torch.Tensor) -> torch.Tensor:
            return torch.einsum("nmd,nd->nm", J[r], vec)

        j_dq = jdot("F", dq_r["F"]) + jdot("U", dq_r["U"])
        dd = pair_d(gidx) - d0

        flat = keep.reshape(-1)

        def put(name: str, ten: torch.Tensor) -> None:
            recs[name].append(ten.reshape(-1)[flat].float().cpu())

        put("d", d0); put("dmin", dmin0); put("cap", cap)
        put("cls", cls0.float()); put("J_dq", j_dq); put("dd", dd)
        for r in ("F", "U"):
            put(f"h_{r}", h[r])
            put(f"Gu_{r}", -jdot(r, u[r]))
            put(f"Gb_{r}", -jdot(r, bl_r[r]))
            put(f"inv_{r}", inv[r].float())

    tab = {k: torch.cat(v) for k, v in recs.items()}
    n_rows = tab["d"].numel()

    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() < 3:
            return float("nan")
        am, bm = a - a.mean(), b - b.mean()
        return float((am * bm).sum()
                     / (am.norm() * bm.norm()).clamp_min(1e-12))

    def slope(y: torch.Tensor, x: torch.Tensor) -> float:
        xm = x - x.mean()
        return float((xm * (y - y.mean())).sum()
                     / (xm * xm).sum().clamp_min(1e-12))

    gu = tab["Gu_F"] * tab["inv_F"] + tab["Gu_U"] * tab["inv_U"]
    gb = tab["Gb_F"] * tab["inv_F"] + tab["Gb_U"] * tab["inv_U"]
    comp = []
    for r in ("F", "U"):
        sel = tab[f"inv_{r}"] > 0
        comp.append(float(((tab[f"Gu_{r}"][sel] <= tab[f"h_{r}"][sel] + 1e-6)
                           .float().mean())) if sel.any() else float("nan"))
    near = tab["d"] < 0.05
    crossed = (tab["d"] > tab["dmin"]) & ((tab["d"] + tab["dd"]) < tab["dmin"])

    summary = {
        "protocol": {"num_envs": args.num_envs, "episodes": args.episodes,
                     "seed": args.seed, "delta_source": args.delta_source,
                     "d_record": args.d_record, "gamma": gamma, "dt": dt,
                     "backlog_aware": bool(args.backlog_aware),
                     "contact_gt": bool(args.contact_gt),
                     "dmin_self": args.dmin_self or None,
                     "dmin_cross": args.dmin_cross or None,
                     "gamma_override": args.gamma or None},
        "violation": {
            "episodes": int(slot_viol.numel()),
            "violation_episodes": int(slot_viol.sum()),
            "violation_rate": round(float(slot_viol.float().mean()), 6),
        },
        "n_rows": int(n_rows),
        "A_j_semantics": {
            "slope_dd_vs_Jdq": slope(tab["dd"], tab["J_dq"]),
            "pearson_dd_vs_Jdq": corr(tab["dd"], tab["J_dq"]),
            "slope_cross": slope(tab["dd"][tab["cls"] == CLASS_CROSS],
                                 tab["J_dq"][tab["cls"] == CLASS_CROSS])
            if (tab["cls"] == CLASS_CROSS).sum() > 10 else None,
            "slope_self": slope(tab["dd"][tab["cls"] != CLASS_CROSS],
                                tab["J_dq"][tab["cls"] != CLASS_CROSS]),
        },
        "B_energy_store": {
            "corr_dd_vs_negGu": corr(tab["dd"], -gu),
            "corr_dd_vs_negGb": corr(tab["dd"], -gb),
            "near_mean_abs_Gu": float(gu[near].abs().mean()),
            "near_mean_abs_Gb": float(gb[near].abs().mean()),
            "backlog_to_increment_ratio":
                float(gb[near].abs().mean() / gu[near].abs().mean().clamp_min(1e-9)),
        },
        "C_compliance": {"frac_Gu_le_h_F": comp[0], "frac_Gu_le_h_U": comp[1]},
        "crossing_events": {
            "n": int(crossed.sum()),
            "mean_Gu": float(gu[crossed].mean()) if crossed.any() else None,
            "mean_Gb": float(gb[crossed].mean()) if crossed.any() else None,
            "mean_dd": float(tab["dd"][crossed].mean()) if crossed.any() else None,
            "mean_cap": float(tab["cap"][crossed].mean()) if crossed.any() else None,
        },
    }
    if contact is not None:
        F = torch.stack(force_rows)           # (T, N, 4)
        M = torch.stack(margin_rows)          # (T, N, 4) cross/self_F/self_U/table
        R = torch.stack(reset_rows)           # (T, N)
        entry: dict = {"force_floor_p50_N": {}, "force_p99_N": {}}
        adjud = {}
        for ai, arm in enumerate(ARM_KEYS):
            fa = F[..., ai]
            floor = fa.median().item()
            thr = max(1.0, 2.0 * floor + 1.0)
            entry["force_floor_p50_N"][arm] = round(floor, 2)
            entry["force_p99_N"][arm] = round(fa.quantile(0.99).item(), 2)
            hit = fa > thr
            prev = torch.cat([torch.zeros(1, n, dtype=torch.bool), hit[:-1]])
            prev &= ~R                        # reset step: fresh episode
            onset = hit & ~prev
            # relevant sphere margins for this arm: cross + own-self + table
            self_col = 1 if arm[0] == "F" else 2
            mrel = torch.minimum(torch.minimum(M[..., 0], M[..., self_col]),
                                 M[..., 3])
            om = mrel[onset]
            key = f"onsets_{arm}"
            if om.numel():
                adjud[key] = {
                    "n": int(om.numel()), "thr_N": round(thr, 2),
                    "margin_mm_p10_p50_p90": [
                        round(om.quantile(q).item() * 1000, 2)
                        for q in (0.10, 0.50, 0.90)],
                    "frac_margin_positive":
                        round(float((om > 0).float().mean()), 4),
                    "frac_in_0_10mm":
                        round(float(((om > 0) & (om < 0.010)).float().mean()), 4),
                    "frac_in_3_7mm":
                        round(float(((om > 0.003) & (om < 0.007)).float().mean()), 4),
                }
            else:
                adjud[key] = {"n": 0, "thr_N": round(thr, 2)}
        entry["per_arm"] = adjud
        summary["contact_adjudication"] = entry

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if contact is not None:
        torch.save({"force": F, "margin": M, "reset": R},
                   out_dir / f"contact_trace_{stamp}.pt")
    torch.save({k: v for k, v in tab.items()}, out_dir / f"audit_rows_{stamp}.pt")
    (out_dir / f"audit_summary_{stamp}.json").write_text(
        json.dumps(summary, indent=1))
    print("AUDIT_SUMMARY " + json.dumps(summary), flush=True)
    print("AUDIT_FILE " + str(out_dir / f"audit_summary_{stamp}.json"), flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # kit teardown hangs in headless batch mode
