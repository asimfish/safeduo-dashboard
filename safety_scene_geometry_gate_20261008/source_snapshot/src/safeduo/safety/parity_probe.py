"""A3 GT ?????????????? body??? gt_hit=100% ??????

???stdout??? flush??
  PROBE_ORDER  <arm> sensor body ?? == articulation body ???
  PROBE_REST   <arm> ??????? > 0.1N ? body???/?/????/?? base ??
  PROBE_SPH    ??????? violation ?? + min_margin
  PROBE_RATES  30 ? teleport ????? GT ????????
               gt_all(???) / gt_nobase(? arm_base ?) / pred_raw / pred_policy

???python -m safeduo.safety.parity_probe --num_envs 16 --headless
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--force_eps", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg  # noqa: E402
from safeduo.safety.types import ARM_KEYS  # noqa: E402

# ?? base_own_table verdict=allow ? link ??? contact_semantics.yaml ???
BASE_LINKS = {"fr3_link0", "base_link", "shoulder_link"}


def gt_flags(env, sphere_idx: dict, eps: float) -> torch.Tensor:
    flags = []
    for arm in ARM_KEYS:
        f = env._contact[arm].data.net_forces_w
        fb = f[:, sphere_idx[arm]]
        flags.append(fb.norm(dim=-1).amax(dim=-1))
    return torch.stack(flags, dim=-1).amax(dim=-1) > eps


def main():
    n = args.num_envs
    cfg = make_duo_env_cfg(num_envs=n)
    cfg.enable_contact_gt = True
    env = DuoEnv(cfg)
    env.reset()
    dev = env.device

    # --- body ?????/base ?? ---
    idx_all, idx_nobase = {}, {}
    for arm in ARM_KEYS:
        art = env._arms[arm]
        sensor = env._contact[arm]
        s_names = list(getattr(sensor, "body_names", []) or [])
        a_names = list(art.body_names)
        print(f"PROBE_ORDER {arm} match={s_names == a_names} "
              f"sensor_n={len(s_names)} art_n={len(a_names)}", flush=True)
        if s_names != a_names:
            print(f"PROBE_ORDER_DETAIL {arm} sensor={s_names} art={a_names}", flush=True)
        sph = env._sph._body_idx[arm]
        idx_all[arm] = sph
        keep = [int(b) for b in sph.tolist() if a_names[int(b)] not in BASE_LINKS]
        idx_nobase[arm] = torch.tensor(keep, dtype=torch.long, device=dev)

    # --- ?????? body ?? ---
    for _ in range(5):
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    for arm in ARM_KEYS:
        art, sensor = env._arms[arm], env._contact[arm]
        f = sensor.data.net_forces_w
        fmax = f.norm(dim=-1).amax(dim=0)
        s_names = list(getattr(sensor, "body_names", []) or art.body_names)
        sph = set(idx_all[arm].tolist())
        rows = [{"body": s_names[b], "fmax": round(float(fmax[b]), 2),
                 "sphere": b in sph, "base": s_names[b] in BASE_LINKS}
                for b in range(len(s_names)) if float(fmax[b]) > 0.1]
        print(f"PROBE_REST {arm} {json.dumps(rows)}", flush=True)
    out = env.compute_dist()
    mm = {k: round(float(v.min()), 4) for k, v in out.min_margin.items()}
    print(f"PROBE_SPH rest violation={int(out.violation.sum())}/{n} "
          f"min_margin={json.dumps(mm)}", flush=True)

    # --- teleport ??????? ---
    limits = {}
    for arm in ARM_KEYS:
        art, jid = env._arms[arm], env._joint_idx[arm]
        lim = art.data.soft_joint_pos_limits[0, jid]
        limits[arm] = (lim[:, 0], lim[:, 1])
    gen = torch.Generator(device=dev).manual_seed(7)
    cnt = {"configs": 0, "gt_all": 0, "gt_nobase": 0, "pred_raw": 0,
           "pred_policy": 0, "miss_nobase_raw": 0, "fp_nobase_raw": 0}
    for _ in range(args.iters):
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
        pred_raw = (out.dists < 0.0).any(dim=1)
        gt_a = gt_flags(env, idx_all, args.force_eps)
        gt_nb = gt_flags(env, idx_nobase, args.force_eps)
        cnt["configs"] += n
        cnt["gt_all"] += int(gt_a.sum())
        cnt["gt_nobase"] += int(gt_nb.sum())
        cnt["pred_raw"] += int(pred_raw.sum())
        cnt["pred_policy"] += int(out.violation.sum())
        cnt["miss_nobase_raw"] += int((gt_nb & ~pred_raw).sum())
        cnt["fp_nobase_raw"] += int((pred_raw & ~gt_nb).sum())
    print("PROBE_RATES " + json.dumps(cnt), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
    app.close()
