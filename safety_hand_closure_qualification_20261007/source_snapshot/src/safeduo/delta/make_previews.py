"""Render sanity previews of the delta sources into artifacts/delta_previews/.

Usage (from repo root):
    .venv/bin/python src/safeduo/delta/make_previews.py

For L1 and every L2 scenario this rolls the stream out on the analytic toy
world and renders (a) facing-pair EE trajectories in the x-z plane and (b)
per-arm delta magnitude timelines, so a human can eyeball that each scenario
produces the intended conflict pattern before it ever touches Isaac.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from safeduo.delta._contract_stub import ARM_KEYS  # noqa: E402
from safeduo.delta.l1_random import JacobianMapper, L1Params, L1RandomDelta  # noqa: E402
from safeduo.delta.l2_scenarios import SCENARIOS, build_scenario  # noqa: E402
from toy_system import BASE_SEP, TOY_DOF, ToyWorld  # noqa: E402

OUT = REPO / "artifacts" / "delta_previews"
N, DT, STEPS = 4, 0.02, 300
COLORS = {"F_L": "tab:blue", "F_R": "tab:cyan", "U_L": "tab:orange", "U_R": "tab:red"}


def _mapper(world):
    return JacobianMapper(world.ee_jacobian, (0.2, -0.4, 0.1), (1.1, 0.4, 0.55))


def rollout(source, world, steps=STEPS):
    q = world.default_q()
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    ee_log = {a: [] for a in ARM_KEYS}
    d_log = {a: [] for a in ARM_KEYS}
    for _ in range(steps):
        state = world.state(q, qd)
        cmd = source.sample(state)
        for a in ARM_KEYS:
            ee_log[a].append(state.ee_pos[a][0].clone())
            d_log[a].append(cmd.delta_q[a][0].norm().item())
        q = {a: q[a] + cmd.delta_q[a] for a in ARM_KEYS}
        qd = {a: cmd.delta_q[a] / DT for a in ARM_KEYS}
    return ({a: torch.stack(v) for a, v in ee_log.items()},
            {a: torch.tensor(v) for a, v in d_log.items()})


def render(name, ee, dmag, note=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for a in ARM_KEYS:
        xs, zs = ee[a][:, 0], ee[a][:, 2]
        ax1.plot(xs, zs, color=COLORS[a], label=a, lw=1.2)
        ax1.scatter(xs[0], zs[0], color=COLORS[a], marker="o", s=25)
        ax1.scatter(xs[-1], zs[-1], color=COLORS[a], marker="x", s=35)
    ax1.axvline(BASE_SEP / 2, color="gray", ls="--", lw=0.8, label="center")
    ax1.axhline(0.0, color="k", lw=1.0)
    ax1.set_xlabel("x (m)"), ax1.set_ylabel("z (m)")
    ax1.set_title(f"{name}: EE paths (o=start, x=end)")
    ax1.legend(fontsize=7, ncol=2)
    t = torch.arange(dmag["F_L"].shape[0]) * DT
    for a in ARM_KEYS:
        ax2.plot(t, dmag[a], color=COLORS[a], label=a, lw=1.0)
    ax2.set_xlabel("t (s)"), ax2.set_ylabel("|delta_q| (rad/step)")
    ax2.set_title("per-arm delta magnitude")
    if note:
        fig.suptitle(note, fontsize=8, y=1.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)


def main():
    g = torch.Generator().manual_seed(1234)
    world = ToyWorld(N, dt=DT)

    l1 = L1RandomDelta(N, L1Params(), mapper=_mapper(world), dof_of=TOY_DOF)
    l1.reset(torch.arange(N), g)
    render("l1_random_jacobian", *rollout(l1, world), note="L1: OU + EE waypoints")

    l1j = L1RandomDelta(N, L1Params(ou_mix=0.3), dof_of=TOY_DOF)
    l1j.reset(torch.arange(N), g)
    render("l1_random_jointspace", *rollout(l1j, world),
           note="L1: OU + joint-space waypoints (geometry-free fallback)")

    for name, cls in SCENARIOS.items():
        world = ToyWorld(N, dt=DT)
        src = build_scenario(name, N, _mapper(world), split="train", n_variants=4,
                             dof_of=TOY_DOF)
        src.reset(torch.arange(N), torch.Generator().manual_seed(7))
        render(name, *rollout(src, world), note=cls.attacks[:110])
        print(f"rendered {name}")
    print(f"previews written to {OUT}")


if __name__ == "__main__":
    main()
