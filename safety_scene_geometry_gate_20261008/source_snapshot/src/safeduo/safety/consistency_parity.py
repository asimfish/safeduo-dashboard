"""G0 一致性对拍：训练版顺序投影（A4）vs 部署版 slack-QP（C 的 strong_cbf_qp）。

口径（ROUND2_DELTAS #3）：hard-conflict 状态上的尾部分歧——P99 改写差异，
不是平均一致率。对齐条件：同一约束行（toy 解析行，d < d_act 预门控）、
同 gamma/d_min/vmax、C 的 QP 退到 vanilla（extrap=zero、borrow=0、tau=0、
p 冻结为外部给定值）、alpha=1（QP 无 alpha 输入）。

改写差异 = max_arm ||exec_A - exec_QP||_inf / (vmax*dt)，
hard-conflict = 该状态最小跨机 margin < hard_thresh。

用法（本地 CPU 即可）：
  cd src && ../.venv/bin/python -m safeduo.safety.consistency_parity
输出 artifacts/parity/consistency_<date>.json + stdout 摘要。
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))  # tests/ 里的 toy_system（C 的解析对拍世界）

from tests.toy_system import ToyProvider, ToyWorld  # noqa: E402

from safeduo.baselines.strong_cbf_qp import StrongCBFQPConfig, StrongCBFQPFilter  # noqa: E402
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop  # noqa: E402
from safeduo.safety.types import ARM_KEYS, DeltaCmd  # noqa: E402

N = 4096
JITTER = 0.30
CMD_AMP = 0.04
D_ACT = 0.25         # 行预门控（两边同一集合）
HARD_THRESH = 0.08   # hard-conflict 判据：最小跨机 margin
GAMMA, D_MIN, VMAX = 4.0, 0.03, 1.5
DT = 1 / 50


def run_once(p_val: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    world = ToyWorld(N, dt=DT)
    q = world.default_q(jitter=JITTER, generator=gen)
    qd = {a: torch.zeros_like(q[a]) for a in ARM_KEYS}
    state = world.state(q, qd)
    rows = ToyProvider(world).rows(state)
    rows.valid = rows.valid & (rows.d < D_ACT)
    cmd = DeltaCmd(delta_q={
        a: (torch.rand(N, 3, generator=gen) * 2 - 1) * CMD_AMP for a in ARM_KEYS
    })
    p = torch.full((N,), p_val)
    alpha = torch.ones(N, 4)

    mine = VelocityDamperBackstop(BackstopConfig(
        gamma=GAMMA, d_min=D_MIN, vmax=VMAX, max_passes=60, tol=1e-8),
        dof_of={a: 3 for a in ARM_KEYS})
    exec_a, active_a, info_a = mine.project(cmd, rows, alpha, p, DT,
                                            torch.full_like(rows.d, D_MIN))

    qp_cfg = StrongCBFQPConfig(
        d_min=D_MIN, d_act=1e6, gamma=GAMMA, vmax=VMAX,
        extrap="zero", borrow_frac=0.0, tau_delay=0.0,
        prio_deadband=1e9, prio_rate=0.0,  # 冻结 p，由外部注入
        solver_iters=800, solver_tol=1e-8,
    )
    filt = StrongCBFQPFilter(N, ToyProvider(world), qp_cfg, dof_of={a: 3 for a in ARM_KEYS})
    filt.p = p.clone()
    # C 的 filter 内部重新取行；provider 相同，行门控由 d_act=1e6 + 我们改过的
    # rows.valid 无法传入（filter 自取），改为把 d_act 设为 D_ACT 让 C 侧自门控
    filt.cfg.d_act = D_ACT
    out_qp = filt.filter(state, cmd)

    box = VMAX * DT
    diffs = []
    for a in ARM_KEYS:
        diffs.append((exec_a.delta_q[a] - out_qp.delta_exec[a]).abs().amax(dim=-1))
    diff = torch.stack(diffs, dim=-1).amax(dim=-1) / box  # (N,)
    hard = world.min_cross_margin(q) < HARD_THRESH

    def pct(x, qq):
        return float(torch.quantile(x, qq).item()) if x.numel() else float("nan")

    d_hard = diff[hard]
    return {
        "p": p_val, "n_states": N, "n_hard": int(hard.sum()),
        "rewrite_diff_all": {"p50": pct(diff, 0.5), "p90": pct(diff, 0.9),
                             "p99": pct(diff, 0.99), "max": float(diff.max())},
        "rewrite_diff_hard": {"p50": pct(d_hard, 0.5), "p90": pct(d_hard, 0.9),
                              "p99": pct(d_hard, 0.99),
                              "max": float(d_hard.max()) if d_hard.numel() else float("nan")},
        "my_residual_max": float(torch.maximum(info_a["residual_F"],
                                               info_a["residual_U"]).max()),
        "qp_slack_max": float(torch.maximum(out_qp.info["slack_F"],
                                            out_qp.info["slack_U"]).max()),
    }


def main():
    torch.manual_seed(0)
    results = [run_once(p, seed=100 + i) for i, p in enumerate((-1.0, 0.0, 1.0))]
    worst_p99 = max(r["rewrite_diff_hard"]["p99"] for r in results)
    summary = {
        "date": str(date.today()),
        "config": {"N": N, "jitter": JITTER, "cmd_amp": CMD_AMP, "d_act": D_ACT,
                   "hard_thresh": HARD_THRESH, "gamma": GAMMA, "d_min": D_MIN,
                   "vmax": VMAX, "dt": DT},
        "runs": results,
        "worst_hard_p99": worst_p99,
        "gate": "PASS" if worst_p99 < 0.05 else "FAIL",
        "metric_doc": "max_arm ||exec_A - exec_QP||_inf / (vmax*dt), P99 over hard-conflict states",
    }
    out_dir = REPO / "artifacts" / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"consistency_{date.today().strftime('%Y%m%d')}.json"
    out.write_text(json.dumps(summary, indent=1))
    print("CONSISTENCY " + json.dumps({
        "worst_hard_p99": round(worst_p99, 5), "gate": summary["gate"],
        "per_p": {str(r["p"]): round(r["rewrite_diff_hard"]["p99"], 5) for r in results},
        "n_hard": [r["n_hard"] for r in results],
    }), flush=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
