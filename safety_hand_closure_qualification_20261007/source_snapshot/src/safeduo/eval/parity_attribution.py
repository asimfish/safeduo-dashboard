"""Attribution companion to parity.py: same harness, UR arms start UNFOLDED so
the known-issue penetrating shoulder<->forearm rows (STATUS_C W2 @A/@B) are
absent -- isolating the pure sequential-projection-vs-QP gap from the sphere
artifact. Writes artifacts/parity/parity_report_unfolded.json/.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo / "src"))

from safeduo.baselines.real_geometry import RealGeometryProvider  # noqa: E402
from safeduo.eval.parity import run_parity  # noqa: E402
from safeduo.safety.types import ARM_KEYS, DOF_OF  # noqa: E402

UR_UNFOLDED_Q = (0.0, -1.2, 1.0, -1.4, -1.57, 0.0)  # elbow opened, wrist clear


class UnfoldedProvider(RealGeometryProvider):
    def default_q(self, jitter: float = 0.0, generator=None) -> dict:
        q = super().default_q(jitter, generator)
        for arm in ("U_L", "U_R"):
            base = torch.tensor(UR_UNFOLDED_Q, device=self.device).expand(
                self.n, DOF_OF[arm]).clone()
            if jitter > 0.0:
                base = base + (torch.rand(self.n, DOF_OF[arm], device=self.device,
                                          generator=generator) * 2 - 1) * jitter
            q[arm] = base
        return q


if __name__ == "__main__":
    prov = UnfoldedProvider(8)
    mm = prov.min_margin_by_class(prov.default_q())
    print("unfolded init margins:",
          {k: round(v.min().item(), 4) for k, v in mm.items()})
    rep = run_parity(provider=prov, n_envs=8, n_states=512,
                     out_dir=None)
    out = _repo / "artifacts" / "parity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity_report_unfolded.json").write_text(json.dumps(rep, indent=2))
    lines = ["# Parity, unfolded-UR attribution run", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rep.items() if k != "config"]
    (out / "parity_report_unfolded.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rep, indent=2))
