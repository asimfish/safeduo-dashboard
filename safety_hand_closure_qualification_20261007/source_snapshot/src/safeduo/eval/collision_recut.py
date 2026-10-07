"""Offline damaging/near-contact re-cut of Block-1 records parquet (W6-1).

G1 Round-46 arbitration split, applied to EXISTING step records -- no sim
re-run needed: the records carry per-step per-class min margins, so the
per-episode deepest sphere-layer penetration is recomputable offline.
Contact force is NOT in the records (duo_env contact sensors only exist under
enable_contact_gt, and the harness recorder never read them), so the force
criterion of the G1 definition is inactive here; per the arbitration its
threshold (20 N / 2 steps) is awaiting A/B-line calibration anyway. Depth-only
splitting errs exclusively in the near-contact direction (an in-band
sustained-force episode would be under- never over-classified as damaging).

Validation: the episode reconstruction + depth semantics reproduce A4's
main-table footnote numbers bit-for-bit on records_safeduo_l2eval.parquet
(1026 episodes, 50 collisions, depth p50 0.51 / p90 1.74 / max 3.27 mm,
92% < 2 mm, 0% > 5 mm -- checked 2026-08-12 before this module was written).

Usage:
  PYTHONPATH=src .venv/bin/python -m safeduo.eval.collision_recut \
      --records-dir artifacts/block1/r2_vs_baselines
Outputs (into --records-dir unless --out given):
  main_table_v2.md       Tab.1 v2: damaging + near-contact + original
                         sphere-layer columns side by side
  collision_split.json   per-cell full split detail (depth stats, class
                         attribution, cross-check record)
The original sphere-layer collision count is cross-checked against
manifest.json per cell; any mismatch aborts (guards episode-reconstruction
drift between this module and the harness).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from safeduo.eval.block1_harness import (
    MARGIN_PAD,
    MARGIN_TRACE_KEYS,
    episode_slices,
    render_tab1,
)
from safeduo.eval.metrics import CollisionSplitConfig, collision_split_stats

RECORD_MARGIN_COLS = {mk: f"min_margin_{mk}" for _, mk in MARGIN_TRACE_KEYS}


def load_records_grid(path: "str | Path") -> dict:
    """records parquet -> {"done", "violation" (T, N) bool,
    "margins": {class: (T, N)}}. The harness writes a flattened row-major
    (T, N) grid (env fastest); that layout is asserted, with a sort fallback
    for any future writer that shuffles rows."""
    import pyarrow.parquet as pq

    cols = ["t", "env_id", "done", "violation"] + list(RECORD_MARGIN_COLS.values())
    tab = pq.read_table(str(path), columns=cols)

    def t(name, dtype):
        return torch.tensor(tab.column(name).to_numpy(zero_copy_only=False)) \
            .to(dtype)

    env_id = t("env_id", torch.long)
    n = int(env_id.max().item()) + 1
    rows = env_id.numel()
    if rows % n != 0:
        raise ValueError(f"{path}: {rows} rows not divisible by {n} envs")
    T = rows // n
    order = None
    if not (env_id.view(T, n) == torch.arange(n)).all():
        t_col = t("t", torch.float64)
        order = torch.argsort(t_col * n + env_id.double(), stable=True)

    def grid(name, dtype):
        v = t(name, dtype)
        if order is not None:
            v = v[order]
        return v.view(T, n)

    return {
        "done": grid("done", torch.bool),
        "violation": grid("violation", torch.bool),
        "margins": {mk: grid(col, torch.float32)
                    for mk, col in RECORD_MARGIN_COLS.items()},
    }


def episode_major(grid: dict) -> dict:
    """Env-major (T, N) grids -> episode-major traces for the split:
    {"violation" (H, E) bool, "margins": {class: (H, E)}, "n_episodes",
    "n_dropped_tail"}. Same boundary + padding semantics as
    block1_harness.chop_episodes (violation pads False, margins pad +1.0)."""
    done = grid["done"]
    episodes, n_tail = episode_slices(done)
    if not episodes:
        raise ValueError("no complete episodes in records")
    H = max(e - s + 1 for _, s, e in episodes)
    E = len(episodes)
    viol = torch.zeros(H, E, dtype=torch.bool)
    margins = {mk: torch.full((H, E), MARGIN_PAD) for mk in grid["margins"]}
    for j, (env, s, e) in enumerate(episodes):
        ln = e - s + 1
        viol[:ln, j] = grid["violation"][s:e + 1, env]
        for mk, m in grid["margins"].items():
            margins[mk][:ln, j] = m[s:e + 1, env]
    return {"violation": viol, "margins": margins,
            "n_episodes": E, "n_dropped_tail": n_tail}


def recut_records(path: "str | Path",
                  cfg: "CollisionSplitConfig | None" = None) -> dict:
    """One records file -> G1 split stats (collision_split_stats output +
    sphere-layer episode count for the manifest cross-check)."""
    em = episode_major(load_records_grid(path))
    out = collision_split_stats(em["margins"], em["violation"], cfg)
    out["collision_episodes"] = int(em["violation"].any(dim=0).sum().item())
    out["n_dropped_tail"] = em["n_dropped_tail"]
    return out


def discover_cells(records_dir: Path) -> list:
    """-> [(method, flow, path)] from records_<method>_<flow>.parquet names."""
    cells = []
    for p in sorted(records_dir.glob("records_*.parquet")):
        m = re.fullmatch(r"records_(.+)_(l1|l2eval)\.parquet", p.name)
        if m:
            cells.append((m.group(1), m.group(2), p))
    return cells


V2_NOTES = """
## Table notes (v2 re-cut, W6-1)

1. Collision taxonomy (G1 Round-46 arbitration): **damaging** = sphere-layer
   penetration depth strictly > {band_mm:.0f} mm (the conservative inflation
   band) OR sustained contact force > {force_n:.0f} N for >= {force_steps}
   steps; **near-contact** = violation inside the band (shallow graze);
   **sphere-layer** = the original any-margin<0 count (pre-split column,
   unchanged from main_table.md).
2. The force criterion is INACTIVE in this re-cut: the records carry no
   contact-force column (duo_env contact sensors exist only under
   enable_contact_gt and the harness recorder never read them) and the 20 N
   threshold itself awaits A/B-line calibration per the arbitration. Depth-only
   splitting can only err toward near-contact, never toward damaging.
3. Sphere-layer counts are cross-checked against manifest.json per cell
   (exact match required); the safeduo/l2eval depth stats reproduce the A4
   footnote bit-for-bit (p50 0.51 / p90 1.74 / max 3.27 mm).
4. Depth stats and per-class attribution for every cell are in
   collision_split.json next to this file.
"""


def run_recut(records_dir: "str | Path", out_path: "str | Path | None" = None,
              manifest_path: "str | Path | None" = None,
              cfg: "CollisionSplitConfig | None" = None) -> dict:
    records_dir = Path(records_dir)
    cfg = cfg or CollisionSplitConfig()
    manifest_path = Path(manifest_path or records_dir / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    cells = discover_cells(records_dir)
    if not cells:
        raise SystemExit(f"no records_*.parquet in {records_dir}")

    results: dict = {}
    detail: dict = {}
    for method, flow, path in cells:
        split = recut_records(path, cfg)
        man = manifest["metrics"].get(f"{method}_{flow}")
        if man is None:
            raise SystemExit(f"{method}_{flow} missing from manifest.json")
        if split["collision_episodes"] != man["collision_episodes"] \
                or split["episodes"] != man["episodes"]:
            raise SystemExit(
                f"{method}/{flow}: re-cut ({split['collision_episodes']}"
                f"/{split['episodes']}) disagrees with manifest "
                f"({man['collision_episodes']}/{man['episodes']}) -- episode "
                "reconstruction drift, refusing to write a table")
        met = dict(man)
        met.update(split)
        results[(method, flow)] = {"metrics": met}
        detail[f"{method}_{flow}"] = split
        print(f"[recut] {method}/{flow}: sphere-layer "
              f"{split['collision_episodes']}/{split['episodes']} -> damaging "
              f"{split['damaging_episodes']} + near-contact "
              f"{split['near_contact_episodes']} (max depth "
              f"{split['collision_depth_mm_max']:.2f} mm)", flush=True)

    table = render_tab1(results)
    table = table.replace("# Block 1 main table (Tab.1 schema)",
                          "# Block 1 main table v2 (G1 damaging/near-contact "
                          "split, offline re-cut)")
    table += V2_NOTES.format(band_mm=cfg.band_m * 1e3, force_n=cfg.force_n,
                             force_steps=cfg.force_steps)
    out_path = Path(out_path or records_dir / "main_table_v2.md")
    out_path.write_text(table, encoding="utf-8")
    split_json = {
        "config": {"band_m": cfg.band_m, "force_n": cfg.force_n,
                   "force_steps": cfg.force_steps},
        "source_manifest": str(manifest_path),
        "cells": detail,
    }
    (out_path.parent / "collision_split.json").write_text(
        json.dumps(split_json, indent=2))
    print(f"[recut] wrote {out_path}", flush=True)
    return {"results": results, "table": table, "out_path": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--band-mm", type=float, default=5.0)
    args = ap.parse_args()
    run_recut(args.records_dir, out_path=args.out or None,
              manifest_path=args.manifest or None,
              cfg=CollisionSplitConfig(band_m=args.band_mm * 1e-3))


if __name__ == "__main__":
    main()
