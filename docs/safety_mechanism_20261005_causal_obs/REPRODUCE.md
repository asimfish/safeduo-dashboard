# Reproduce the bounded evidence in the existing simulation workspace

Source workspace: `/home/liyufeng/safeduo`.
Owned analysis namespace:
`artifacts/safety_mechanism_20261005_causal_obs`.
Owned raw archive:
`/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs`.
These NAS paths are local artifacts, not downloadable web URLs. Closed raw
records and metadata are sealed with SHA256 readback before delivery.

Frozen checkpoint:
`artifacts/runs/a31b_refine_cross_20260912/model_last.pt`, SHA256
`4dc303940d9fa6de2dbdf1e38599e7219ec3e816179a0884920826be46abd0d6`.
Frozen evaluation helper SHA256:
`174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12`.
Production config: `duo_env_a31_pending_guard.yaml`.
The complete source/config/bank identities and effective CLI arguments are in
`holdout_plan.json`, each child's protocol, and its random manifest. The effective
IsaacLab checkout is `/home/liyufeng/safeduo_isaaclab`, not the similarly named
`/home/liyufeng/IsaacLab` checkout.

To rescore the existing evidence, in the existing configured environment:

```bash
cd /home/liyufeng/safeduo
source /home/liyufeng/safeduo_setup/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python artifacts/safety_mechanism_20261005_causal_obs/analyze_results.py
python artifacts/safety_mechanism_20261005_causal_obs/verify_observers.py
python artifacts/safety_mechanism_20261005_causal_obs/verify_native_visual.py
```

The first command checks all frozen source/checkpoint hashes, all six full
960-step cells, exact paired initial poses/tapes, FIFO6, target limits, original
negative-margin endpoints and observed row budgets. The second command verifies
the separate v2 120-step finite-guard prefix and the older cached-state visual
replay. The third verifies the latest v4 native state, actual USD camera pose and
calibrated center framing. Rescoring is deterministic analysis, not a new trial.

To rerun the holdout, copy `holdout_plan.json` to a new isolated registration,
change only its output_root to a never-used location, and invoke:

```bash
python artifacts/safety_random_space_20261004/isolated_campaign_v2.py \
  --plan /absolute/path/to/new_plan.json \
  --out /absolute/path/to/new_output_root
```

This repeats the registered experiment. It does not provide new command or
initial-state generalization evidence. Generate and preregister fresh tapes and
an independent initial-state design for those claims. Preserve all failed jobs,
partial attempts and negative outcomes; never overwrite this campaign.

Unit evidence: `admission_green.log` (4 passing admission/budget tests), original
`admission_red.log` (legacy-only omission fails), and `finite_guard_tests.log`
(3 passing finite-gate tests). Full v1 holdout still has the all-forecast finite
validation gap; the v2 prefix is not its replacement. The camera startup and
v3 stale pose metadata failures have separate reports.

No revised production controller, hardware action or realtime deployment is
part of this delivery. The original carry-task campaign still has19/192 complete
gate passes, and remains distinct from sphere-pressure experiments.
