"""Reset-free endurance evaluation: pure-random success rate vs duration.

Implements section 2B of the teleop-safety eval spec (2026-08-13): N runs x
T-minute continuous four-arm random roaming, one success per run iff the run
has ZERO violation steps (sphere-margin caliber -- conservative under the
+5mm inflation adjudication, spec section 2), reported WITH a two-sided
Clopper-Pearson 95% CI, never as a bare x/N. Per-run rows carry the total and
per-class violation-step counts plus per-class worst margins so the v4
self_U fat-sphere artifact can be split out of the cross-machine story
(Round-110 double-bookkeeping; the GT-verified secondary success column
counts cross/self_F/table only and is labeled as diagnostic).

Violation caliber (2026-08-15 C11 fix of the exemption-blind counting, B-line
08-14 05:50 finding): a violation step now consumes the OFFICIAL exempt-aware
violation semantics. The table channel honors the near_table conditional
exemption (contact_semantics: low-speed hand-on-own-table contact is a
designed-legal state, violation_on_contact=false), matching
SphereDistOut.violation instead of the raw min_margin sign that structurally
failed every l1_ws run (table roaming visits the table). cross/self_F/self_U
have no exemption by design, so their raw-margin counts ARE the official
semantics and are unchanged. Rows dual-book both calibers: the primary
violation/success fields are exempt-aware, the old exemption-blind counts
stay under *_raw (history comparison), and table_exempt_steps is the
transparency column (steps with >=1 exempted table contact).

Reset-free semantics: violation termination is disabled
(coordinator.terminate_on_violation=False, the parity-v3.1 measurement flag)
and episode_length_s is stretched past the run duration, so one env = one
uninterrupted run; a violation marks the run failed but the stream keeps
playing (violation_steps stays a meaningful dose count).

Batching: runs = windows x num_envs. Every window reseeds the env stream
generator (seed + window index) and rebuilds the flow source at that
window's amp, so the seed/amp of every run is logged and reproducible.
Statistics runs are headless multi-env WITHOUT cameras (the viz camera is a
single-instance sensor: num_envs>1 with a camera IndexErrors -- spec note 3);
use --device cpu so the GPUs stay with training.

Flows:
  l1        four-arm nonstop waypoint roaming (the spec's l1 profile: no
            pauses, short segments, extra OU jitter; --tame keeps L1Params
            defaults instead), amp per window;
  l1_ws     same engine, waypoints sampled over the FULL shared workspace in
            EE space (delta.l1_workspace: per-arm reach-clamped boxes covering
            the whole 0.574 m cross-machine overlap band) -- the T2-C spatial
            expansion; use for the long-duration windows where joint-space
            roaming drifts away from the conflict band;
  l1_full   R25 coverage stream (delta.l1_coverage, v7-only): octant-LRU
            waypoints over each arm's FULL reachable box (own-side half,
            deep cross-penetration, extension shell) + 30% conflict-band
            revisits + 10% deep-penetration slabs, 6D orientation intent
            (S2-uniform axis, U(30,120) deg), arrival-triggered segments
            and 3-tier speeds {0.15,0.45,0.80} m/s; params own their design
            defaults, so --tame is a no-op for this flow;
  directed  head_on_crossing + handover_approach eval-split conflict battery
            (the spec's adversarial directed-approach subset);
  skill     T1/T2 skill-library replay (SkillReplayDelta): the 13 recorded
            four-arm cooperation skills listed in --skill-dir's
            skills_manifest.json, looped so the whole window is a nonstop
            stream, per-env skill drawn through the window seed. T2 noise
            tier comes from --skill-tier, or per window from that window's
            amp (skill_tier_for_amp); every run row carries family=<skill>
            + skill_tier, and the batch additionally writes
            skill_matrix_<method>.{md,json} (the spec-2A bucket matrix).
            T4 R5 (additive): skill rows also carry task_success /
            task_waypoint_frac / task_miss_reason (EE waypoint task-
            completion caliber, eval/skill_success.py judging the EXECUTED
            joint stream), and the batch writes the same matrix over that
            caliber as skill_task_matrix_<method>.{md,json}.

Amp tiers (spec section 1 calibration: 0.015 = real-glove scale, 0.03 =
brisk human, 0.06 = "can't brake"):
  --tier regular      amps [0.015, 0.02, 0.025, 0.03]
  --tier adversarial  amps [0.03, 0.04, 0.05, 0.06]
  --tier extreme      amps [0.06, 0.08, 0.10] -- BEYOND braking scale (T2-C
            amplitude expansion, owner's deliberate-strike layer): at 60 Hz
            0.10 rad/step commands ~6 rad/s joint speed, past the FR3/JAKA
            actuator envelopes, so the executed motion saturates at the
            physical velocity limits while the intent keeps outrunning the
            brake -- label these windows adversarial-extreme in reports.

Durations: 60 / 300 s are the owner-spec windows; 600 s (10 min) is the T2-B
endurance tier -- run_window buffers margins at (T, N) floats so a 600 s
window at 25 envs is ~14 MB, no special handling. Wall-clock scales linearly
with steps (CPU ~15-40 steps/s depending on env count).

Methods reuse the block1_harness drivers (same env-in-the-loop pathways as
the main table): safeduo (checkpoint; rsl_rl and lagrangian Beta formats),
estop / speed / cbf (BaselineFilter working points, --frozen-baselines for
the grid winners), passthrough (alpha=1 damper-only rule stack), raw
(backstop bypassed -- reference row only).

Server usage (Isaac; CPU physics headless):
  source ~/safeduo_setup/env.sh && cd ~/safeduo && \
  PYTHONPATH=src python -m safeduo.eval.endurance_eval --headless --device cpu \
      --methods passthrough estop --flow l1 --tier regular \
      --duration-s 60 --num-envs 25 --env-yaml duo_env_v5.yaml \
      --out artifacts/endurance/v5_regular_60s
Smoke (3 runs x 1 min, rule stack, no checkpoint):
  ... --methods passthrough --flow l1 --amp 0.03 --duration-s 60 \
      --num-envs 3 --runs 3 --out /tmp/c8_smoke
Skill matrix (spec 2A; amps 0.03/0.06 map to noise tiers 2/3):
  ... --methods passthrough --flow skill --amps 0.03 0.06 --duration-s 60 \
      --num-envs 3 --env-yaml duo_env_v5.yaml --out /tmp/c9_skill_smoke

The pure-torch pieces (per-run stats, aggregation, CI) are Mac-importable
and unit-tested in tests/test_endurance_eval.py; only main() touches Isaac.

R11 pair attribution (additive, default off): ``--pair-recorder`` writes one
JSONL row for the argmin self_U sphere pair of every env-step whose official
self_U raw sphere margin is negative.  It is a sidecar only; the established
endurance JSON/report payloads are unchanged.  Static pair metadata is bound
before the timed window, per-step reductions stay on the simulation device,
and the recorder performs one device-to-CPU materialization per window.

S13/S18 render-farm export (additive, default off): ``--dump-traj-npz DIR``
writes, per env per window, the S11 PPU-renderer trajectory npz
(tools/ppu_render/README.md schema: q_<arm> executed joints, dt, meta JSON)
extended with the eval's own per-step diagnostics (per-class raw margins,
official exempt-aware violation flags, table_exempt, per-arm executed alpha,
raw policy alpha / hazard probability, clutch latch, and R19 bypass evidence), plus
the <name>_stats.json overlay sidecar the renderer consumes directly --
flash-red frames therefore come from the exact margins this eval recorded,
no recompute drift.  Off (the default) records nothing and the run_window
return schema / numerics are bit-identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from safeduo.eval.metrics import clopper_pearson_interval

CLASS_KEYS = ("cross", "self_F", "self_U", "table")
# Classes with contact-GT backing in the v4 parity battery (cross + table +
# F-self); self_U is the documented conservative-sphere family without GT.
GT_VERIFIED_KEYS = ("cross", "self_F", "table")

TIER_AMPS = {
    "regular": (0.015, 0.02, 0.025, 0.03),
    "adversarial": (0.03, 0.04, 0.05, 0.06),
    # T2-C: beyond-braking scale ("deliberate strike"); shares the 0.06
    # boundary with adversarial the way adversarial shares 0.03 with regular
    "extreme": (0.06, 0.08, 0.10),
}

# owner-spec 1/5 min windows + the T2-B long-endurance 10 min tier
DURATION_TIERS = {"1min": 60.0, "5min": 300.0, "10min": 600.0}


def skill_tier_for_amp(amp: float) -> int:
    """Default T2 noise tier for a skill-flow window when --skill-tier is
    not given, anchored to the same spec section-1 amp calibration as
    TIER_AMPS: below glove scale -> exact replay (0), glove band -> mild (1),
    brisk band -> medium (2), can't-brake and beyond -> aggressive (3)."""
    if amp < 0.015:
        return 0
    if amp < 0.03:
        return 1
    if amp < 0.06:
        return 2
    return 3

FLOW_CHOICES = ("l1", "l1_ws", "l1_full", "directed", "skill")
METHOD_CHOICES = ("safeduo", "estop", "speed", "cbf", "passthrough", "raw")


# --------------------------------------------------------------------------
# pure-torch: per-run statistics from per-class margin traces
# --------------------------------------------------------------------------

def table_flags_from_module(sph) -> tuple:
    """Channel-level exempt-aware table flags for the step just computed.

    Reads the per-step cache SphereDistanceModule.compute leaves on the
    module (last_table_margin / last_table_viol_exempt, the table-slice pair
    margins and near_table VIOLATION-exemption mask) and reduces them to two
    (N,) bools:

      table_viol    any table pair margin<0 AND NOT exempt -- exactly the
                    table component of the official SphereDistOut.violation
                    union (exempt-aware);
      table_exempt  any table pair margin<0 AND exempt -- the designed-legal
                    low-speed hand-on-own-table contact that the raw caliber
                    used to miscount (transparency signal).

    Pure reduction of compute()'s own intermediates: no exemption logic is
    re-derived here, so the yaml semantics stay single-sourced. On the legacy
    (semantics-free) path the mask is all-False and table_viol degenerates to
    the raw margin sign, which IS official there.
    """
    d = sph.last_table_margin                                  # (N, P_t)
    ex = sph.last_table_viol_exempt                            # (N, P_t)
    assert d is not None and ex is not None, \
        "compute() must run before table_flags_from_module()"
    neg = d < 0.0
    return (neg & ~ex).any(dim=1), (neg & ex).any(dim=1)


PAIR_RECORD_FIELDS = (
    "run_id",
    "step",
    "pair_id",
    "qualified_link_i",
    "qualified_link_j",
    "sphere_margin",
    "d_min",
)


class SelfUNegativePairRecorder:
    """Window-buffered argmin attribution for negative self_U env-steps.

    The recorder is deliberately constructed only when the CLI switch is on.
    It uses the complete ``pairs_self`` table rather than ``active_pairs``:
    the latter is capped and therefore cannot prove which self_U pair is the
    true minimum.  One row is emitted per negative *class-minimum* env-step,
    with ties resolved by the module's stable pair-table order (lowest global
    pair id).  ``sphere_margin`` is the same raw shell margin used by
    ``SphereDistOut.min_margin['self_U']``; ``d_min`` is recorded as separate
    configured evidence and is not subtracted from that value.

    Per-step tensors remain on the sphere module's device.  ``finish_window``
    stacks and transfers them once, avoiding a CUDA synchronization per step
    during the later diagnostic run.
    """

    def __init__(self, sph) -> None:
        is_u = sph.self_robot == 1
        if not bool(is_u.any()):
            raise ValueError("sphere module has no self_U pairs to record")
        start = int(sph._slice_self.start)
        stop = int(sph._slice_self.stop)
        local_global_ids = torch.arange(
            start, stop, dtype=torch.long, device=sph.pairs_self.device
        )
        self._sph = sph
        self._pairs = sph.pairs_self[is_u]
        self._global_pair_ids = local_global_ids[is_u]
        self._radius_sum = (
            sph.radii[self._pairs[:, 0]] + sph.radii[self._pairs[:, 1]]
        )

        pairs_cpu = self._pairs.detach().cpu().tolist()
        ids_cpu = self._global_pair_ids.detach().cpu().tolist()
        dmins_cpu = sph.pair_dmin[self._global_pair_ids].detach().cpu().tolist()
        self._pair_evidence = {
            int(pair_id): (
                str(sph.qualified_names[int(pair[0])]),
                str(sph.qualified_names[int(pair[1])]),
                float(d_min),
            )
            for pair_id, pair, d_min in zip(ids_cpu, pairs_cpu, dmins_cpu)
        }
        self.rows: list[dict] = []
        self._run_id_base: int | None = None
        self._min_margin_steps: list[torch.Tensor] = []
        self._pair_id_steps: list[torch.Tensor] = []

    @property
    def pending_min_margins(self) -> tuple[torch.Tensor, ...]:
        """Read-only view used by CPU contract tests and diagnostics."""
        return tuple(self._min_margin_steps)

    def start_window(self, run_id_base: int) -> None:
        if self._run_id_base is not None:
            raise RuntimeError("pair recorder window already active")
        if isinstance(run_id_base, bool) or not isinstance(run_id_base, int):
            raise TypeError("run_id_base must be an int")
        if run_id_base < 0:
            raise ValueError("run_id_base must be non-negative")
        self._run_id_base = run_id_base
        self._min_margin_steps = []
        self._pair_id_steps = []

    def capture_step(self) -> None:
        if self._run_id_base is None:
            raise RuntimeError("start_window() must precede capture_step()")
        centers = self._sph.last_centers
        if centers is None:
            raise RuntimeError("sphere compute() must precede pair capture")
        ci = centers[:, self._pairs[:, 0]]
        cj = centers[:, self._pairs[:, 1]]
        margins = (ci - cj).norm(dim=-1) - self._radius_sum.unsqueeze(0)
        min_margin, argmin = margins.min(dim=1)
        pair_ids = self._global_pair_ids[argmin]
        self._min_margin_steps.append(min_margin.detach().clone())
        self._pair_id_steps.append(pair_ids.detach().clone())

    def finish_window(self) -> None:
        if self._run_id_base is None:
            raise RuntimeError("no active pair recorder window")
        run_id_base = self._run_id_base
        self._run_id_base = None
        if not self._min_margin_steps:
            self._pair_id_steps = []
            return

        margins = torch.stack(self._min_margin_steps).cpu()
        pair_ids = torch.stack(self._pair_id_steps).cpu()
        for step, env_idx in (margins < 0.0).nonzero(as_tuple=False).tolist():
            pair_id = int(pair_ids[step, env_idx])
            link_i, link_j, d_min = self._pair_evidence[pair_id]
            self.rows.append(
                {
                    "run_id": run_id_base + int(env_idx),
                    "step": int(step),
                    "pair_id": pair_id,
                    "qualified_link_i": link_i,
                    "qualified_link_j": link_j,
                    "sphere_margin": float(margins[step, env_idx]),
                    "d_min": d_min,
                }
            )
        self._min_margin_steps = []
        self._pair_id_steps = []

    def abort_window(self) -> None:
        """Discard an incomplete window after the endurance run fails."""
        self._run_id_base = None
        self._min_margin_steps = []
        self._pair_id_steps = []


def write_pair_records_jsonl(rows: list[dict], path: Path) -> None:
    """Atomically write the additive R11 pair-evidence sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            if tuple(row) != PAIR_RECORD_FIELDS:
                raise ValueError(
                    f"pair record fields {tuple(row)} != {PAIR_RECORD_FIELDS}"
                )
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
    tmp.replace(path)

def dump_traj_npz(res: dict, dt: float, out_dir: "str | Path",
                  meta: "dict | None" = None, rows: "list | None" = None,
                  prefix: str = "traj", run_id_base: int = 0) -> list:
    """Write one PPU-renderer npz + stats JSON per env (S13, additive sidecar).

    ``res`` is a run_window result recorded with record_traj=True (carries
    the ``q`` {arm: (T, N, dof)} executed-joint, ``alpha_by_step``
    (T, N, 4), and S18 offline-theta control traces on top of the established
    margin/table traces).

    npz schema = the S11 render-farm contract (tools/ppu_render/README.md):
    q_F_L/q_F_R (T, 7) + q_U_L/q_U_R (T, 6) float32 executed joint
    positions, ``dt`` scalar, ``meta`` JSON string -- extended with the
    per-step diagnostics the eval itself recorded: margin_<class> (T,)
    float32 raw class-min sphere margins, viol_<class> (T,) bool OFFICIAL
    flags (table honors the near_table exemption exactly like the run rows;
    cross/self raw IS official), table_exempt (T,) bool, alpha (T, 4)
    float32.  ``alpha_policy`` is the raw P(safe) detector head and
    ``hazard_prob = 1 - alpha_policy`` is P(danger); ``engaged`` is the
    online hysteresis latch.  ``bypass_blocked_cls`` (T, 4, 3, 2) records
    the geometry-only R19 hazard/gray blockers, independently of the latch,
    so a candidate latch can recompute bypass eligibility offline.  The
    <name>_stats.json sidecar is the render-overlay contract
    ({margin_by_class, alpha}); note its margins stay the RAW sphere
    readout (same transparency convention as worst_margin_by_class), so an
    exempted legal table contact still flashes -- viol_table inside the npz
    is the official caliber for exact bookkeeping.  When the complete S19
    causal group is present, schema v3 also stores the pre-state and active
    rows, the matching raw/executed transition, and post-state pair evidence;
    the temporal contract is explicitly ``pre[t] -> action[t] -> post[t]``.

    ``rows``, when given, are this window's run rows in env order; row e is
    merged into env e's npz meta (full provenance: method/flow/amp/seed/
    run_id/violation counts travel with the trajectory).  File stem is
    ``{prefix}_run{run_id:03d}`` -- the renderer uses it as overlay title.
    Returns the list of written npz paths.
    """
    import numpy as np

    from safeduo.delta._contract_stub import ARM_KEYS

    trace_keys = (
        "q", "alpha_by_step", "alpha_policy_by_step",
        "hazard_prob_by_step", "engaged_by_step", "bypass_arm_by_step",
        "bypass_blocked_cls_by_step",
    )
    for key in trace_keys:
        if key not in res:
            raise ValueError(
                f"run_window result has no '{key}' buffer -- "
                "record_traj=True is required for dump_traj_npz")
    causal_step_names = [
        "pre_active_pair_id", "pre_active_margin", "pre_active_closing",
        "pre_active_class", "pre_active_dmin", "pre_active_exempt",
        "pre_active_mask", "pre_active_arm_mask", "pre_active_J_F",
        "pre_active_J_U", "pre_table_margin_by_pair",
        "pre_table_exempt_by_pair", "bs_active", "priority_p",
        "backstop_residual_F", "backstop_residual_U",
        "post_table_margin_by_pair", "post_table_exempt_by_pair",
    ]
    for arm in ARM_KEYS:
        causal_step_names.extend([
            f"pre_q_{arm}", f"pre_qd_{arm}", f"pre_target_{arm}",
            f"pre_backlog_{arm}", f"cmd_{arm}", f"exec_{arm}",
            f"post_qd_{arm}", f"post_target_{arm}", f"post_backlog_{arm}",
        ])
    causal_static_names = [
        "table_pair_id", "table_pair_sphere_i", "table_pair_table_id",
        "table_pair_arm", "table_pair_dmin", "table_pair_link_name",
    ]
    causal_result_keys = ([f"{key}_by_step" for key in causal_step_names]
                          + causal_static_names)
    causal_present = [key in res for key in causal_result_keys]
    if any(causal_present) and not all(causal_present):
        missing = [key for key, present in zip(causal_result_keys, causal_present)
                   if not present]
        raise ValueError(f"incomplete causal transition trace; missing {missing}")
    has_causal_trace = all(causal_present)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    margins = res["margins"]
    n = margins[CLASS_KEYS[0]].shape[1]
    if rows is not None and len(rows) != n:
        raise ValueError(f"{len(rows)} rows != {n} envs")
    viol = {k: (margins[k] < 0.0) for k in CLASS_KEYS}
    viol["table"] = res["table_viol"].to(dtype=torch.bool)
    paths = []
    for e in range(n):
        row = rows[e] if rows is not None else {}
        rid = int(row.get("run_id", run_id_base + e))
        name = f"{prefix}_run{rid:03d}"
        meta_e = {
            **(meta or {}), **row, "run_id": rid, "env_idx": e,
            "traj_schema": ("safeduo.ppu.v3" if has_causal_trace
                            else "safeduo.ppu.v2"),
            "trajectory_schema_version": 3 if has_causal_trace else 2,
            "hazard_prob_semantics": "1-alpha_policy=P(danger)",
        }
        if has_causal_trace:
            meta_e["transition_alignment"] = "pre[t] -> action[t] -> post[t]"
        npz_path = out_dir / f"{name}.npz"
        payload = dict(
            dt=np.float64(dt),
            meta=json.dumps(meta_e, ensure_ascii=False),
            **{f"q_{a}": res["q"][a][:, e].numpy().astype(np.float32)
               for a in ARM_KEYS},
            **{f"margin_{k}": margins[k][:, e].numpy().astype(np.float32)
               for k in CLASS_KEYS},
            **{f"viol_{k}": viol[k][:, e].numpy() for k in CLASS_KEYS},
            table_exempt=res["table_exempt"][:, e].to(torch.bool).numpy(),
            alpha=res["alpha_by_step"][:, e].numpy().astype(np.float32),
            alpha_policy=(res["alpha_policy_by_step"][:, e].numpy()
                          .astype(np.float32)),
            hazard_prob=(res["hazard_prob_by_step"][:, e].numpy()
                         .astype(np.float32)),
            engaged=res["engaged_by_step"][:, e].to(torch.bool).numpy(),
            bypass_arm=(res["bypass_arm_by_step"][:, e].to(torch.bool)
                        .numpy()),
            bypass_blocked_cls=(res["bypass_blocked_cls_by_step"][:, e]
                                .to(torch.bool).numpy()))
        if has_causal_trace:
            payload.update({
                key: res[f"{key}_by_step"][:, e].numpy()
                for key in causal_step_names
            })
            n_transitions = res["alpha_by_step"].shape[0]
            payload["transition_id"] = np.arange(n_transitions, dtype=np.int64)
            payload["pre_state_id"] = payload["transition_id"].copy()
            payload["post_state_id"] = payload["transition_id"] + 1
            for key in causal_static_names:
                value = res[key]
                payload[key] = (value.numpy() if isinstance(value, torch.Tensor)
                                else np.asarray(value, dtype=np.str_))
        np.savez_compressed(npz_path, **payload)
        (out_dir / f"{name}_stats.json").write_text(json.dumps({
            "npz": npz_path.name, "dt": float(dt),
            "margin_by_class": {
                k: [float(v) for v in margins[k][:, e]] for k in CLASS_KEYS},
            "alpha": {
                a: [float(v) for v in res["alpha_by_step"][:, e, i]]
                for i, a in enumerate(ARM_KEYS)},
        }))
        paths.append(npz_path)
    return paths


def run_rows_from_margins(margins: dict, dt: float, meta: "dict | None" = None,
                          extra_per_env: "dict | None" = None,
                          table_viol: "torch.Tensor | None" = None,
                          table_exempt: "torch.Tensor | None" = None) -> list:
    """margins: {class: (T, N) min-margin trace, post-step} -> one row per env.

    A violation step is a control step whose post-step sphere state has any
    class violation under the OFFICIAL exempt-aware semantics; per-class
    counts use the same traces so the total is exactly the any-class union
    (multi-class steps count once in the total). success = zero violation
    steps across ALL classes (spec section 2 strict caliber);
    success_gt_verified counts GT-verified classes only (diagnostic column
    for the v4 self_U artifact, PROTOCOL_V5 abolishes the need on v5).

    Caliber fix (2026-08-15 C11): cross/self_F/self_U stay margin<0 (no
    exemption exists for them, raw IS official). The table channel consumes
    ``table_viol`` -- the (T, N) bool trace of the channel-level exempt-aware
    flag (any table pair margin<0 AND NOT near_table-exempt, i.e. the table
    component of SphereDistOut.violation; collect it in run_window via
    table_flags_from_module). ``table_exempt`` is the (T, N) transparency
    trace (any table pair margin<0 AND exempt = legal hand-on-own-table
    contact) summed into ``table_exempt_steps`` per row. Both calibers land
    in every row: primary fields are official, the old exemption-blind
    counts keep flowing under ``*_raw`` so historical comparisons never
    break; ``table_caliber`` records which path produced the row. Margin-only
    callers (recuts of legacy records that stored no exemption trace) omit
    the traces and get ``table_caliber="raw_fallback"`` with official==raw --
    that output stays exemption-blind on table and must be labeled as such.
    ``worst_margin_by_class`` stays the raw sphere-shell readout on purpose
    (an exempted legal contact still reads a negative table margin; the
    exempt columns explain it away instead of hiding it).
    """
    T, N = margins[CLASS_KEYS[0]].shape
    stack = torch.stack([margins[k] for k in CLASS_KEYS])          # (C, T, N)
    viol_raw = stack < 0.0                    # (C, T, N) exemption-blind (old)
    viol = viol_raw.clone()                   # (C, T, N) official exempt-aware
    t_idx = CLASS_KEYS.index("table")
    exempt_aware = table_viol is not None
    if exempt_aware:
        table_viol = table_viol.to(dtype=torch.bool)
        assert tuple(table_viol.shape) == (T, N), \
            f"table_viol trace shape {tuple(table_viol.shape)} != {(T, N)}"
        # a non-exempt negative pair forces the raw class min-margin negative,
        # so official table violations must be a subset of raw ones; anything
        # else means the trace does not belong to these margins -- refuse
        assert not bool((table_viol & ~viol_raw[t_idx]).any()), \
            "table_viol=True where raw table margin >= 0 -- trace mismatch"
        viol[t_idx] = table_viol
    if table_exempt is not None:
        table_exempt = table_exempt.to(dtype=torch.bool)
        assert tuple(table_exempt.shape) == (T, N), \
            f"table_exempt trace shape {tuple(table_exempt.shape)} != {(T, N)}"
    any_viol = viol.any(dim=0)                                     # (T, N)
    any_raw = viol_raw.any(dim=0)                                  # (T, N)
    gt_idx = [CLASS_KEYS.index(k) for k in GT_VERIFIED_KEYS]
    gt_viol = viol[gt_idx].any(dim=0)                              # (T, N)
    gt_raw = viol_raw[gt_idx].any(dim=0)                           # (T, N)
    rows = []
    for e in range(N):
        n_viol = int(any_viol[:, e].sum())
        n_raw = int(any_raw[:, e].sum())
        first = int(any_viol[:, e].float().argmax()) if n_viol else -1
        first_raw = int(any_raw[:, e].float().argmax()) if n_raw else -1
        row = {
            "env_idx": e,
            "steps": T,
            "duration_s": round(T * dt, 3),
            "violation_steps": n_viol,
            "violation_steps_raw": n_raw,
            "violation_steps_by_class": {
                k: int(viol[i, :, e].sum()) for i, k in enumerate(CLASS_KEYS)},
            "violation_steps_by_class_raw": {
                k: int(viol_raw[i, :, e].sum())
                for i, k in enumerate(CLASS_KEYS)},
            "worst_margin_by_class": {
                k: round(float(stack[i, :, e].min()), 5)
                for i, k in enumerate(CLASS_KEYS)},
            "first_violation_step": first,
            "first_violation_step_raw": first_raw,
            "success": bool(n_viol == 0),
            "success_raw": bool(n_raw == 0),
            "violation_steps_gt_verified": int(gt_viol[:, e].sum()),
            "violation_steps_gt_verified_raw": int(gt_raw[:, e].sum()),
            "success_gt_verified": bool(int(gt_viol[:, e].sum()) == 0),
            "success_gt_verified_raw": bool(int(gt_raw[:, e].sum()) == 0),
            "table_exempt_steps": (int(table_exempt[:, e].sum())
                                   if table_exempt is not None else 0),
            "table_caliber": "exempt_aware" if exempt_aware else "raw_fallback",
        }
        if meta:
            row.update(meta)
        if extra_per_env:
            for key, vals in extra_per_env.items():
                v = vals[e]
                row[key] = round(float(v), 5) if isinstance(v, float) else v
        rows.append(row)
    return rows


def aggregate_success(rows: list, conf: float = 0.95) -> dict:
    """Success-rate aggregate with two-sided Clopper-Pearson CI (both the
    strict caliber and the GT-verified diagnostic), plus violation-step dose
    stats and per-class run/step tallies.

    Dual caliber (C11): primary keys aggregate the official exempt-aware
    fields; the ``*_raw`` keys aggregate the old exemption-blind counts and
    ``table_exempt_steps`` summarizes the transparency column. Legacy rows
    (pre-fix JSONs, where the primary fields WERE the raw caliber) re-
    aggregate fine: every raw/exempt lookup falls back to the primary field,
    so official==raw there, with ``table_caliber`` reporting raw_fallback.
    """
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    k_raw = sum(1 for r in rows if r.get("success_raw", r["success"]))
    k_gt = sum(1 for r in rows if r["success_gt_verified"])
    vsteps = torch.tensor([float(r["violation_steps"]) for r in rows]) \
        if rows else torch.zeros(0)
    vsteps_raw = torch.tensor(
        [float(r.get("violation_steps_raw", r["violation_steps"]))
         for r in rows]) if rows else torch.zeros(0)
    ex_steps = [int(r.get("table_exempt_steps", 0)) for r in rows]
    calibers = sorted({str(r.get("table_caliber", "raw_fallback"))
                       for r in rows})

    def _dose(t):
        return {
            "mean": round(float(t.mean()), 3) if n else 0.0,
            "max": int(t.max()) if n else 0,
            "median": round(float(t.median()), 1) if n else 0.0,
        }

    def _per_class(field, fallback):
        return {
            key: {
                "runs_with_violation": sum(
                    1 for r in rows
                    if r.get(field, r[fallback])[key] > 0),
                "total_violation_steps": sum(
                    r.get(field, r[fallback])[key] for r in rows),
            } for key in CLASS_KEYS
        }

    agg = {
        "n_runs": n,
        "n_success": k,
        "success_rate": (k / n) if n else 0.0,
        "success_ci95": list(clopper_pearson_interval(k, n, conf)),
        "n_success_raw": k_raw,
        "success_rate_raw": (k_raw / n) if n else 0.0,
        "success_raw_ci95": list(clopper_pearson_interval(k_raw, n, conf)),
        "n_success_gt_verified": k_gt,
        "success_rate_gt_verified": (k_gt / n) if n else 0.0,
        "success_gt_ci95": list(clopper_pearson_interval(k_gt, n, conf)),
        "conf": conf,
        "violation_steps": _dose(vsteps),
        "violation_steps_raw": _dose(vsteps_raw),
        "per_class": _per_class("violation_steps_by_class",
                                "violation_steps_by_class"),
        "per_class_raw": _per_class("violation_steps_by_class_raw",
                                    "violation_steps_by_class"),
        "table_exempt_steps": {
            "runs_with_exempt_contact": sum(1 for s in ex_steps if s > 0),
            "total_steps": sum(ex_steps),
            "mean": round(sum(ex_steps) / n, 3) if n else 0.0,
        },
        "table_caliber": calibers[0] if len(calibers) == 1 else "mixed",
    }
    return agg


# --------------------------------------------------------------------------
# Isaac side
# --------------------------------------------------------------------------

class _RawShim:
    """No-safety passthrough backstop (reference rows only)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def project(self, cmd, rows, alpha, p, dt, dmin=None, backlog=None, **kw):
        from safeduo.delta._contract_stub import ARM_KEYS

        act = torch.zeros(alpha.shape[0], len(ARM_KEYS),
                          dtype=torch.bool, device=alpha.device)
        # DuoEnv records these two keys unconditionally for its causal trace.
        # Raw mode has no projection residual; explicit zeros preserve that
        # meaning while keeping the shared telemetry path total.
        residual = torch.zeros(alpha.shape[0], dtype=alpha.dtype,
                               device=alpha.device)
        return cmd, act, {"residual_F": residual, "residual_U": residual.clone()}


def source_families(src) -> "list | None":
    """Per-env family names from any source exposing the .names/.assignment
    duck-typed protocol (ConflictMixSource scenario families, SkillReplayDelta
    skills); None when the source has no family notion (l1/l1_ws)."""
    if hasattr(src, "assignment") and hasattr(src, "names"):
        return [src.names[int(i)] for i in src.assignment.cpu()]
    return None


def ckpt_arm_aware(path: "str | Path") -> "bool | None":
    """True if the checkpoint was trained with arm-aware pair obs (R15).

    Reads config.arm_aware_obs saved by LagrangianPPO.save(); returns None
    (leave the env yaml untouched) for non-lagrangian or metadata-less
    checkpoints so pre-R15 behavior is bit-identical.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        flag = (ck.get("config") or {}).get("arm_aware_obs")
        if flag is not None:
            return bool(flag) or None
    return None


def ckpt_p2_obs(path: "str | Path") -> "bool | None":
    """True if the checkpoint was trained with the P2 obs tail block (R22).

    Reads config.p2_obs saved by LagrangianPPO.save(); returns None for
    pre-R22 or metadata-less checkpoints so their eval path stays
    bit-identical (mirror of ckpt_arm_aware).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        flag = (ck.get("config") or {}).get("p2_obs")
        if flag is not None:
            return bool(flag) or None
    return None


def make_endurance_flow(flow: str, env, amp: float, env_yaml: str,
                        wild: bool = True, ee_speed: "float | None" = None,
                        skill_dir: str = "artifacts/skill_trajs",
                        skill_tier: "int | None" = None,
                        geometry: str = "v5"):
    """Build the delta source for one window at one amp.

    Any source exposing .assignment (per-env long) + .names (list) gets its
    per-env family logged into the run rows (source_families) --
    ConflictMixSource and SkillReplayDelta both follow the duck-typed
    protocol, so directed and skill flows produce family-bucketed rows for
    free.

    geometry="v7" (R15 eval wiring, 2026-08-20): l1_ws swaps in
    WorkspaceRoamDeltaV7 (UR5 FK chain + S4-calibrated boxes) with the
    V7_ROAM_PROFILE stream (the v5 wild profile's 0.8-2.0s segments never
    finish v7's longer transits -- in-band dwell drops to ~5%); directed
    injects the curriculum geometry key. Default "v5" is bit-identical to
    the pre-R15 behavior.
    """
    if flow == "l1_full":
        # R25 coverage stream (delta/l1_coverage.py). v7-only: the full-domain
        # boxes are reach-derived from the v7 layout (full_boxes_v7); there is
        # no v5 calibration, so refuse loudly instead of roaming a wrong box.
        if geometry != "v7":
            raise ValueError("--flow l1_full requires --geometry v7 "
                             "(only the v7 full-domain boxes are calibrated)")
        from safeduo.delta.l1_coverage import make_l1_full_v7

        return make_l1_full_v7(env.num_envs, amp_max=float(amp),
                               device=env.device, env_yaml=env_yaml)
    if flow in ("l1", "l1_ws"):
        from safeduo.delta.l1_random import L1Params, L1RandomDelta

        p = L1Params(amp_max=float(amp))
        if wild:
            # the endurance stream profile (support-line l1_wild / demo l1):
            # nonstop roaming, short segments, extra OU jitter
            p.pause_prob = 0.0
            p.seg_dur = (0.8, 2.0)
            p.ou_sigma = 0.02
        if flow == "l1_ws":
            if geometry == "v7":
                from safeduo.delta.l1_workspace_v7 import (
                    V7_ROAM_PROFILE,
                    WorkspaceRoamDeltaV7,
                    v7_roam_params,
                )

                return WorkspaceRoamDeltaV7(
                    env.num_envs, params=v7_roam_params(float(amp)),
                    device=env.device, env_yaml=env_yaml,
                    ee_speed=float(V7_ROAM_PROFILE["ee_speed"]))
            from safeduo.delta.l1_workspace import WorkspaceRoamDelta

            return WorkspaceRoamDelta(env.num_envs, params=p,
                                      device=env.device, env_yaml=env_yaml)
        return L1RandomDelta(env.num_envs, params=p, device=env.device)
    if flow == "directed":
        from safeduo.delta.l2_env_source import ConflictMixSource

        cfg = {"mix": {"l1": 0.0, "l2": 1.0},
               "scenarios": {"head_on_crossing": 1.0,
                             "handover_approach": 1.0},
               "split": "eval", "n_variants": 100, "stages": None,
               "geometry": geometry}
        src = ConflictMixSource(env.num_envs, cfg, device=env.device,
                                amp_max=float(amp), env_yaml=env_yaml)
        if ee_speed is not None:
            for holder in [src] + list(src.sources):
                m = getattr(holder, "mapper", None) or getattr(holder, "_mapper", None)
                if m is not None and hasattr(m, "ee_speed"):
                    m.ee_speed = float(ee_speed)
        return src
    if flow == "skill":
        from safeduo.delta.skill_replay import (
            SkillNoiseParams,
            SkillReplayDelta,
            load_manifest_library,
        )

        lib = load_manifest_library(Path(skill_dir) / "skills_manifest.json")
        tier = skill_tier_for_amp(amp) if skill_tier is None else int(skill_tier)
        params = SkillNoiseParams.tier(tier)
        # amp keeps its endurance meaning as the window's per-step delta cap
        # (l1/directed use it the same way); recordings peak at ~0.014
        # rad/step, so amp >= 0.015 replays uncropped
        params.amp_max = float(amp)
        # skills last 15-17.5 s; loop so a 60-600 s reset-free window stays a
        # nonstop stream (each lap restarts from the tier's randomized start
        # phase; the env's skill -- and the run's family -- never changes)
        return SkillReplayDelta(env.num_envs, lib, params=params,
                                device=env.device, loop=True)
    raise ValueError(f"unknown flow {flow}")


def control_trace_step(driver, step_cache: dict) -> dict:
    """Freeze one control decision for S18 offline-theta replay.

    The environment cache holds the gate that was actually applied.  A
    ClutchDriver additionally exposes the raw detector output and the latch
    state that produced that gate.  Without a clutch wrapper, the historical
    reference semantics from clutch_eval are used: applied alpha is the raw
    policy score and an arm is considered engaged only at passthrough alpha.

    R19 ``bypass_blocked_cls`` is intentionally recorded separately from the
    actual bypass mask: it depends on geometry/semantics but not on the latch,
    which is what makes a counterfactual theta replay possible.  Returned
    tensors are detached CPU clones so later simulator steps cannot mutate
    the evidence.
    """
    from safeduo.delta._contract_stub import ARM_KEYS

    alpha_exec = step_cache["alpha"]
    alpha_policy = getattr(driver, "last_alpha_policy", None)
    if alpha_policy is None:
        alpha_policy = alpha_exec
    engaged = getattr(driver, "engaged", None)
    if engaged is None:
        engaged = alpha_exec > 1.0 - 1e-6

    n, n_arms = alpha_exec.shape
    if n_arms != len(ARM_KEYS):
        raise ValueError(f"alpha shape {tuple(alpha_exec.shape)} has "
                         f"{n_arms} arms, expected {len(ARM_KEYS)}")
    if tuple(alpha_policy.shape) != (n, n_arms):
        raise ValueError(f"alpha_policy shape {tuple(alpha_policy.shape)} "
                         f"!= {(n, n_arms)}")
    if tuple(engaged.shape) != (n, n_arms):
        raise ValueError(f"engaged shape {tuple(engaged.shape)} "
                         f"!= {(n, n_arms)}")
    if bool(((alpha_policy < 0.0) | (alpha_policy > 1.0)).any()):
        raise ValueError("alpha_policy must stay in [0, 1]")

    bypass = step_cache.get("bypass_arm")
    if bypass is None:
        bypass = torch.zeros(n, n_arms, dtype=torch.bool,
                             device=alpha_exec.device)
    blocked = step_cache.get("bypass_blocked_cls")
    if blocked is None:
        blocked = torch.zeros(n, n_arms, 3, 2, dtype=torch.bool,
                              device=alpha_exec.device)
    if tuple(bypass.shape) != (n, n_arms):
        raise ValueError(f"bypass_arm shape {tuple(bypass.shape)} "
                         f"!= {(n, n_arms)}")
    if tuple(blocked.shape) != (n, n_arms, 3, 2):
        raise ValueError(f"bypass_blocked_cls shape {tuple(blocked.shape)} "
                         f"!= {(n, n_arms, 3, 2)}")

    def frozen(x):
        return x.detach().cpu().clone()

    policy = frozen(alpha_policy)
    return {
        "alpha_policy": policy,
        "hazard_prob": 1.0 - policy,
        "engaged": frozen(engaged).to(torch.bool),
        "bypass_arm": frozen(bypass).to(torch.bool),
        "bypass_blocked_cls": frozen(blocked).to(torch.bool),
    }


def control_transition_pre_step(env) -> tuple[dict, dict]:
    """Freeze the state and geometry used to decide one control transition.

    This is an evaluation-only probe called after ``driver.act`` and directly
    before ``env.step``.  It intentionally recomputes only the constraint-row
    view from the already-computed sphere state; it neither changes the delta
    source nor writes environment state.  Pairing its result with
    :func:`control_transition_post_step` gives the unambiguous causal sample
    ``pre[t] -> action[t] -> post[t]``.

    The first result contains tensors with a leading environment dimension.
    The second contains static table-pair identity shared by all environments
    and all steps in the window.  All tensors are detached CPU clones so the
    PhysX buffers cannot rewrite recorded evidence during ``env.step``.
    """
    from safeduo.delta._contract_stub import ARM_KEYS

    def frozen(x):
        return x.detach().cpu().clone()

    state = env.scene_state()
    out = env._last_out
    if out is None:
        raise RuntimeError("control trace requires a computed pre-step sphere state")
    rows = env._provider.rows_from(out, env._body_pos_cache)
    sph = env._sph
    if sph.last_table_margin is None or sph.last_table_viol_exempt is None:
        raise RuntimeError("control trace requires pre-step table-pair evidence")

    step = {
        "pre_active_pair_id": frozen(out.active_idx).to(torch.int64),
        "pre_active_margin": frozen(rows.d),
        "pre_active_closing": frozen(out.active_pairs[..., 1]),
        "pre_active_class": frozen(rows.cls).to(torch.int64),
        "pre_active_dmin": frozen(out.active_dmin),
        "pre_active_exempt": frozen(out.viol_exempt).to(torch.bool),
        "pre_active_mask": frozen(rows.valid).to(torch.bool),
        "pre_active_arm_mask": frozen(rows.arm_mask).to(torch.bool),
        "pre_active_J_F": frozen(rows.J["F"]),
        "pre_active_J_U": frozen(rows.J["U"]),
        "pre_table_margin_by_pair": frozen(sph.last_table_margin),
        "pre_table_exempt_by_pair": frozen(sph.last_table_viol_exempt).to(torch.bool),
    }
    for arm in ARM_KEYS:
        q = state.q[arm]
        target = env._targets[arm]
        step[f"pre_q_{arm}"] = frozen(q)
        step[f"pre_qd_{arm}"] = frozen(state.qd[arm])
        step[f"pre_target_{arm}"] = frozen(target)
        step[f"pre_backlog_{arm}"] = frozen(target - q)

    sl = sph._slice_table
    pair = sph.pair_table[sl]
    sphere_i = pair[:, 0]
    static = {
        "table_pair_id": frozen(torch.arange(sl.start, sl.stop, device=pair.device)),
        "table_pair_sphere_i": frozen(sphere_i).to(torch.int64),
        "table_pair_table_id": frozen(pair[:, 1]).to(torch.int64),
        "table_pair_arm": frozen(sph.arm_id[sphere_i]).to(torch.int64),
        "table_pair_dmin": frozen(sph.pair_dmin[sl]),
        "table_pair_link_name": [sph.qualified_names[int(i)] for i in sphere_i.tolist()],
    }
    return step, static


def control_transition_post_step(env, step_cache: dict, pre: dict) -> dict:
    """Complete one aligned causal trace after the matching ``env.step``."""
    from safeduo.delta._contract_stub import ARM_KEYS

    def frozen(x):
        return x.detach().cpu().clone()

    state = env.scene_state()
    sph = env._sph
    if sph.last_table_margin is None or sph.last_table_viol_exempt is None:
        raise RuntimeError("control trace requires post-step table-pair evidence")
    result = dict(pre)
    result.update({
        "bs_active": frozen(step_cache["bs_active"]).to(torch.bool),
        "priority_p": frozen(step_cache["p"]),
        "backstop_residual_F": frozen(step_cache["backstop_residual_F"]),
        "backstop_residual_U": frozen(step_cache["backstop_residual_U"]),
        "post_table_margin_by_pair": frozen(sph.last_table_margin),
        "post_table_exempt_by_pair": frozen(sph.last_table_viol_exempt).to(torch.bool),
    })
    for arm in ARM_KEYS:
        q = state.q[arm]
        target = env._targets[arm]
        result[f"cmd_{arm}"] = frozen(step_cache["cmd"].delta_q[arm])
        result[f"exec_{arm}"] = frozen(step_cache["exec"].delta_q[arm])
        result[f"post_qd_{arm}"] = frozen(state.qd[arm])
        result[f"post_target_{arm}"] = frozen(target)
        result[f"post_backlog_{arm}"] = frozen(target - q)
    return result


def run_window(env, driver, steps: int, log_every: int = 600,
               ee_tracker=None, pair_recorder: "SelfUNegativePairRecorder | None" = None,
               run_id_base: int = 0, post_reset=None,
               record_traj: bool = False,
               record_causal_trace: bool = False) -> dict:
    """Drive one reset-free window; the caller has already reseeded + swapped
    the delta source. Returns per-class margin traces (T, N), the exempt-
    aware table-channel flag traces (T, N) bool (C11 caliber fix: table_viol
    = official table violation, table_exempt = exempted legal contact; both
    read off the same post-step compute as the margins), plus per-env alpha
    mean and tube fraction diagnostics.

    ``ee_tracker`` (T4 R5, additive): optional skill_success.EETraceTracker;
    when given, the post-step EXECUTED joint state (env.scene_state().q) is
    recorded every step so the skill flow can judge per-run task completion
    (EE waypoint criteria) after the window. None = bit-identical behavior.

    ``pair_recorder`` (R11, additive): optional full-pair self_U argmin
    recorder.  ``None`` is the default and preserves the historical output
    keys and numerical path.  When enabled, evidence is buffered on-device
    and materialized once after the complete window.

    ``record_traj`` (S13/S18, additive): when True the post-step EXECUTED
    joint state and offline-theta traces are buffered on CPU for
    dump_traj_npz.  ``record_causal_trace`` adds the much larger S19 diagnostic
    group, explicitly pairing the pre-state and constraint rows used for
    action ``t`` with that action's command/execution cache and post-state. It
    requires record_traj=True and stays off for ordinary render trajectories
    and long endurance windows.
    """
    margins = {k: [] for k in CLASS_KEYS}
    table_viol, table_exempt = [], []
    q_steps: "dict | None" = None
    alpha_steps: "list | None" = None
    control_steps: "dict | None" = None
    control_static: "dict | None" = None
    if record_causal_trace and not record_traj:
        raise ValueError("record_causal_trace requires record_traj=True")
    if record_traj:
        from safeduo.delta._contract_stub import ARM_KEYS

        q_steps = {a: [] for a in ARM_KEYS}
        alpha_steps = []
        control_steps = {}
    alpha_sum = torch.zeros(env.num_envs)
    tube_sum = torch.zeros(env.num_envs)
    obs, _ = env.reset()
    driver.reset(torch.arange(env.num_envs, device=env.device))
    if post_reset is not None:
        # R15 skill-q0 teleport hook (additive, None = bit-identical): the
        # v7 skill library starts from its own designed q0, not the env birth
        # pose (S4/S5 landed in parallel), and SkillReplayDelta is a pure
        # delta stream -- replaying from birth shifts every EE waypoint by
        # the origin gap (battery7 first-fire: 100/100 "wp0 unreached").
        # The first driver action still sees the pre-teleport obs (one 60 Hz
        # step of staleness, no violation exposure at rest).
        post_reset(env)
    if pair_recorder is not None:
        pair_recorder.start_window(run_id_base)
    t0 = time.time()
    try:
        for t in range(steps):
            action = driver.act(env, obs)
            pre_trace = None
            if record_causal_trace:
                pre_trace, static = control_transition_pre_step(env)
                if control_static is None:
                    control_static = static
                else:
                    for key in ("table_pair_id", "table_pair_sphere_i",
                                "table_pair_table_id", "table_pair_arm",
                                "table_pair_dmin"):
                        if not torch.equal(control_static[key], static[key]):
                            raise RuntimeError(
                                f"static control-trace field '{key}' changed within window")
                    if control_static["table_pair_link_name"] != static["table_pair_link_name"]:
                        raise RuntimeError(
                            "static control-trace table link names changed within window")
            obs, _, term, trunc, _ = env.step(action)
            done = term | trunc
            if bool(done.any()):
                # reset-free precondition broken (timeout inside the window or
                # violation termination left on) -- runs would silently splice
                # episodes; refuse to produce numbers instead
                raise RuntimeError(
                    f"env reset inside an endurance window at step {t} "
                    f"(terminated={int(term.sum())} truncated={int(trunc.sum())})"
                    " -- check terminate_on_violation=False and episode_length_s")
            mm = env._last_out.min_margin
            for k in CLASS_KEYS:
                margins[k].append(mm[k].detach().cpu().clone())
            tv, te = table_flags_from_module(env._sph)
            table_viol.append(tv.detach().cpu().clone())
            table_exempt.append(te.detach().cpu().clone())
            if pair_recorder is not None:
                pair_recorder.capture_step()
            if ee_tracker is not None:
                ee_tracker.record(env.scene_state().q)
            c = env._step_cache
            if record_traj:
                st_q = env.scene_state().q
                for a in q_steps:
                    q_steps[a].append(st_q[a].detach().cpu().clone())
                alpha_steps.append(c["alpha"].detach().cpu().clone())
                trace = control_trace_step(driver, c)
                if record_causal_trace:
                    trace.update(control_transition_post_step(env, c, pre_trace))
                for key, value in trace.items():
                    control_steps.setdefault(key, []).append(value)
            alpha_sum += c["alpha"].detach().cpu().mean(dim=-1)
            tube_sum += c["tube"].detach().cpu().float()
            if t % log_every == 0:
                rate = (t + 1) / max(time.time() - t0, 1e-9)
                print(f"[endurance] step {t}/{steps} ({rate:.1f} steps/s, "
                      f"eta {((steps - t) / max(rate, 1e-9)):.0f}s)", flush=True)
    except BaseException:
        if pair_recorder is not None:
            pair_recorder.abort_window()
        raise
    if pair_recorder is not None:
        pair_recorder.finish_window()
    out = {
        "margins": {k: torch.stack(v) for k, v in margins.items()},
        "table_viol": torch.stack(table_viol),
        "table_exempt": torch.stack(table_exempt),
        "alpha_mean": alpha_sum / steps,
        "tube_fraction": tube_sum / steps,
        "wall_s": time.time() - t0,
    }
    if record_traj:
        out["q"] = {a: torch.stack(v) for a, v in q_steps.items()}
        out["alpha_by_step"] = torch.stack(alpha_steps)
        for key, values in control_steps.items():
            out[f"{key}_by_step"] = torch.stack(values)
        if record_causal_trace:
            out.update(control_static)
    return out


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["passthrough"],
                        choices=list(METHOD_CHOICES))
    parser.add_argument("--ckpt", type=str, default="",
                        help="checkpoint for the safeduo method (rsl_rl PPO "
                             "and lagrangian Beta-actor formats)")
    parser.add_argument("--flow", choices=list(FLOW_CHOICES), default="l1")
    parser.add_argument("--duration-s", type=float, required=True,
                        help="continuous run duration per env (60 / 300 per "
                             "spec; 600 = the T2-B 10 min endurance tier, "
                             "see DURATION_TIERS)")
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--runs", type=int, default=0,
                        help="total runs (rounded up to whole windows of "
                             "num-envs); default = one window per amp")
    parser.add_argument("--amp", type=float, default=0.03)
    parser.add_argument("--amps", nargs="+", type=float, default=None,
                        help="explicit amp list, one window per amp")
    parser.add_argument("--tier", choices=list(TIER_AMPS), default=None,
                        help="amp-tier preset (overrides --amp/--amps)")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--tame", action="store_true",
                        help="keep L1Params defaults (pauses on) instead of "
                             "the nonstop endurance profile")
    parser.add_argument("--ee-speed", type=float, default=None,
                        help="optional EE speed crank for the directed flow")
    parser.add_argument("--skill-tier", type=int, choices=(0, 1, 2, 3),
                        default=None,
                        help="T2 noise tier for --flow skill (0 = exact "
                             "replay ... 3 = aggressive; SkillNoiseParams"
                             ".TIERS). Default: derived per window from that "
                             "window's amp via skill_tier_for_amp (<0.015 -> "
                             "0, <0.03 -> 1, <0.06 -> 2, else 3). Independent "
                             "of --tier, which stays the AMP preset picker; "
                             "in the skill flow amp acts as the per-step "
                             "delta cap (recordings peak ~0.014 rad/step, so "
                             "amp >= 0.015 replays uncropped)")
    parser.add_argument("--skill-dir", type=str,
                        default="artifacts/skill_trajs",
                        help="skill library directory containing "
                             "skills_manifest.json (--flow skill)")
    parser.add_argument("--skill-q0-teleport", action="store_true",
                        help="teleport arms to each env's assigned skill "
                             "start pose after the window reset (v7 caliber: "
                             "the skill library q0 differs from the env "
                             "birth pose; default off = pre-R15 behavior)")
    parser.add_argument("--env-yaml", type=str, default="duo_env_v4.yaml")
    parser.add_argument("--geometry", type=str, choices=("v5", "v7"),
                        default="v5",
                        help="flow-source scene family: v7 gives l1_ws the "
                             "UR5-chain WorkspaceRoamDeltaV7 + S4 boxes and "
                             "V7_ROAM_PROFILE stream, and directed the v7 "
                             "curriculum geometry (default v5 = pre-R15)")
    parser.add_argument("--frozen-baselines", action="store_true",
                        help="use the grid-winner working points for "
                             "estop/speed/cbf (block1 BASELINE_FROZEN_W5; "
                             "v5 re-scan winners via --baseline-cfg)")
    parser.add_argument("--baseline-cfg", type=str, default="",
                        help='JSON ctor overrides {"estop": {...}, ...}')
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--clutch", action="store_true",
                        help="R17 execution semantics: per-arm hysteresis "
                             "binarization of the policy alpha (ENGAGED = "
                             "bitwise passthrough, LOCKED = frozen arm); "
                             "default off = bit-identical continuous gating "
                             "(eval/clutch.py, tests/test_clutch.py)")
    parser.add_argument("--theta-hi", type=float, default=0.7,
                        help="clutch re-engage threshold (alpha > theta_hi "
                             "unlocks a LOCKED arm)")
    parser.add_argument("--theta-lo", type=float, default=0.4,
                        help="clutch lock threshold (alpha < theta_lo "
                             "freezes an ENGAGED arm)")
    parser.add_argument("--bypass-mm", type=float, default=None,
                        help="R19 execution-layer bypass: emergency band "
                             "width (mm) above the per-row lock line; "
                             "engaged arms clear of the band pass through "
                             "bitwise (analytic stack fully bypassed). "
                             "Default None = off = bit-identical")
    parser.add_argument(
        "--pair-recorder",
        action="store_true",
        help="write negative self_U argmin pair evidence to the additive "
             "negative_pair_records_<method>.jsonl sidecar (default off)",
    )
    parser.add_argument(
        "--dump-traj-npz", type=str, default=None, metavar="DIR",
        help="write per-env per-window PPU-renderer trajectory npz "
             "(<method>_run<id>.npz, S11 tools/ppu_render schema: executed "
             "q_<arm>/dt/meta + per-step margins, official violation flags "
             "and alpha) plus the <name>_stats.json render-overlay sidecar "
             "into DIR (default off = no extra buffers, established outputs "
             "unchanged)",
    )
    parser.add_argument(
        "--causal-trace",
        action="store_true",
        help="with --dump-traj-npz, add the large S19 pre/action/post "
             "constraint trace for short root-cause runs; keep off for "
             "ordinary videos and long endurance windows",
    )
    parser.add_argument("--out", type=str, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.causal_trace and not args.dump_traj_npz:
        parser.error("--causal-trace requires --dump-traj-npz DIR")
    app = AppLauncher(args).app  # noqa: F841

    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.eval.block1_harness import (
        BASELINE_FROZEN_W5,
        PassthroughDriver,
        PolicyDriver,
        make_baseline_driver,
    )
    from safeduo.eval.success_report import render_endurance_report

    if args.tier:
        amps = list(TIER_AMPS[args.tier])
    elif args.amps:
        amps = [float(a) for a in args.amps]
    else:
        amps = [float(args.amp)]
    if args.runs > 0:
        n_windows = max(1, -(-args.runs // args.num_envs))
        amps = [amps[w % len(amps)] for w in range(n_windows)]
    ctrl_steps = None

    overrides = {}
    if args.frozen_baselines:
        overrides = {k: dict(v) for k, v in BASELINE_FROZEN_W5.items()}
    for k, v in (json.loads(args.baseline_cfg) if args.baseline_cfg else {}).items():
        overrides[k] = {**overrides.get(k, {}), **v}

    # R15 eval wiring (2026-08-20): an --arm-aware-obs checkpoint carries
    # 13-dim pair rows (obs 531); the env must be built with the matching
    # caliber or the actor rejects the obs outright (battery7 first-fire
    # lesson: the crash even hides behind Isaac's rc=0 hard exit). The flag
    # is read from the checkpoint itself; v6-era ckpts keep the default path.
    arm_aware = ckpt_arm_aware(args.ckpt) if args.ckpt else None
    # R22 P2 eval wiring: a --p2-obs checkpoint expects the 16-dim obs tail
    # (531 -> 547); same self-describing-ckpt pattern as arm_aware above.
    p2 = ckpt_p2_obs(args.ckpt) if args.ckpt else None
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device,
                           yaml_name=args.env_yaml, coordinator=True,
                           arm_aware_obs=arm_aware, p2_obs=p2)
    # reset-free: no violation termination (parity-v3.1 measurement flag),
    # no timeout inside the window
    cfg.coordinator["terminate_on_violation"] = False
    ctrl_dt = cfg.sim.dt * cfg.decimation
    cfg.episode_length_s = float(args.duration_s) + 5.0
    cfg.seed = args.seed
    env = DuoEnv(cfg)
    if args.bypass_mm is not None:
        env.set_r19_bypass_mm(args.bypass_mm)
        print(f"[endurance] R19 bypass ON, emergency band = "
              f"{args.bypass_mm} mm", flush=True)
    ctrl_steps = int(round(args.duration_s / ctrl_dt))

    # T4 R5 (additive): skill flow judges per-run TASK completion (EE
    # waypoint criteria, configs/skill_success.yaml) on top of the safety
    # caliber. The window loops the skill; task_success = at least one
    # complete ordered waypoint pass of the executed (gated) motion.
    skill_judge = None
    if args.flow == "skill":
        if args.geometry == "v7":
            # v7 caliber: V7SkillFK + skill_success_v7.yaml thresholds
            # (S5 recalibration); the judge class itself is unchanged
            from safeduo.eval.skill_success_v7 import v7_judge

            skill_judge = v7_judge(skill_dir=args.skill_dir)
        else:
            from safeduo.eval.skill_success import SkillSuccessJudge

            skill_judge = SkillSuccessJudge(skill_dir=args.skill_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for method in args.methods:
        if method == "safeduo":
            if not args.ckpt:
                raise SystemExit("--ckpt required for the safeduo method")
            driver = PolicyDriver(args.ckpt, device=str(env.device))
        elif method in ("passthrough", "raw"):
            driver = PassthroughDriver()
        else:
            driver = make_baseline_driver(method, env, overrides.get(method))
        if args.clutch:
            from safeduo.eval.clutch import wrap_clutch

            driver = wrap_clutch(driver, True, args.theta_hi, args.theta_lo,
                                 env.num_envs, env.device)
        rows = []
        pair_recorder = (SelfUNegativePairRecorder(env._sph)
                         if args.pair_recorder else None)
        raw_shim = None
        if method == "raw":
            raw_shim = _RawShim(env._backstop)
            env._backstop = raw_shim
        try:
            for w, amp in enumerate(amps):
                seed_w = args.seed + w
                env._gen.manual_seed(seed_w)
                env._pending_cmd = None
                tier_w = None
                if args.flow == "skill":
                    tier_w = (int(args.skill_tier)
                              if args.skill_tier is not None
                              else skill_tier_for_amp(amp))
                env._delta_src = make_endurance_flow(
                    args.flow, env, amp, args.env_yaml,
                    wild=not args.tame, ee_speed=args.ee_speed,
                    skill_dir=args.skill_dir, skill_tier=tier_w,
                    geometry=args.geometry)
                tier_txt = f" skill_tier={tier_w}" if tier_w is not None else ""
                print(f"[endurance] === {method} window {w + 1}/{len(amps)} "
                      f"amp={amp}{tier_txt} seed={seed_w} ({ctrl_steps} steps"
                      f" x {env.num_envs} envs) ===", flush=True)
                tracker = None
                if skill_judge is not None:
                    from safeduo.eval.skill_success import EETraceTracker

                    # trace FK must match the judge's waypoint FK (battery7
                    # second-fire RCA: default V5SkillFK traces vs V7SkillFK
                    # waypoints put F_L ~0.19 m off in every family and ran
                    # JAKA kinematics on UR joints; v5 path: judge.fk IS
                    # V5SkillFK, so this stays bit-identical)
                    tracker = EETraceTracker(fk=skill_judge.fk)
                post_reset = None
                if args.flow == "skill" and args.skill_q0_teleport:
                    def post_reset(e):
                        from safeduo.delta._contract_stub import ARM_KEYS

                        src = e._delta_src
                        qref = src._qref(src._t_start)
                        for arm in ARM_KEYS:
                            art = e._arms[arm]
                            q = art.data.joint_pos.clone()
                            q[:, e._joint_idx[arm]] = qref[arm].to(
                                dtype=q.dtype, device=q.device)
                            art.write_joint_state_to_sim(
                                q, torch.zeros_like(q))
                            e._targets[arm][:] = q[:, e._joint_idx[arm]]
                res = run_window(
                    env,
                    driver,
                    ctrl_steps,
                    post_reset=post_reset,
                    ee_tracker=tracker,
                    pair_recorder=pair_recorder,
                    run_id_base=w * env.num_envs,
                    record_traj=bool(args.dump_traj_npz),
                    record_causal_trace=bool(args.causal_trace),
                )
                extra = {"alpha_mean": [float(v) for v in res["alpha_mean"]],
                         "tube_fraction": [float(v) for v in res["tube_fraction"]]}
                fams = source_families(env._delta_src)
                if fams is not None:
                    extra["family"] = fams
                if tracker is not None and fams is not None:
                    from safeduo.eval.skill_success import skill_success_columns

                    extra.update(skill_success_columns(
                        tracker, fams, judge=skill_judge, dt=ctrl_dt))
                    print(f"[endurance] task success "
                          f"{sum(extra['task_success'])}/{env.num_envs} "
                          f"(EE waypoint caliber)", flush=True)
                meta = {"method": method, "flow": args.flow, "amp": amp,
                        "seed": seed_w, "window": w}
                if tier_w is not None:
                    meta["skill_tier"] = tier_w
                w_rows = run_rows_from_margins(res["margins"], ctrl_dt,
                                               meta=meta, extra_per_env=extra,
                                               table_viol=res["table_viol"],
                                               table_exempt=res["table_exempt"])
                for j, r in enumerate(w_rows):
                    r["run_id"] = w * env.num_envs + j
                rows.extend(w_rows)
                if args.dump_traj_npz:
                    npz_meta = {"env_yaml": args.env_yaml,
                                "geometry": args.geometry,
                                "duration_s": args.duration_s,
                                "ckpt": args.ckpt or None,
                                "clutch_enabled": bool(args.clutch),
                                "theta_hi": (args.theta_hi
                                             if args.clutch else None),
                                "theta_lo": (args.theta_lo
                                             if args.clutch else None),
                                "bypass_mm": args.bypass_mm,
                                "bypass_trace_valid":
                                    args.bypass_mm is not None,
                                "initial_engaged": "all_true",
                                "hazard_prob_source":
                                    "1-driver.last_alpha_policy",
                                "causal_trace": bool(args.causal_trace),
                                "date": time.strftime("%Y-%m-%d %H:%M:%S")}
                    paths = dump_traj_npz(
                        res, ctrl_dt, args.dump_traj_npz, meta=npz_meta,
                        rows=w_rows, prefix=method)
                    print(f"[endurance] dumped {len(paths)} traj npz -> "
                          f"{args.dump_traj_npz}", flush=True)
                print(f"[endurance] window done in {res['wall_s']:.0f}s: "
                      f"{sum(1 for r in w_rows if r['success'])}/{len(w_rows)}"
                      f" success (raw caliber "
                      f"{sum(1 for r in w_rows if r['success_raw'])}"
                      f"/{len(w_rows)}, table exempt steps "
                      f"{sum(r['table_exempt_steps'] for r in w_rows)})",
                      flush=True)
        finally:
            if raw_shim is not None:
                env._backstop = raw_shim._inner
        agg = aggregate_success(rows)
        payload = {
            "method": method, "flow": args.flow, "tier": args.tier,
            "amps": amps, "duration_s": args.duration_s,
            "num_envs": env.num_envs, "seed_base": args.seed,
            "env_yaml": args.env_yaml, "geometry": args.geometry,
            "ckpt": args.ckpt or None,
            "baseline_cfg": overrides.get(method), "tag": args.tag,
            "dt": ctrl_dt, "steps_per_run": ctrl_steps,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregate": agg, "runs": rows,
        }
        if args.flow == "skill":
            payload["skill_dir"] = args.skill_dir
            payload["skill_q0_teleport"] = bool(args.skill_q0_teleport)
            payload["skill_tier_arg"] = args.skill_tier
        if args.clutch:
            # key present only when enabled: the default payload schema stays
            # byte-identical to the pre-R17 output
            payload["clutch"] = {"enabled": True, "theta_hi": args.theta_hi,
                                 "theta_lo": args.theta_lo}
        if args.bypass_mm is not None:
            payload["bypass_mm"] = args.bypass_mm
        payloads.append(payload)
        (out_dir / f"endurance_{method}.json").write_text(
            json.dumps(payload, indent=1))
        if pair_recorder is not None:
            pair_path = out_dir / f"negative_pair_records_{method}.jsonl"
            write_pair_records_jsonl(pair_recorder.rows, pair_path)
            print(f"[endurance] wrote {pair_path} "
                  f"({len(pair_recorder.rows)} negative self_U steps)",
                  flush=True)
        lo, hi = agg["success_ci95"]
        print(f"[endurance] {method}: success {agg['n_success']}/"
              f"{agg['n_runs']} rate {agg['success_rate']:.3f} "
              f"CP95 [{lo:.4f}, {hi:.4f}] | gt-verified "
              f"{agg['n_success_gt_verified']}/{agg['n_runs']} | raw caliber "
              f"{agg['n_success_raw']}/{agg['n_runs']} (table exempt steps "
              f"{agg['table_exempt_steps']['total_steps']})", flush=True)

    report = render_endurance_report(payloads)
    (out_dir / "endurance_report.md").write_text(report, encoding="utf-8")
    (out_dir / "endurance_all.json").write_text(json.dumps({
        "payloads": payloads,
        "spec": "TELEOP_SAFETY_DATA_EVAL_SPEC_20260813 section 2B",
    }, indent=1))
    if args.flow == "skill":
        from safeduo.eval.success_report import (
            render_skill_matrix,
            skill_matrix_from_runs,
        )

        for p in payloads:
            matrix = skill_matrix_from_runs(p["runs"])
            stem = out_dir / f"skill_matrix_{p['method']}"
            stem.with_suffix(".json").write_text(json.dumps(matrix, indent=1))
            stem.with_suffix(".md").write_text(render_skill_matrix(matrix),
                                               encoding="utf-8")
            print(f"[endurance] wrote {stem}.md", flush=True)
            # T4 R5 (additive): same matrix over the TASK-completion caliber
            if any("task_success" in r for r in p["runs"]):
                tmatrix = skill_matrix_from_runs(p["runs"],
                                                 success_key="task_success")
                tstem = out_dir / f"skill_task_matrix_{p['method']}"
                tstem.with_suffix(".json").write_text(
                    json.dumps(tmatrix, indent=1))
                tstem.with_suffix(".md").write_text(
                    render_skill_matrix(tmatrix), encoding="utf-8")
                print(f"[endurance] wrote {tstem}.md", flush=True)
    print(f"[endurance] wrote {out_dir}/endurance_report.md", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # skip kit close (known 15-30 min hang on this box)
