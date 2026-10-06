**Verdict: Pass for the bounded offline FIFO oracle and evidence audit.** This is not candidate approval, an empirical model validation, or a safety certificate. No coefficients were fitted or selected. Parent model/validation work remains independent.

**Dimensions:** Functional achievement, correctness within the supplied diagonal model, isolation, API units, and maintainability: Pass, supported by 19 CPU tests, six detected deliberate mutations, and 150 exact bindings across six frozen block0 cases. Independent physical model validation and hardware safety: Unable to determine from this task; the parent's supplied validation summary is reviewed below. Performance: no runtime deadline or performance claim evaluated; this is an offline NumPy oracle.

**Blocking findings:** None for use as this offline reference. Interpreting either old target feasibility or a fitted affine forecast as certified physical safety remains unsupported. The existing mechanisms below are observations about prior code, not changes introduced here.

**Non-blocking observations:** (1) Target endpoints omit transient motion and FIFO timing. (2) Original command-space halfspaces have no horizon-dependent command-to-state sensitivity. (3) Longer-horizon error must distinguish a known future actuator sequence from a runtime hold assumption. Evidence and the minimum candidate-design requirements follow.

**Minimum required candidate work:** Preserve the registered split; validate parent coefficients independently; explicitly choose the future-command assumption and temporal constraint; account for zero command authority over the pending prefix, empirical model error, geometric linearization, row selection, and existing arbitration. This review supplies an oracle and algebra, not new controller thresholds or a production patch.

**Exact timing and independent evidence.** At the pre-state of control step `t`, the actuator FIFO contains `[P0,...,P5]`. The source controller first integrates and soft-limit-clamps the projected increment into the new absolute target `T[t]`. `TargetDelayQueue.push` appends `T[t]` and pops `P0`; repeated physics substeps use that same popped target. Six control intervals are already committed:

| Horizon from pre[t] | Target used during the last interval | Saved post state | Current new target authority |
|---|---|---|---|
| 1 | P0 | q[t] | zero |
| 6 | P5 | q[t+5] | zero |
| 7 | T[t] | q[t+6] | first possible |
| 12 | T[t+5], or T[t] under hold assumption | q[t+11] | model and future-assumption dependent |
| 18 | T[t+11], or T[t] under hold assumption | q[t+17] | model and future-assumption dependent |

Registered control `dt=.016666 s`: target arrival starts at `6*dt=.099996 s`; its first possible recorded post-state influence is at `7*dt=.116662 s`. Horizon7 is an earliest causal opportunity, not a measured nonzero physical derivative. A fitted model with zero direct position gain but nonzero velocity response can first move position at horizon8; a target-independent model never responds. The tests cover both cases.

The source queue class itself was extracted by AST and exercised with a minimal detach/clone array stand-in for delays 0, 1, and 6; no simulator or torch module was imported. Its SHA256 matches the registered block0 source. Static adapter/integrator tracing and actual saved trace bindings complement this queue test:

- [Evaluator queue and adapter](/home/liyufeng/safeduo/src/safeduo/eval/perturbations.py:59): append/pop at lines 74–76, pre/apply at 100–109.
- [Persistent target integration](/home/liyufeng/safeduo/src/safeduo/envs/duo_env.py:1202): projection precedes clamped target accumulation at 1220.
- [Trace target and state capture](/home/liyufeng/safeduo/src/safeduo/eval/research_battery.py:174): q is post-step; controller_target is issued target; actuator_target is the delayed applied target. The historical field named `applied_margin_delta` is based on the integrated controller-target increment, not measured motion or the delayed actuator target.
- [Snapshot capture](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/guard_runner_v2.py:278): snapshot `returned_cmd` is the actual integrated target increment; `actual_project_return` is the projection return before target integration.

**Frozen evidence scope.** Before inspecting numerical motion/forecast residuals, selection was recorded in [astra_fifo_selection.json](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/astra_fifo_selection.json): the first two distinct environment IDs in chronological `(step,env)` order for each of three methods, old block0 only, seed 278060357. Cases are admission_full (158,60), (162,59); joint_reference (116,38), (141,47); joint_repair (116,38), (140,47).

The audit read those first-failure snapshots, their block0 trace windows `t-1..t+7`, and corresponding pre-step forecast slices. NPZ compression requires decoding requested full array members in memory; all numerical reductions are confined to the selected cases/windows. No block1/block2 raw evidence, aggregate old holdout outcome file, parent fitter, fitted coefficient, or camera artifact was opened for this audit. Manifest metadata includes all old filenames/hashes. Following the parent's later explicit authorization, completed predictor/calibration summaries (including block2 summary metrics) and the new exploratory experiment design were read **only for review**, as recorded below. No fitting, coefficient choice, model threshold, or candidate tuning followed those reads. The user-supplied prior 192-pair totals and camera verdicts were not recomputed.

All 150 exact checks passed. They bind pre/post q and qd, previous issued target, pending actuator queue, projection history, clamped target integration, effective target delta, seven applied targets, and each queue transition. All six cases have `actuator_target[t:t+6] == pre_pending[t]` and `actuator_target[t+6] == controller_target[t]`; none of the first six applied target vectors equals that current new target in these cases. Each of 36 consumed input hashes was checked again at audit completion. Raw files match the sealed manifest, whose hash and SEALED.json hash match the new DESIGN.json.

`qd_post[t] = pre_qd_compact[t+1]` is available only through t=958 in the 960-step dense trace. The selected snapshots also contain their own post_qd, which agrees exactly with the next pre_qd. No synthetic final post_qd was created.

**Observed target/motion discrepancy.** The seven newly failing rows in the six selected cases were all already selected, nonexempt before and after, and had negative velocity-aware `d_eff`. For every one, the old target-only minimum forecast equals the positive pre-distance, whereas the measured next distance is negative. Distances below are in **mm**, increments/rates contracted with the saved COM-referenced controlled-joint J:

| Method | step/env | pair | target forecast | measured post | J·effective target increment | J·actual q displacement |
|---|---|---|---:|---:|---:|---:|
| admission_full | 158/60 | 9019 | 0.097619 | -0.627054 | 3.203523 | -0.725731 |
| admission_full | 162/59 | 8941 | 0.832127 | -5.090905 | 8.622188 | -6.004611 |
| joint_reference | 116/38 | 8995 | 3.241830 | -3.977247 | -1.541629 | -7.178433 |
| joint_reference | 141/47 | 8887 | 1.183748 | -4.641831 | -7.732592 | -5.826251 |
| joint_reference | 141/47 | 8891 | 0.417877 | -5.357455 | -7.670986 | -5.776155 |
| joint_repair | 116/38 | 8995 | 3.995351 | -3.317960 | 1.928708 | -7.267599 |
| joint_repair | 140/47 | 8891 | 6.542791 | -0.747275 | -8.847147 | -7.291614 |

The applied-target error `J·(applied_target-pre_q)` is positive for these seven rows (32.392794–46.374500 mm), while `J·actual_q_displacement` is negative. Thus even the actually delivered target's linear endpoint is not a one-step motion prediction. Across the six cases, the maximum per-joint difference between actual q displacement and integrated target increment ranges from 0.009579808 to 0.037662983 rad. The first admission_full case has nonpositive signed target residual on its failing U row (-2.828477e-8 m), yet its actual-motion substitution into that row gives +0.003929226 m. These are distinct variables; the substitution is diagnostic, not an assertion that the old row was intended as a physical-state certificate.

The independent frozen-J endpoint reconstruction agrees exactly on all seven failing-row forecasts; maximum difference across the selected valid rows is 2.3378725e-8 m from float32 versus float64 arithmetic. This number is reported as numerical discrepancy, not a physical tolerance. The one-step geometry residual `d_post-(d_pre+J·dq)` on failing rows is between -4.571218e-5 and +8.157815e-5 m. These observations do not identify a unique PD, inertia, coupling, contact, control, or geometry cause. They do rule out treating target endpoint displacement as equal to the saved one-step q displacement in these examples.

**What the old forecasts and constraints actually do.** [full_finite_guard.py](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/full_finite_guard.py:14) takes the minimum of current distance and frozen-J endpoint distances for all pending targets, issued target, and boxed raw proposal. It checks qd for finiteness but does not propagate it. Taking a minimum over endpoints is order-insensitive; it does not simulate the FIFO, acceleration, overshoot, or intervening trajectory. Its raw proposal also precedes later reference-governor/projection changes. It controls admission, not the measured geometry or actor's original 32-row observation.

The original [backstop](/home/liyufeng/safeduo/src/safeduo/safety/backstop.py:190) separately incorporates closing velocity and pending target backlogs into `d_eff`, then forms `cap=gamma*(d_eff-dmin)*dt` and `G=-J` acting on a new target increment. Thus the positive target-only forecast here does **not** mean every old risk signal missed the threat. Velocity lookahead had already predicted danger. Original positive-cap cross-robot priority allocation, negative-cap retreat, authority clamp, per-arm progress budgets, soft-limit bounds, and bypass semantics remain additional conditions; repairing their command-space feasibility alone does not make the current command act inside the pending prefix.

**Independent affine derivation and API.** For each joint let `c_q=(a,b,c)`, `c_v=(d,e,f)`, using features `(qd, applied_target-q, 1)` supplied by the parent:

```text
q_next = q + a*qd + b*(target-q) + c
v_next = d*qd + e*(target-q) + f
x_next = A*x + B*target + k
A = [[1-b, a], [-e, d]], B = [b,e]^T, k = [c,f]^T
```

For an applied target represented as `o_k + g_k*new_target`, propagate

```text
base_state_next = A*base_state + B*o_k + k
sensitivity_next = A*sensitivity + B*g_k
base_state_0 = [q0,qd0]^T; sensitivity_0 = 0
displacement_base_h = base_state_h.position - q0
q_h-q0 = displacement_base_h + position_sensitivity_h * new_target
```

Before queue exhaustion, `o_k=P_k`, `g_k=0`. Under the default held-target assumption afterwards, `o_k=0`, `g_k=1`. For `r=h-6>0`, the state sensitivity is `sum_{i=0}^{r-1} A^i B`; it is exactly zero at h1 and h6. At h7, position sensitivity is `b`, velocity sensitivity is `e`. For h12/18 the sums have 6/12 terms. Horizon values are requested explicitly; duplicates and arbitrary order are preserved.

```python
from astra_fifo_oracle import DiagonalDynamics, affine_horizons

model = DiagonalDynamics(c_q, c_v, dt_seconds=0.016666)
affine = affine_horizons(model, q, qd, pending, horizons=(1, 6, 12, 18))
displacement, velocity = affine.evaluate(new_target)
# displacement == affine.displacement_base + affine.displacement_sensitivity*new_target
```

All state vectors have shape `(26,)`, pending `(6,26)`, and coefficients `(26,3)`. The oracle accepts any positive joint count for synthetic tests. `c_q` units are `(s, dimensionless, rad)`, `c_v` units `(dimensionless, 1/s, rad/s)`. The fitted coefficients already represent one control interval: **do not multiply them by dt again**. dt metadata must match the coefficient training interval. The optional `DiagonalDynamics.from_centered(...)` converts user-supplied feature means and response means exactly into raw-feature intercepts; it estimates nothing. Standardized feature scales must be undone by the caller. Raw-feature fits use the ordinary constructor.

`rollout(model,q,qd,applied_targets)` is the direct recurrence for offline replay of a supplied applied sequence. Such a sequence at h12/18 contains future decisions and is not runtime information. `affine_horizons` defaults to holding the current new target after the fixed queue; its longer-horizon discrepancy includes subsequent command differences. Explicit future offsets/gains can express a single issue with later fixed targets or known cumulative increments; they are a declared mathematical assumption, not future policy knowledge. Future clipping or policy-dependent decisions generally invalidate one global affine map. Constrain the new command to the existing target bounds so `new_target = previous_issued_target + increment` remains valid.

**Units and candidate halfspaces.** The [DeltaCmd contract](/home/liyufeng/safeduo/src/safeduo/safety/types.py:137) is one-step joint increment (rad); the [ConstraintRows contract](/home/liyufeng/safeduo/src/safeduo/baselines/base.py:9) has distance in m and `J*qd` in m/s, hence J in m/rad. Original gamma has units 1/s and `gamma*(d_eff-dmin)*dt` is m. New affine endpoint rows must change both sensitivity and offset:

```text
new_target = T + u                    T,u: rad
d_h ~= d0 + J*(b_h + S_h*(T+u))      S_h: dimensionless
d_h >= required_distance
G_h = -J*diag(S_h)                    m/rad
h_h = d0-required_distance + J*(b_h+S_h*T)   m
G_h*u <= h_h                         m <= m
```

`displacement_halfspaces` implements this combined full-joint row exactly, with caller-supplied required distances. It sets no threshold, exemption, slack, gamma, or physical epsilon. It neither applies nor changes production constraints. `required_distance=0` corresponds to the existing strict `<0` geometry boundary for retained rows; an existing per-row braking distance is a different caller-selected temporal constraint. The deep threshold remains the original `<-0.005 m` reporting criterion.

At h<=6, `G_h=0`. If `h_h<0`, the model predicts a violation that no new target can repair by that horizon. Do not pass such a row through the old `h=max(h,0.9*authority)` unchanged: with zero authority it would replace a negative h by zero and erase that infeasibility. Likewise, changing d_eff while retaining `G=-J` would still invent immediate command authority. These are candidate-design implications, not changes made to the old solver.

If a candidate instead chooses a velocity-damper condition at horizon h, with `v_h=v_b+V_h*(T+u)`, then

```text
J*v_h >= -gamma*(d_h-dmin)
G = -(J*diag(V_h) + gamma*J*diag(S_h))          m/(rad*s)
h = J*(v_b+V_h*T) + gamma*(d0+J*(b_h+S_h*T)-dmin)  m/s
```

Multiplying the **whole** row by dt converts it to meter units. Inserting dt only into one side, retaining the old cap for an endpoint constraint, or omitting `J*(b_h+S_h*T)` is dimensionally or algebraically wrong. Per-robot priority allocation and alpha budgets require an explicit reconciliation with these coupled rows; the oracle does not silently inherit the old h split. Checking only endpoints h1/6/12/18 does not prove all intervening states safe. The same oracle can expose every integer horizon for a separately specified check, but model/linearization uncertainty still remains.

**Parent's exploratory admission design review.** The later authorized reads were [PREDICTOR_RESULTS.json](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/PREDICTOR_RESULTS.json), [MODEL_CALIBRATION.json](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/MODEL_CALIBRATION.json), and [RANDOM_EXPERIMENT_DESIGN.json](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/RANDOM_EXPERIMENT_DESIGN.json). The latter explicitly states nominal exploratory heuristics, failed conservative-bound requirements, coefficients not causally identified, and no hardware/production approval. **Pass for that claim language and design-level admission-only scope; runner implementation and new outcomes are not verified here.** The affine halfspace derivations above are reference mathematics only: the parent has explicitly declined inversion of fitted coefficients and command optimization.

The completed summary contains 12 prior cells. For old block2 joint_reference after the 60-step zero prefix, the reported one-step RMSE improves from CV 0.0011153970 to empirical PD 0.0008588680 rad. This average improvement coexists with large tails:

| Block2 reference summary, after zero prefix | Maximum absolute error (rad) |
|---|---:|
| h6, known FIFO / actual future sequence (identical prefix) | 0.2145348093 |
| h12, actual future applied ORACLE | 0.4629222501 |
| h12, known FIFO then held decision | 0.4629979297 |
| h18, actual future applied ORACLE | 0.5107994922 |
| h18, known FIFO then held decision | 0.5119438927 |

Including the zero prefix increases the reported h6 maximum to 0.3082396368 rad and h18 maximum to 0.8534172674 (actual future) / 0.8434474705 (held decision). The old block1 calibration global observed maxima are h6 0.5459972045 rad, h18 1.1448090098 (actual future) / 1.1494223635 (held decision). These observed maxima are not distribution-free bounds. The reported 0.5108 rad h18 figure belongs specifically to actual-future replay after the zero prefix; it must not be labeled a runtime held-target bound. The parent's reported negative fitted input coefficients further preclude interpreting every coefficient as a positive physical PD response; their signs were not used to tune or invert this oracle, and MODEL_FIT.json was not inspected.

The new registered four-method design is joint_reference, velocity_admission, pd_admission, and motion_admission, with combined motion versus reference primary and CV/PD ablations. At the design level it retains the old target-based admission rows and unions additional rows from nominal CV/PD forecasts, uses the original actor32, finite-pass backstop, reference gap .050, FIFO6, and original strict/deep physical metrics, with capacity9021 and no dropped rows. It declares the actual pending six targets followed by a held current proposal, not future policy knowledge. Its model/results hash bindings agree with the completed summaries.

That is a coherent **exploratory row-admission experiment**, provided implementation fulfills the registered retention, finite/overflow abort, no-dropping, common-input, and scoring requirements. A larger admitted set does not itself establish monotone physical safety: finite-pass projection, constraints in conflict, and subsequent state/command changes can alter outcomes. Row retention is a software invariant to verify; conservativeness is a physical claim not established here. Actual original nonexempt full geometry, paired rescues/new failures, and achieved motion must determine the fresh experiment's outcome. No new runner was implemented or executed by this task.

**Verification and reproducibility.** Use `PYTHONDONTWRITEBYTECODE=1` on every command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/astra_fifo_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 /home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/astra_fifo_tests.py --mutations
PYTHONDONTWRITEBYTECODE=1 python3 /home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/astra_fifo_audit.py
```

The 19 tests cover the literal source FIFO, zero/one/six delays, pending prefix causality, exact scalar closed forms, position/velocity response distinctions, all 26 joints, arbitrary horizons, future-command assumptions, constants/means, nonzero target origin, row sign, units, nonfinite inputs and overflow. A real synthetic red case demonstrated that NumPy einsum can return infinity despite `errstate(over='raise')`; the oracle now explicitly rejects nonfinite recurrence results and the added regression passes. All six deliberately injected errors were detected in memory: one-step early arrival, wrong q feedback sign, omitted affine bias, omitted target-origin offset, reversed constraint sign, and LIFO delivery. The float64 test tolerances (up to 3e-14) are arithmetic comparison tolerances only. No production test suite or simulator test was run because production code was not modified and this task is an isolated NumPy reference.

**Changed files, all in this H:** `astra_fifo_oracle.py`, `astra_fifo_tests.py`, `astra_fifo_audit.py`, `astra_fifo_selection.json`, `astra_fifo_evidence.json`, `astra_fifo_mutation_results.json`, `astra_fifo_tests.log`, `ASTRA_FIFO_REVIEW.md`, `ASTRA_FIFO_REVIEW.json`. The machine-readable review binds the code and evidence digests. No parent/design, old namespace, RAW, production, actor, threshold, exemption, or queue was written; no GPU, hardware, process termination, external message, or publication action was taken. Empirical fitting or retrospective agreement cannot certify future or hardware safety.
