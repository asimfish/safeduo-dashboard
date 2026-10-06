**Verdict: Pass — scoped independent CPU/source review before new policy outcomes.** No blocking implementation finding in the reviewed sources. This verdict concerns the registered exploratory admission implementation; it is not a physical safety certificate or a claim about future experiment results. No GPU, simulator, bank, policy outcome, or production mutation was used in this review.

**Dimensions:**

| Dimension | Status | Evidence and scope |
|---|---|---|
| Functional achievement | Pass | Four registered modes, exact pending-six recurrence followed by held governed proposal, old target rows retained, 9021-row selection, correct diagnostics. |
| Correctness/reliability | Pass within reviewed CPU seams | 14 independent tests; real forecast functions and actual runner safety/chunk methods exercised; no numerical comparison to new outcomes. |
| Architecture/isolation | Pass | Intervention enters the admission forecast. Original production backstop and actor code are unchanged; research wrappers and the same reference governor remain explicit. |
| API/units | Pass | Supplied one-control-interval coefficients used without an extra dt; q displacements in rad, distance forecasts in m; horizon axis identified. |
| Style/maintainability | Pass with reporting cautions | Copied helpers are byte-identical. Legacy diagnostic terminology needs the narrow interpretation described below. |
| Performance | Unable to determine for GPU campaign | CPU suite passed; no throughput, memory-pressure, or simulation deadline benchmark was run. This is not a real-time performance approval. |

**Blocking findings:** None found in scope. No parent runner file was changed. Full simulator lifecycle, registered launch integrity, GPU execution, and actual post-step outcomes remain outside this review; this is an implementation review, not completion of those gates.

**Non-blocking improvements / scientific interpretation:**

1. `baseline_ids` and `forecast_added_count` retain their historical **measured-distance union** baseline. At [guard_runner.py:101](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/guard_runner.py:101), `baseline=union_mask(full,base,full.dists)`; the count at line 121 includes both old target-based additions and enabled motion additions beyond that baseline. It must not be presented as “new motion rows versus joint_reference.” At a common state, reconstruct the old target-based mask as `baseline_mask | (target_forecast <= dmin+.010)` using the saved target_forecast. Then `selected_mask & ~old_target_mask` is the motion-only addition. The old target rows are retained in control regardless of this diagnostic naming. Severity: Minor; pre-existing field semantics, newly exposed reporting risk.
2. The old shadow comment at [guard_runner.py:103](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/guard_runner.py:103) and metadata at line 221 are operationally harmless: `shadow_d`/`missed` are computed after `mask`, and cannot change it or the returned command. Scientifically, “the reference-governed proposal does not affect control” would be too broad. The separately computed **same proposal value** is used by empirical PD forecasts at h>=7 and can add rows in pd_admission/motion_admission. Interpret/reword this as “the duplicate shadow endpoint statistic is passive.” The old raw target forecast still uses its original boxed raw proposal. Severity: Minor; inherited wording becomes ambiguous in the new PD modes.
3. The receipt's single `shape=[T,64,9021]` at [guard_runner.py:261](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/guard_runner.py:261) describes distance fields, not every NPZ member. Keep explicit per-field schema/axis handling in readers: the new q arrays are `(T,4,64,26)`. The actual writer is correct and a round-trip test passed. Severity: Minor; consumer compatibility/documentation caution, not a writer defect.

**Minimum required repair:** None for the reviewed runtime behavior. Preserve those three interpretations in the eventual analysis/report and make sure the launch binds the reviewed source/model bytes. If sources change, this source-bound verdict does not automatically cover the new version.

**Review boundary and immutable copies.** Compared the new runner directly with the sealed prior `guard_runner_v2.py`, not a guessed Git base. The material differences are the motion forecast call, four new mode definitions, scale1/reference governor in every mode, removal of joint repair, common capacity9021, extra stored forecast components, and truthful nominal-model metadata. These match RANDOM_EXPERIMENT_DESIGN.json. Exact byte-copy checks passed:

| New file | Prior file | SHA256 |
|---|---|---|
| target_forecast.py | full_finite_guard.py | e4644512a380e22d06984401d7ad5081219c3b60b4172f1ea01eb5d3902373d6 |
| reference_envelope.py | reference_envelope.py | 7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47 |
| projection_diagnostics.py | projection_diagnostics.py | 1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869 |

The reviewed model file hashes to `924cd6a75e1a4db437b44b7c202a8a7a70bcc9bcae6696e6964ffc4ffd58bfb2`, matching the registered model binding. The loader returned finite `(26,3)` c_q/c_v and dt=.016666 on CPU. It preserves the fitted signed gains (two negative c_q input gains, three negative c_v input gains); none are inverted, clipped into a physical interpretation, or optimized here. No coefficients were fitted, tuned, or selected by this review.

**Simultaneous recurrence and exact causal prefix.** At [motion_forecast.py:53](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/motion_forecast.py:53), each iteration forms `debt=target-current_q`, then computes both next_q and next_v from the same old current_q/current_v/debt. Assignment to the new state happens only after both expressions and finite checks. This matches the independent NumPy affine oracle for three synthetic 26-joint states, including negative gains, bias terms, and horizons1/6/7/12/18.

```text
next_q = old_q + c_q0*old_v + c_q1*(applied-old_q) + c_q2
next_v =         c_v0*old_v + c_v1*(applied-old_q) + c_v2
```

There is no additional dt in the empirical recurrence. CV uses `qd*(dt*h)`, as required. The loop applies pending[0] through pending[5] at horizons1–6 and proposal at horizon7 onward. Perturbing the proposal leaves the first six outputs exactly unchanged; the seventh-step displacement difference is c_q[:,1] times the proposal difference. The source uses only currently pending targets and the current raw command; it does not read a future command tape for forecasting.

**Held proposal semantics.** [governed_proposal](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/motion_forecast.py:29) uses the copied reference_bounds with gap .050, speed box, and original joint limits. It clips the current raw increment into that reachable interval, integrates it into the existing issued target, and soft-limit-clamps the result. The independent test covers both a reachable envelope and an unreachable envelope that must slew at the existing speed bound. The forecast holds this **reference-governed raw proposal**, not the eventual result of the safety projection. That distinction is correctly stated by `future_assumption` in guard metadata. Disagreement after the queue drains can therefore include subsequent projection and future command effects; it is not pure identified-dynamics error.

**Mode and row-set behavior.** Let T be the original target forecast, V the minimum of current distance and the four CV forecast distances, and P the analogous empirical PD minimum. The implementation at [motion_forecast.py:93](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/motion_forecast.py:93) returns exactly:

| Mode | Admission forecast |
|---|---|
| joint_reference | T, original object returned unchanged |
| velocity_admission | min(T,V) |
| pd_admission | min(T,P) |
| motion_admission | min(T,V,P) |

For the same input state, every candidate forecast is <=T elementwise. The unchanged `union_mask` ORs original selected rows, measured-distance critical rows, and forecast-critical rows. Thus every old target-based row remains admitted. Tests check these set inclusions and separate handcrafted rows admitted only by T, only by CV, and only by PD. This is a common-state set-inclusion result, not a guarantee that evolving trajectories or finite-pass projections become monotonically safer.

Both CV and PD forecasts/diagnostics are evaluated and finite-checked in **all** modes, including joint_reference. Mode flags govern which forecast enters the admission minimum. Consequently, the four modes share a diagnostic fail-closed gate; “reference unchanged” refers to the target-based admission/control on successful finite calls, not removal of all new diagnostic computation or identical wall-clock cost. A predictor abort invalidates/interrupts that window; it must not be scored as safe.

**9021 capacity and measured geometry.** [guard_runner.py:82](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/guard_runner.py:82) requires the registered `(64,9021)` full geometry. It constructs a temporary all-row view for J and forecast checking. The original full geometry and original 32-row active-pair tensor remain unchanged. The unchanged selector computes width from the largest actual mask count, aborts if width exceeds capacity, and emits every selected ID; it never truncates a larger union to fit. A CPU test admitted all 9021 rows, recovered all IDs exactly, retained exemptions/distances, and verified explicit failure at capacity9020. A second test executed the actual runner safety closure on 64 synthetic environments with all9021 rows admitted, confirming this behavior through the real call seam.

The finite gate reads **all** geometry/J rows before selection. Injecting NaN into row9020's J, distance, closing value, or dmin, outside the original 32 active rows, caused the actual closure to raise and write its in-memory `before projection/physics` abort receipt. Pending targets, q/qd, limits, model coefficients, every empirical intermediate state/debt, each CV/PD linear distance and combined forecast are checked on this path. Scalar/broadcast shape handling outside the registered internal shapes was not treated as a general public API guarantee.

**Actor and original projection.** The production actor/observation path was traced at [duo_env.py:1368](/home/liyufeng/safeduo/src/safeduo/envs/duo_env.py:1368): observations use `compute_dist()`'s original active-pair features, not the new safety-only union returned by the patched safety_dist_out. The runner does not replace actor weights or this observation function. New states may naturally produce different actor outputs later; this does not change the actor implementation or its original row budget.

The Python `env._backstop.project` attribute **is wrapped**, as in the prior reference experiment. Its underlying production solver implementation remains the original callable: outer finite gate → copied reference governor → copied passive diagnostic observer → inner finite gate → original projection. Joint-repair installation is removed. The copied diagnostic checked_project returns the exact original tuple/objects; an independent CPU test also compared passive installed-observer results with the original real VelocityDamperBackstop on identical synthetic inputs. The .050 reference governor is the same copied code used by prior joint_reference. No learned sensitivity is passed into J, G, h, alpha, priority, command optimization, or FIFO manipulation.

The actual safety-closure test also verified that issued targets, every pending target, full measured geometry, and the original 32-row features remained unchanged by admission. The pre-step wrapper calls its captured original pre-step implementation and then checks integrated/applied targets; it does not push, pop, clear, or rebase the queue itself. This complements the already completed independent source FIFO audit.

**Diagnostic axis and time contract.** The horizon order is `(1,6,12,18)`. `pd_q_horizons` and `cv_q_horizons` contain absolute predicted q, not displacement. A single call produces `(4,64,26)`, correctly broadcasting current q across the leading horizon axis. `flush_forecast` stacks calls along a new time axis. The actual writer was executed with an in-memory NPZ round-trip containing distinct time/horizon/env/joint values; axes were preserved exactly.

| Field | Per call | In chunk | Interpretation |
|---|---|---|---|
| forecast, target_forecast, cv_forecast, pd_forecast | (64,9021) | (T,64,9021) | Mode-specific or component minimum distance, m |
| cv_h6, pd_h6 | (64,9021) | (T,64,9021) | Distance at h6 alone, not the minimum over horizons |
| cv_q_horizons, pd_q_horizons | (4,64,26) | (T,4,64,26) | Absolute q at horizons1/6/12/18, rad |
| selected_ids, baseline_ids | (64,9021), padded -1 | (T,64,9021) | Actual selection / measured-union diagnostic baseline |

For chunk start s, step t, environment e, joint j, read `pd_q_horizons[t-s, hi, e, j]`. `hi=1` selects h6. Forecast at pre[t] corresponds to saved post indices t, t+5, t+11, t+17 for h1/6/12/18. The independent distance test confirms pd_h6 and cv_h6 use q_horizons[1], not axis index6. End-of-trajectory comparisons must omit unavailable future samples rather than reindexing or imputing them.

The later supplied [audit_prediction_h6.py](/home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/audit_prediction_h6.py:15) was also read for its indexing contract only, without executing it or reading outcomes. It compares distance to pre[t+6] using the next chunk's first six frames, takes q_horizons[:,1], and compares it to saved post q[t+5]. Its 954 origins t=0..953 are correct for available pre-geometry frames0..959; t=954's post q959 exists but its matching pre960 full geometry does not. It uses the same run directory's trajectory and endpoint exemptions. This is consistent with an own-trajectory nominal forecast-error audit; it is not an empirical audit result yet.

**Scientific truth and limitations.** `pd_scope='empirical nominal, no causal or conservative guarantee'`, the CV scope, future assumption, and registered candidate_claim are accurate. A nominal PD forecast can add rows using signed correlated coefficients without constituting a causal PD model. Neither observed calibration maxima nor additional row admission makes a conservative safety certificate. The original nonexempt full geometry `<0` and deep `<-.005 m` remain the physical scoring definitions; the guard first-failure expression still uses actual post geometry and existing exemptions. No outcome counts were inspected or recomputed in this review.

**Actual verification and changed files.** The independent command completed with **14 tests passed, zero failures/errors**:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 /home/liyufeng/safeduo/artifacts/safety_fifo_predictor_20261006_0630/astra_motion_runner_tests.py
```

The test harness imports pure CPU safety/forecast code, compiles the literal admission helpers and GuardTrace class from their AST, supplies a minimal CPU scene/base-trace stand-in, and uses an in-memory filesystem for runner output. It does not import or start the simulator/battery entry point. The original projection test uses the real CPU production backstop. Synthetic coefficients are deliberate, independently chosen test inputs; no block2 or new policy outcome data is used. The tests do not exercise real GPU numerical behavior, actor checkpoint execution, full RiskTrace initialization, or physical dynamics. No new performance claim is made.

Written only: `astra_motion_runner_tests.py`, `astra_motion_runner_tests.log`, `ASTRA_MOTION_RUNNER_REVIEW.md`, `ASTRA_MOTION_RUNNER_REVIEW.json`. The JSON records source/model/design and test artifact hashes. Parent code, prior FIFO deliverables, production, old namespaces, and RAW were not modified; running GPU/bank work was not queried, launched, stopped, or otherwise touched.
