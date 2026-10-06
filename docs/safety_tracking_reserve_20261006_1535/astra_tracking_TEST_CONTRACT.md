Independent tracking-reserve CPU test contract

State: preregistered from the user specification before candidate source/interface inspection. At intake only INTAKE.json existed. Candidate binding remains RED / unavailable until registration and all three candidate files are present. Missing source is not an implementation failure and will never be reported as a passing candidate.

Behavior and impact: the candidate changes only the reference-envelope gap used for newly issued targets. It must not reinterpret the real six-slot actuator FIFO, revise geometry exemptions/thresholds, change old target admission, alter the actor, finite projection solver or original rate box, or turn malformed input into a permissive gap. The nominal reserve is not a physical feasibility or conservative safety certificate.

Inputs and units: q and queued absolute joint targets are rad; qd is rad/s; dt is s; distance and d_min are m; signed distance Jacobian entries are m/rad. ARM_KEYS order is F_L,F_R,U_L,U_R with7,7,6,6 joints. Every row's signed displacement sums all participating arms before any minimum over rows, queue slots or forecast types. Positive J*dq means increasing distance. Physical strict/deep scoring thresholds are outside this heuristic and unchanged.

Risk contract: for every original nonexempt geometry row, include its current d-d_min, its six separate frozen-J queued-target endpoint estimates (absolute target minus current q), and its frozen-J qd*(18*dt) estimate. Risk is their minimum. Do not integrate six absolute targets as six increments, use only the last queue target, replace the queue with the newest proposal, or use abs(J*dq). Eligibility/exemption comes from the actual original full-row mask, not an inferred structural/class exemption. No fabricated missing-row or missing-queue fallback.

Gap contract: baseline0.050rad; fixed-tight0.010rad; adaptive0.010+0.040*clip((risk_m-0.010m)/0.040m,0,1)rad. Risk-meter thresholds and output-radian range are different quantities. Exact gap saturation boundaries and continuity are tested; numerical comparison tolerances, if required for float32 arithmetic, are derived from operation rounding and never change physical/risk classification thresholds.

Reference-envelope contract: intersect the original delta-target rate box with the measured-q-centered reference interval. If unreachable in one step, retain the original nearest admissible rate-box singleton in the recovery direction; report the gap remains unreachable, do not snap the target or enlarge the rate box. Baseline0.050 must preserve the prior scalar semantics. All result intervals must be finite, ordered, and within the unchanged rate bound.

Independent oracle: scalar Python rational/Decimal arithmetic enumerates each row and each endpoint. The reference interval oracle enumerates breakpoints and minimizes interval distance rather than copying the parent's vectorized max/min/where implementation. Candidate adapters will be small and will bind the actual interface only after source and registration exist; no invented required function signature is imposed now.

Required examples/properties:

1. q equals all targets, yet closing qd alone reduces the reserve; reversing qd does not conceal the current margin.
2. Change the current proposed/new action arbitrarily while preserving pre-q/qd and the first six targets: reserve is identical. A six-slot queue state machine delivers the new target on transition7, not1 or6.
3. A hazardous intermediate FIFO target followed by a benign last target remains detected. Six absolute targets are never summed into a trajectory.
4. Left/right and F/U contributions sum in the same row, including cancellation, reinforcement and sign reversal. Minimum must consider every row, including the last of9021; row permutation/duplication cannot change risk.
5. A severe exempt row is excluded exactly; toggling its original exemption reintroduces it. Keep d_min per row and distinguish d from d-d_min. All-exempt and zero-row behavior require explicit interface/registration semantics; never silently invent a physical margin.
6. Risk<=0.010 maps to0.010gap; >=0.050 maps to0.050gap; interior midpoint and adjacent float32 boundary values; monotonicity, batch isolation and unit conversion consistency.
7. NaN/Inf, finite overflow, malformed/broadcast-mismatched shapes, nonpositive/nonfinite dt, missing/short/long FIFO and missing required arm/J components abort visibly. Validate before masking/saturation can hide corruption. Baseline bypass requirements will be checked against the registered interface and full producer invariants.
8. Reachable/intersecting/tangent/unreachable reference intervals, both directions, extreme target debt, asymmetric measured state and per-environment gap. Original rate/soft bounds remain; tight gap cannot create an instantaneous target snap. No queue/input mutation.
9. Actual guard wiring: reserve uses pre-step real state/queue before issuing or appending the candidate action; all modes share original target admission, finite-row checks, actor and original solver. Source extraction/CPU fakes may exercise a narrow pure boundary; no claim of Isaac or numeric-forward verification.

Red-state evidence: retain the missing-candidate result; before candidate PASS, run plausible deliberately wrong local oracle variants (ignore velocity, latest-target-only, single-arm-only, abs displacement, current-proposal contamination, d_min/exemption omission, snap-to-gap, NaN-to-clear). Each must fail a concrete counterexample. These injected variants are owned test code, never parent/production edits.

Failure/recovery: exceptions or explicit invalid output must fail the test and prevent a PASS review. A valid tight gap is not evidence of physical safety. Preserve original failing candidate/source SHA and counterexamples; do not overwrite them with a later corrected PASS.

Nondeterminism: deterministic seeded synthetic CPU fixtures only; no outcome tuning, no old raw campaign reread, no simulation/GPU/browser, no background watcher. Test and review writes are limited to this namespace's astra_* files and the explicitly requested ASTRA_TRACKING_REVIEW.json. Old seals and all parent/production files are read-only.

Unresolved interface details to resolve from registration/source: exact return fields, per-environment versus per-robot reserve scope, explicit all-exempt sentinel/error policy, rate/soft-limit argument shapes, required producer finite-check placement. Any conflict with the user's mathematical/causal contract is a finding, not an oracle adjustment to match observed output.
