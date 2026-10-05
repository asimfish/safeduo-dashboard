# ADR: audit actual next-step motion before changing safety control

Status: Accepted for isolated observation only.

The previous joint repair raised strict failures from25 to30/192. Some saved
linear sets are infeasible, while other feasible returns precede negative measured
geometry. Immediate integral-target displacement is not measured PD displacement.

Candidate A: calibrate a one-step empirical delivered-target/velocity predictor.
It uses q, pre-step qd and the oldest actual FIFO target. It can be tested on old
data and prospective commands; joint coupling/contact and passive joints limit it.

Candidate B: clone complete physics and roll out the FIFO/PD state. This could
represent more dynamics but requires a demonstrated state-cloning boundary,
including contact/warm-start state, and a measured execution budget. Neither exists
as a validated seam in the current evidence.

Decision: A is an observation module, compared with static joints, immediate
delivered-target snapping and constant-velocity Euler. All predictors see the same
physical trajectory. No model output enters admission, projection or actuators.
Do not turn an empirical point prediction into a conservative safety claim.

Input interface: current q/qd, oldest pre-step FIFO target, dt and frozen per-joint
coefficients. Output: predicted controlled-joint increment. The observer adds the
current full-body distance-rate remainder not explained by controlled Jqd, and
compares against next measured full geometry. This remainder is a constant-rate
approximation, not passive-joint dynamics identification.

Training: all12 closed conditions from safety_joint_guard_20261005_1005, explicitly
reused as calibration. Historical external validation: all12 conditions from the
later feasible-guard campaign; their outcomes are already known, so they are not
a new untouched holdout. No splits/hyperparameters change after scoring.

Prospective test:192 fresh command tapes, reused complete192 feasible-guard
initial poses, two unchanged controllers and384 complete16s method windows. Each
tape contains450 mixed-hold and450 IID steps after60 zero steps; order is balanced
32/32 within each bank and fixed before physics. All26 direction components vary,
four arms have independent amplitudes. Same bank/tape matched between controllers.
This is0 new initial poses and192 new commands, not384 independent samples.

All original thresholds, exemptions, actor32 observations, target integration and
strict FIFO6 remain. Keep every violation and underexposed case. No hardware,
production-control change, deadline claim or new carrying-task claim.
