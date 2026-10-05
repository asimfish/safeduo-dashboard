# Frozen dynamics prediction protocol

Observed:joint repair25→30/192;4 rescued/9 new. Expected for this experiment:
quantify actual next-step prediction error and missed negative geometry, without
changing any safety decision. A predictor may fail; all results remain visible.

Hypotheses: (H1) queued target and velocity explain motion more accurately than
instant target snapping; (H2) a per-joint fit still misses coupled/contact motion;
(H3) controlled-joint prediction error is distinct from geometric/passive-joint
linearization error; (H4) current/prospective official exemption changes create a
separate prediction boundary. These are discriminating observational checks,
not unique causal counterfactuals.

Fit one nonnegative coefficient pair per joint:
delta_q = a * qd * dt + b * (delivered_target - q).
Minimize unregularized squared error over all calibration transitions; no outlier,
failure or zero-prefix exclusions. No intercept, hyperparameter search, gain cap,
post-fit clipping or refitting. A rank-deficient joint aborts calibration.

Baselines:static controlled joints, delivered-target snap, velocity Euler. Actual
J*delta_q is a labeled hindsight decomposition, never an online predictor.

Primary new metrics:full-stream controlled-joint MAE/RMSE/max; signed full-row
next-distance errors/MAE/RMSE/max and fixed0.1mm histogram upper-edge p95/p99;
negative next nonexempt rows predicted >=0; environment-steps and windows with
misses; exact official physical violations/deep by four channels; pre/post mask
transitions separately. Report full nonexempt and pre-distance<=80mm cohorts.
Also report near-cohort error and miss steps by zero-prefix/mixed/IID phase;
these sequential phases share a state trajectory, not randomized independent
episodes. All metric arithmetic uses archived native float32 predictions and
float64 reductions. Recomputing the elementary predictor on CPU permits a fixed
2e-7rad float32 rounding difference; actual q/qd/FIFO identities require exact
array equality. No tolerance enters physical margin or missed-sign metrics.
Do not declare a model conservative from finite observed errors.

New inputs and source/model SHA are registered before new physics. Fresh tapes
use .005/.015/.025/.05rad per-arm amplitudes and original mixed holds
1/4/15/30/90/180 plus IID; each environment contains both450-step regimes in a
balanced frozen order. All960 steps are retained, including zero prefix.

Only full admission and joint reference from the preceding frozen study execute.
All4 predictions are passive and share exact actual q/geometry. Full9021 J is
checked online; archive float32 full-row predictions/post geometry in32-frame
chunks and preserve SHA. Predictor source and metrics freeze before new outcomes.
Strict FIFO6 and model input oldest target are checked against actual applied
targets offline. Invalid/capacity/nonfinite conditions count as invalid, never safe.

Scope:simulation geometry margins, not mesh contacts/hardware reliability. Old
calibration/external validation and new command tests have separate denominators.
Candidate-control promotion is forbidden for this observation protocol.
