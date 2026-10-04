# Queue-envelope admission counterfactual

Diagnostic replay is exactly equal to the old development cell in q_initial,
commands, q, EE, execution, official margins and issued/delivered targets. 39/64
windows violate. Several first offending rows appear only after motion begins,
although 6 controller targets are already in flight. In env47 table row8974,
at step67 the solver issues +6.553mm opening while measured geometry moves
-4.158mm; residual is zero. At step72 the geometry crosses from +6.011mm to
-13.817mm while issuing +22.085mm opening. No exemption changed. COM rates agree
with body rates on this row. Env35 self_U row7310 similarly has substantial
negative stored target projection before retreat becomes physically effective.

Candidate: add rows that are forecast dangerous at any of the actual pending
targets, the current issued target, or the boxed/soft-limited upcoming command
proposal. Forecast uses the same physical Jacobian and full distances. Preserve
all original priority and raw-critical rows. The original solver, actor32,
backlog_aware=false, horizons, FIFO6 and actuators are unchanged.

This changes row admission only. Candidate capacity1024 is declared before any
candidate outcome, with abort/no dropping; report observed required widths. The
baseline remains original dynamic512. Additional capacity is an implementation
budget for the new admission policy, not a second top-k tuning factor. It cannot
prove that linear target forecasts cover nonlinear motion or continuous time.

First run: matched old development tape60317411, all64,960steps. Then exactly the
three pre-frozen command-only holdout seeds in holdout_registration.json, baseline
and candidate, one fresh process each. Initial bank reuse is explicit. Every
completed/invalid/failed window stays in evidence, even if the candidate is worse.
No candidate switch or parameter search after holdout. Original margins>=0 gate,
deep<-5mm endpoint, target increment limits, full duration/no resets and strict
FIFO remain unchanged. Production files and checkpoint are not modified.
