# Astra review response

Frozen v1 experiments are not changed after seeing holdout or development outcomes.
They have a known observation gap: forecast_only_min finite does not prove all
unadmitted forecasts finite. Results are bounded empirical evidence, not safety
approval or production promotion.

A separate v2 development-prefix diagnostic adds explicit all-full-row d/dmin/
closing/forecast and Jacobian finite checks, plus q/qd, issued/queued/proposal
target checks before projection/physics. Invalid data aborts rather than silently
disappearing from admission. On valid numbers its policy is the frozen v1 policy.
It records stable rows8974/7310 and slots14/35/47, with actual ordered queues.
Its 120-step tape prefix is exactly the existing full development tape; no new
full16s windows are counted. Compare all q/actions/targets against v1 prefix.

The current plans inherited old prose fields `candidate`, `registered_utc`, and
prior-campaign continuation metadata. These do not define this experiment. The
new jobs/design and helper SHA, runtime random_manifest, MECHANISM_CONTRACT and
holdout_registration define admission1024, original actor32, original solver and
strict FIFO6. This clarification does not mutate frozen in-flight plans.

Aggregate added-row counts alone cannot identify the offending row's admission
time. Only the new per-row receipt may support that narrower interpretation.
Even an observed earlier admission and fewer failures do not prove unique root
cause, physical inevitability, continuous-time or hardware safety.
