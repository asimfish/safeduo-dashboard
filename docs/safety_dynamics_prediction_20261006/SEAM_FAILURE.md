Observed: all six initial launches stopped at step0 before physics; protocol
error AttributeError full_J. Exit code was0 despite protocol.failed, so the
protocol gate correctly counted0 completed/384 invalid planned windows.

Expected: observe the same full9021 J used by frozen admission before physics.

Cause: GuardTrace.safety invokes checked_rows, which dereferences
self.original_rows; the first observer wrapped env._provider.rows_from instead.
The two paths are distinct. GPU memory, data shape and randomized outcomes do not
explain a deterministic missing attribute before the first physical step.

Exact seam regression recreates this attribute-dereferencing closure: original
observer raises full_J; corrected capture returns the identical rows and guard
output with exactly one real row producer call. Tests passed2/2. This regression
is not a full simulator or bit-exact noninterference test.

Correction: isolated observer_runner_v2 intercepts self.original_rows and restores
the actual provider after writing. No production source/controller/model/metric,
input seed/tape, threshold or exemption changes. Re-register unchanged inputs in
new retry directories. Preserve failed plans, outputs, errors and denominators.
Abort further scheduling on any retry failure. No outcome-guided sample selection.
