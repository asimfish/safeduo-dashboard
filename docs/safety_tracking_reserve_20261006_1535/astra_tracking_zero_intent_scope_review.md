# Post-observation zero-intent diagnostic: independent scope note

Current reviewed five policy source hashes and registered independent raw scorer
sources remain unchanged. Astra did not run audit_zero_intent.py, inspect partial
candidate arrays, create windows, or duplicate the parent diagnostic.

The diagnostic is explicitly post-observation. Its .050/.010 comparison replays
only the original command-bound helper on the same saved pre-state. It cannot
establish an alternate physical trajectory or a unique physical cause. With the
unchanged six-step FIFO, a newly issued command at a failure step cannot be the
newly delivered target for that same step; transition7 is the first arrival.

Final report/payload review must retain all60 zero-prefix frames and all576
windows, disclose adverse strict/deep window and environment-step outcomes,
class minima and new failures, and distinguish raw-hold bank qualification from
zero-hold behavior under tightened q-centered bounds. Software/source/audit PASS
is never a safety or improvement verdict. User-supplied interim counts in the
JSON are explicitly unverified context and are not inputs to independent scoring.

One reporting clarification (ZERO-SCOPE-1): audit_zero_intent.py currently defines
its displacement as controller_target minus q_initial, then calls the aggregate
maximum_target_drift_rad. This is displacement from initial measured q; calling
it a change from initial controller target additionally requires binding that
initial target and demonstrating its equality to q_initial. An explicit label
can resolve the scope without changing any policy or score.

No final experimental or diagnostic outcome verdict is issued. Closed raw,
camera own-state/pixel review, and the final parent report remain pending.
