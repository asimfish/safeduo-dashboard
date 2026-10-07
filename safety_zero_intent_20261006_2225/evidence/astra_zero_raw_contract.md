# Independent zero-inclusive raw scorer contract

This scorer uses only new-cycle raw arrays, frozen producer identity and registration,
and its own mathematical code. Parent scoring functions and outcome summaries are
not inputs. Parent scores may be compared only after independent results close.

The inventory is three blocks, three conditions, 64 environments per condition:
192 paired cases, 576 windows, 960 post frames including every initial zero-prefix
frame and frame959. Conditions are joint_reference(.050), tight_reference(.010)
and zero_inclusive(.010); the primary comparison is joint_reference versus
zero_inclusive. The first60 frames also receive separate metrics without removing
them from the primary windows. Strict means native float32 <0, deep means native
float32 <-float32(.005), with no epsilon. Class minima, class failure counts and
correlated environment-step durations are preserved. Unavailable/invalid windows
remain in the denominator and are never called safe.

Pairing binds actual native q0, qd0, initial issued target, executed960 command tape,
stored962 tape and bank bytes. Full six-slot pending target arrays, projection
history, actual target integration, actual applied sequence and target debt are
checked. The newly issued target first reaches the actuator on transition7.
First-six native q/qd/applied fingerprints are compared without asserting identity
of unarchived full rigid state. No same-step causal attribution is made.

All-frame original rate/soft-limit bounds and mode reference bounds are independently
recomputed from saved native pre-state and issued target. Candidate input is invalid
if zero is excluded by original bounds. Actual prelimit input, external command and
projector output are bound to their saved diagnostic arrays. Zero inclusion and
zero-command prelimit preservation do not require zero projector output; nonzero
corrections, effective target changes and positive native bound residuals are reported.
The separate producer projection tolerance is not used to change score thresholds.

Thirty chunks per condition reconstruct exemption-aware class minima from all9021
pre-step rows and bind959 preceding post frames. The final post frame is scored from
native class minima; full final post rows are not claimed. Fixed gap archives, all-row
reserve minima and original target admission unions are checked. Reserve forecasts
are diagnostic only. Full Jacobians are not archived every frame, so no independent
full-J or all-LP-every-frame claim is made. Selected first-failure snapshots bind the
first two distinct failing environments by (step,env) per condition.

The final freeze occurs after notified final registration and before any policy
launch. Code/source/plan/receipt hashes are bound. Thereafter the scoring algorithm
must not change to reconcile outcomes. No watcher is started. Following a closure
notification, the single-use CPU supervisor runs two condition threads only after
all three canonical campaigns are terminal. Each counted child must have a matching
actual argv, complete protocol and actual zero exit. The25GiB available-memory gate
is a submission condition, not a reservation. It fails without indefinite waiting.

Each consumed payload is hashed as read, rechecked per condition, and rehashed across
the final ledger. Checkpoints persist but are never silently reused. Failed attempts
are retained. Threads join and the supervisor records the fresh actual child exit;
there are no automatic retries. All writes stay within astra_zero_* or ASTRA_ZERO_*.
PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES='' apply to own CPU children.

Camera own-state group audit, all PNG headers/hashes, actual viewing of the predeclared
72 original images across nine views, raw/parent reconciliation and final report/code
review remain separate later gates. No claim is made that all486 images were viewed,
or that camera states exactly replay numerical states when identity checks fail.
No physics-safety, robust recoverability, hardware or production approval follows.
