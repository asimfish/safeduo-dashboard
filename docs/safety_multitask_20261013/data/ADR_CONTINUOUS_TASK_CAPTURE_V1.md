# Continuous task capture without changing contact dynamics

Task preparation on 2026-10-13 requires complete close/lift/carry/place/release/
retreat episodes. The current host event ledger rejects event8192, giving only
4.267s at the current1920Hz profile. Restarting the boundary would reset prepared
kernel identities and event generations and does not establish one continuous
physical episode.

Two candidates were considered:

1. A bounded65536-entry host ledger with unchanged streamed event records,
   original CUDA argument copying, kernel call order and16 extra passes.
2. Replace the ledger with a rolling ring and independently reconcile streamed
   counts. This supports arbitrarily long episodes but changes index semantics,
   eviction behavior and error diagnosis, requiring broader validation.

Choose candidate1 for a bounded34.133s development task window. It changes five
capacity literals only. No solver parameter, native constraint, actuator gain,
physical mass, collision filter, gate or tensor state changes are permitted.
This is a new capture-capacity iteration; the previous numerical-contact iteration
remains exhausted. At most two native implementation spikes are allowed in this
iteration, and only one capacity implementation is currently selected.

Gate: all14 existing CPU ABI/failure cases, complete sequential coverage through
event65535 and rejection of65536; then one known native115 replay whose captured
physical states and targets must be byte-identical for all832 recorded events.
The known native115 contact/mimic failure must remain a failure. Passing this gate
qualifies capture compatibility for that replay, not the underlying safety model,
long physical duration, manipulation success or arbitrary teleoperation.

Rollback: preserve the old library and all old registrations. Use the new library
only in new individually frozen registrations. New tasks remain developmental
and separately assessed against the unchanged original physical thresholds.
