Independent CPU readback scheduling review — PASS within source/test scope

Reviewed wrapper SHA256: `031ec518350b71104dbbeee5ef315f6f15cd496daa6d932a6b77ae2b0435f3fc`. The registered source preflight passed for all three closed canonical campaigns (12 distinct condition paths). No parent script, original calculation source, raw data or policy was edited.

Exact `analyze.inspect` functions run on four fresh futures. `analyze.main` retains its original serial plan/job order and arithmetic; the route checks path/job/plan/dense. H6 calls the original inspect function and preserves registered row order through `executor.map`. Per-cell numerical state is local. Original full input hash checks remain. Exceptions propagate and do not produce a success execution receipt. Driver task names retain historical names, but actual argv and wrapper SHA disclose the replacement execution truthfully. `cache_used=false` means no persisted outcome cache; futures do hold fresh results in memory.

Eight synthetic tests passed, including wrong arguments, worker errors, post-execution source mutation, source digest rejection, ordered H6 results, and the memory threshold. These are scheduling tests, not raw-result equivalence or Isaac/GPU tests. The first synthetic test fixture omitted an env field; its failed log is retained and only the fixture was corrected.

Launch handoff is reconciled by the appended current receipt review: the original three readers and first-attempt three PIDs are all absent. The wrapper does not itself retire old writers, and original main can overwrite its result JSON; an interruption or restart must preserve any prior output and its execution status.

The measured frozen chunk load was 5.385s for 454,839,473 compressed / 745,324,544 expanded bytes. Time is in NumPy array reads/copies, decompression and CRC; no parallel speedup has yet been measured. Four analysis plus four H6 futures are bounded concurrency. The 25GiB available-memory check is a start threshold, not an atomic reservation or peak-memory guarantee. Available memory at review was 72612876KiB with 48 allowed CPUs. Retained completed analysis cells and concurrent current/next chunks add memory beyond one chunk per worker.

Run with PYTHONDONTWRITEBYTECODE=1 and assertions enabled. Require actual child closure, output/source hashes and execution receipts before claiming full readback completion; partial files following interruption are not completed evidence. No original sequential completion or physical safety is claimed.

Astra retains its existing frozen independent scorer (4/12 conditions processed at the inspected checkpoint) to avoid a competing canonical writer and a further simultaneous raw scan. This choice changes no score or input scope.

Supersession reconciliation (current): the original three readers and the first-attempt driver/analyze/H6 PIDs are all independently absent. Bound ANALYSIS_CPU_ORIGINAL_EXIT and ANALYSIS_PARALLEL_ATTEMPT1_CLOSURE preserve the relative-argv stop-check failure, premature first start, its termination and unfinished logs. Parent reports no canonical outputs from that attempt. Current execution uses the same registered wrapper SHA. No Astra process signal, parent-file edit, policy change or GPU action occurred. This PASS is for source/scheduling/receipt scope; current parallel results remain unverified until completed and bound.
