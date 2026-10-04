# 2026-10-05 mechanism experiment

The production implementation, actor and strict six-step FIFO stay frozen. This
work is an isolated experiment, not a promoted protection architecture.

Observed: stable-risk dynamic512 System0 has 122/192 violation windows; table99,
self_U50 (overlapping classes). The 60317411 seed is development evidence only.
Expected contract: all 960 transitions finish without reset; official nonexempt
sphere margins remain >=0; targets stay in original soft limits and the 0.025rad
increment box; actual actuator delivery equals controller_target[t-6] exactly.
Full original failures and continuous-time limitations remain visible.

Competing hypotheses before the diagnostic replay:

1. Predictively dangerous rows are missing before failure. Falsify if the row that
   first violates is admitted throughout the six-step delivery window and earlier
   predicted closing interval. Record full distances, body rates and row IDs.
2. The admitted projection cannot enforce its inequalities. Record unmodified cap,
   post-clamp residual and active row Jacobians, including authority-clamp debit.
3. Integrated targets and pending FIFO motion overwhelm a valid per-step increment.
   Record measured q/qd, controller and actuator targets, all pending targets, and
   projected vs observed same-row changes. Preserve the six-step FIFO in comparisons.
4. A conditional table exemption is withdrawn during fast motion. Record row identity,
   effective d_min and exemption at both endpoints, including the previous safe rows.
5. The legacy link Jacobian predicts different motion than COM/body readout. Record
   same-row Jqd and body distance rates; evaluate only after evidence distinguishes it.

The first diagnostic is an exact 64-env,16s,60317411 System0 replay with passive
causal instrumentation. Detailed slots are fixed before replay: 0(control),
14(self_F/table),20(cross),29(cross),35(self_U),39(self_U),47(table),54(table).
All 64 environments continue with original physics. Detailed trace is additional
evidence and never changes the window denominator. Diagnostic observer must reproduce
the prior commands, initial q, FIFO and violation outcome; report actual differences.

Only then register a single justified isolated mechanism, a factor-controlled A/B
and newly generated holdout tapes. No tuning on holdout outcomes; no policy-based
initial-state selection, no safer sampling after results, no IID confidence claim.
Images remain real simulation renders; pressure and object-task gates stay separate.

Storage is NAS from launch, bounded CPU traces, preserving failed startup/protocols.
External GPU services are untouched. Router chose systematic-debugging + simulation;
the destructive-command 'guard' homonym was excluded (catalog digest
6c0149b5ee5d2dd241e2ea13623a1ef1c285d43f5a029ab9f796740ce6d1503a).
