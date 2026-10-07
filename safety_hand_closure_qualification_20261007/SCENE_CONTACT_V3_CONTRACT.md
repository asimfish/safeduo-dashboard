# Frozen v3 scene-contact and empty-hand recovery contract

V2 epoch regression exposed NIST fixture and U table contact in 6/72 recorded U paths (including residual force on one later latched path). The original V2 hand-internal metric did not detect these external forces. This is development evidence for v3, not heldout evidence for v3.

Changes: check unintended partners as well as hand-internal scalar normal magnitudes at the existing 0.1 N criterion; retain speed ceiling 3 rad/s and exact-goal whitelist. Reject the five observed unsafe arm/reference/goal contexts: ref20 U_L/48, U_R/48, U_L/13, U_L/100, U_R/100. Profile34/U_L was not executed after a latch and is not labelled an independently unsafe target. No target, gain, collision, object pose, asset or threshold is removed or retuned.

Latched abort commands a one-second linear return from the last issued desired target to the diagnostic neutral. This is restricted to this empty-hand experiment with payloads parked away. It is not a general emergency stop or a loaded-object release policy. Any threshold crossing rejects that whole path even if recovery later succeeds. Recovery requires six consecutive native frames with <=0.02 rad neutral error, <=0.1 N all measured partner forces and <=3 rad/s native speed; latch is never cleared.

Regression: same 12 target specifications and references2/8/20, 72 U requests; five context refusals are expected and count as refusals, not successful executions.

Fresh validation: new OS-random 63-bit seed and one reference frame in each [1,480), [480,960), [960,1453) of the fixed trajectory, excluding all earlier reference frames. Same 12 frozen goals; no resampling failed frames. Admission is provisional in these new contexts; post-contact monitoring cannot establish collision prevention. This is a trajectory-derived conditional experiment, not an IID random task or 26-DOF workspace-volume claim.

Counterfactual: three original-unsafe-goal bypass/guard pairs at2/8/20; additionally one fixture profile13 pair at20 with a deliberate whitelist override in both arms, comparing latched hold to empty-hand return, using the same all-partner contact detector. All four pairs have identical initial q/qd/targets, identical local object positions within1e-6m, and identical velocities. The hold baseline uses the v3 detector to isolate the recovery action; it is not the unchanged V2 algorithm.

First post-initialization physics frame is retained and scored. No state writes in the physics loop. All failures, denied requests, normal point data and original images are preserved. Developer selection, repeat cycles and left/right hands are correlated; no binomial safety confidence interval is claimed. No new full System0 task trial or true-grasp/loaded-release qualification.
