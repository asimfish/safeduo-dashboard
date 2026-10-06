# Ground-truth object binding development diagnostic

Observed: the sealed 20261004 two-pair task passed 19/192. Objects were
perturbed independently, but SkillReplayDelta sampled joint deltas from one
fixed trajectory. DuoEnv.scene_state has no object pose/size/goal fields.
The R37 wrapper adds joint command debt, not object feedback. Its unchanged
planned-lift sensor gate does not condition on actual object separation.

Expected: before claiming real-task safety, bind a timestamped, framed object
state to the task planner, validate the resulting command changes, and observe
actual lift/contact/release independently. Keep the original task and sensor
failures; a new diagnostic cannot turn them into passes.

Hypotheses:
1. Fixed grasp references miss perturbed objects. Prediction: native initial
   pose changes must change pickup references when the binding is enabled;
   nominal references must remain exactly unchanged.
2. Hands never achieve/maintain a friction grasp. Prediction: lifted states
   lack simultaneous object-attributed contact from both assigned hands.
3. Scheduled opening/retreat drags the payload. Prediction: object pose starts
   to diverge during release/clearance while native hand contact persists.
4. The old scheduled support check mixes actual table contact with airborne
   states. Prediction: positive support is predominantly in states whose
   rotated authored bounding box has not cleared the table. This alone does
   not prove the complete contact sensor or collision oracle is valid.

This is development work. Retrospective analysis uses only blocks 0 and 1
(128 known development tasks), never tunes on the frozen block 2. A new
8-case native contrast uses four explicit layouts, repeated under fixed
replay and ground-truth initial-pose binding. No IID, global workspace,
vision-model, new final holdout, hardware, or safety-certification claim.

The initial-pose binding rigidly translates/rotates the nominal pickup poses
about each object's authored center and smoothly removes that correction by
the existing placement phase. It uses native object pose/velocity, authored
size, fixed task target, validity, frame and state time. It is an upstream task
planner input, not a new neural-policy observation/checkpoint. Safety control,
hand events, old task thresholds, gravity, friction and solver rules stay fixed.
No object follow, fixed grasp constraint, pose writes after initialization,
velocity zeroing during execution, target snapping, or sensor-based verdict
replacement is permitted. Registration precedes the new physical run.

Native contact observations are filtered normal forces at the last physics
substep. Exact partner paths and capacity checks do not prove full friction,
wrench, impulse, all-scene collision coverage or hardware safety.
