# Camera and object audit — 2026-09-18

## Confirmed observation gap
`src/safeduo/envs/duo_env.py:1321` assembles joint positions/velocities, active robot sphere pairs/masks, proposed joint delta, previous alpha/p, optional hazard/backlog/coupling features. No object pose, extent, velocity or image is supplied. Lines 421 and 949 explicitly exclude task objects from the sphere safety model/observation. A31b cannot be described as object-aware. Adding object inputs requires a versioned observation layout, compatible training/checkpoint and object/carry collision semantics; changing camera placement alone does not fix it.

## Layout evidence
Crossrow beam geometry local extents are 0.05 x 0.70 x 0.05 m, yaw 90 degrees, centres (-0.05,-0.30,0.825) and (-0.05,+0.30,0.825). Their world AABB separation along y is 0.55 m: no initial object-object overlap. This does not certify reachable or feasible grasping. Prior physical runs lifted neither primary beam. The grasp contact geometry and free hand approach volume need phase-specific checking. Converge has three cans and a bottle, not a shared payload.

## Recording implementation
Independent `safeduo_setup/record_task_multiview.py` adds 3 Camera sensors before simulation play. Four synchronized 1280x720 views form a 2560x1536 composite with labels outside each view. Original overview plus top/front/side manipulation cameras. Snapshots at frame 0/360/600/900/1200. Camera-only recorder does not alter task inputs or checkpoint. Preview uses a separate simplified scene config: removed static props including 3 colliding fixtures (taskboard, spare KLT, scale), restored physical table visibility, and disabled display remapping. This is a NEW scene, not equivalent to prior evidence. Preview frames at 0 s and 6 s inspected. 660 control steps recorded (11 simulation seconds, 22 playback seconds), four views synchronized. New clip is partial and does not certify full task success.

## Unresolved
Do not claim improved safety or fixed grasps from new views. Closed-hand snapshot confirms objects remain on table. All-object traces (661 samples) show beam_neg lift 0 m and beam_pos lift 0.000352 m; object-aware policy changes require a separate compatible experiment.
