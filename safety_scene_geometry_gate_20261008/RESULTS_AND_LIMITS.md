# Scene geometry admission: results and scope

This is supplementary fixed-pose empty-hand qualification. Four arm targets stay fixed at their registered reference values while two U hands receive the registered goals. It adds no System0 task trial, grasp/load/release trial, dynamic arm command campaign, or independent full operating-domain acceptance.

## Independently scored completed runs

| Run and method | U-hand requests | Admitted | Executed and qualified | Refused | Refused and neutral-safe | Aborts | Maximum measured scalar hand/partner normal force |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched old_v3 | 96 | 69 | 67 | 27 | 27 | 2 | 0.839659 N |
| Matched scene_v4 | 96 | 24 | 24 | 72 | 72 | 0 | 0 N |
| Fresh 24-pose scene_v4 | 576 | 162 | 162 | 414 | 414 | 0 | 0 N |

The matched scene gate admitted 22 moving paths and two neutral controls. The fresh gate admitted 148 moving paths and 14 neutral controls. Neutral controls are not counted as moving task completion. A refusal's neutral-safe outcome is not an executed grasp or task success.

Each completed run contains 8640 recorded steps. The matched run has eight environments, 69120 environment states and 360 original images. The fresh run has 24 environments, 207360 environment states and 340 original images. These are correlated step records, not 276480 independent trials. Both runs use 12 registered goals and the original fixed phase schedule and thresholds.

Independent scoring reconstructs scalar normal forces from the absolute magnitudes of valid original native contact points in float64; vector cancellation cannot remove a contact from this oracle. The qualification threshold is 0.1 N for each hand sensor/partner aggregate across the path, with registered speed, tracking, target and joint-limit checks. A reported 0 N is an observation of this recorded stream, not proof of all-body collision freedom. Constructor contacts are not qualified.

## Conservatism and coverage

The new gate admitted 25% of matched requests and 28.125% of fresh requests. It refused 43 paths that the old method executed and qualified, of which 39 had moving goals. This measures an observed paired tradeoff; it is not a calibrated false-rejection probability because the other hand's subsequent behavior can differ after old latching or new refusal.

The fresh bank uses 24 randomly drawn, stratified frames from an existing reference trajectory, with old qualification frames excluded. Its 26 joint marginal spans occupy approximately 1.29%–27.01% of their hard-limit widths. Twelve old allowed hand goals are reused. The bank does not establish coverage of the full 26-dimensional workspace, arbitrary hand goals, layouts, masses, friction, gains, sensing uncertainty or delay variation. Rerunning the same frozen bank adds no new independent sample.

## Mechanism and evidence limits

The geometry extractor accounts for all 192 enabled inventoried colliders, including guide-purpose collision meshes and primitive dimensions that disagree with stale authored extents. Eleven stale extent cases are corrected. The sweep uses native current poses, registered FK checks and declared geometric padding. Its prediction excludes a hand's own arm colliders; the prior exact-goal passport and native all-partner contact monitor remain necessary. This is not a general self-collision certificate.

The interrupted original fresh attempts and failed batch starts remain separate and are not pooled with the completed qualification. The batch recorder's positive-contact pilot retains 4805 nonzero sensor/partner frames and 16485 original point comparisons. The cached offline mapping audit reproduces every existing pilot result field exactly, without changing native runtime sources or thresholds. Final full-run mapping and publication status must be read from their generated gate receipts; this narrative is not a replacement for those receipts.

The six planned panel clips comprise three direct native clips and three labelled state-reconstruction clips. Reconstruction advances no qualification physics and adds no trial. Direct occluded views and the original images remain available. Source registrations, independent result JSON, point streams, images and negative attempts are retained for inspection.

## Broader four-arm acceptance remains rejected

The separately published 128-initial-state, four-method, two-seed-block experiment contains 512 complete native windows and still rejects the overall candidate. It must not be replaced by this limited hand result. Its eight newly failing candidate cases and shared invalid startup require actual motion, initialization and stopping analysis. Locally favorable target displacement or zero current linear residual does not establish next physical-state safety. Full System0, real grasp/load/release and broader operating-domain acceptance remain open.
