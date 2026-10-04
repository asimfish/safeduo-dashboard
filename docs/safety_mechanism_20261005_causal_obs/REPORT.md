# Queue-envelope admission: complete new-command experiment

The strict100ms FIFO baseline fails **120/192** windows; queue-envelope admission fails **40/192** on exactly matched fresh external command tapes and original initial poses. These are empirical sphere-pressure results. Safety strategy remains **BLOCKED**, candidate not promoted, hardware not approved.

A reactive projection first sees some rows only after several closing targets are already queued. The candidate admits rows forecast dangerous at actual pending targets, current issued targets and boxed/soft-limited next command proposals. It preserves the original actor32, original solver/backlog_aware=false, geometry, thresholds, actuators and strictFIFO6. Candidate capacity1024 aborts without dropping; baseline512 remains frozen.

## Frozen paired command holdout

| New command seed | Baseline failures | Candidate failures | Rescued | New failures | Both fail | Both pass |
|---|---:|---:|---:|---:|---:|---:|
| 1930442907 | 44/64 | 13/64 | 31 | 0 | 13 | 20 |
| 1931983758 | 37/64 | 14/64 | 25 | 2 | 12 | 25 |
| 1940677742 | 39/64 | 13/64 | 26 | 0 | 13 | 25 |

384 complete related windows,192 new external tapes,6 independently launched processes. Rescued82,new failures2. The two new failures are seed1931983758 env11 (first833,min table−19.353mm) and env58 (first724,min table−14.778mm); all candidate failures remain visible. No initial violation, partial window, silent reset or row-capacity failure is removed. Source/actor/config identities, pairwise initial/tape equality, original target-limit/increment numerical gates and all960 actual FIFO targets were verified. Recorded dt=.016666s gives FIFO6=.099996s (nominal100ms), with960 steps nominal16s. The target increment box permits the frozen+5e−7rad readback tolerance; measured maximum exceeds nominal .024999rad by1.42e−7rad. This numeric tolerance is not used for negative safety margins.

Three new seeds were generated from OS entropy before candidate decision/outcomes. Each uses a pre-existing policy-blind stable bank:6 targeted arm pairs×8 plus16 general poses. Bank reuse is explicit; no new initial-space coverage or IID safety probability is claimed. External per-arm directions, durations and amplitudes are independent mixed random commands; amplitude0.005/0.015/0.025/0.05rad,hold1/4/15/30/90/180steps. Every original failed window stays in results.json.

## Movement, failure classes and pressure exposure

| Method | Deep<-5mm windows | cross/self_F/self_U/table | Four-arm moving fraction | Pooled exec/command L2 | Actual max rows | >512 env-steps |
|---|---:|---|---:|---:|---:|---:|
| Baseline | 105 | 8 / 28 / 54 / 90 | 46.638% | 0.575539 | 146 | 0 |
| Queue-envelope | 30 | 0 / 0 / 2 / 38 | 43.633% | 0.569234 | 217 | 0 |

Class counts overlap. Four-arm movement means each arm measured joint movement>1mrad in the same pressure control transition; output L2 is not actual motion or task success. Exposure begins after the first pressure target delivery at index66;>=3 samples below80mm are required. Shortfalls are retained.

| Method | Assigned risk pair | Exposed / attempted | Approach EE steps | Recede EE steps |
|---|---|---:|---:|---:|
| Baseline | F_L-F_R | 24/24 | 7045 | 9463 |
| Baseline | F_L-U_L | 20/24 | 6873 | 10257 |
| Baseline | F_L-U_R | 22/24 | 7066 | 10466 |
| Baseline | F_R-U_L | 24/24 | 6457 | 9862 |
| Baseline | F_R-U_R | 21/24 | 6490 | 10525 |
| Baseline | U_L-U_R | 24/24 | 6781 | 8466 |
| Queue-envelope | F_L-F_R | 24/24 | 7021 | 9589 |
| Queue-envelope | F_L-U_L | 20/24 | 6796 | 10054 |
| Queue-envelope | F_L-U_R | 22/24 | 7229 | 10247 |
| Queue-envelope | F_R-U_L | 24/24 | 6218 | 9751 |
| Queue-envelope | F_R-U_R | 21/24 | 6526 | 10352 |
| Queue-envelope | U_L-U_R | 24/24 | 6641 | 8452 |

A separate concurrent fixed-factor experiment is also published in the main scientific panel. Its debt-amplitude limiter, initial banks, command seeds and denominators differ from this admission candidate. Both evidence sets are preserved; no pooled headline or shared holdout claim is made.

Joint10-bin and EE10cm occupancy describe observed marginal/Cartesian cells, not26D joint coverage or collision-free reachable volume. UR periodic angles may describe equivalent poses. Detailed coverage is in results.json.

## Development and independent evidence boundaries

Old development seed60317411 was viewed before candidate selection. Passive observer reproduces eight numerical/state/target fields exactly and39/64 failures. Candidate has17/64, including3 new table failures and25 rescued old failures. Development is not holdout.

Independent GPT-6 Astra xhigh reviewed the concrete helper, contracts and data. It found a missing all-forecast finite validation gate. Frozen v1 holdout is unchanged and does not have evidence that every unadmitted forecast was finite. Separate v2 diagnostic aborts before physics for nonfinite full d/dmin/closing/J/forecast and q/qd/issued/queued/proposal targets. Its120-step development prefix is exact to v1 in all seven measured/control fields;3 fault/immutability tests pass. These120 transitions do not add full16s windows or certify v2 generalization.

Stable-row receipts show env47/row8974 enters at step62 rather than67,env35/row7310 at63 rather than68, before first pressure delivery66. Ordered actual queue is exact. This supports earlier admission for these cases, not a unique global root cause or proof that failure was physically inevitable. A zero residual is for authority-clamped linear constraints, not a physical safety guarantee.

Latest v4 camera development replay:64 complete windows,17 failures,12groups/84real1280×720PNG. Four preselected slots at steps71/75/959,seven angles including full overview,front/reverse,F/U pair and two U opposing heights. Direct native PhysX all64 q/qd/root pose/root velocity before and after each slot's seven renders are exactly equal by completed source-bound assertions. Three full native snapshots are saved; all64 controlled q matches the dense trace at all three steps and qd matches next pre-state at71/75, with final qd bound to its native capture. Selected state, sphere centers, targets and class minima match the replay's own dense trace.

84 actual USD camera transforms and authored optical parameters are recorded; calibration and selected-arm sphere center projection checks pass for every view. This measures center framing, not pixel visibility or complete link silhouettes. Actual SDK update_latest_camera_pose defaults false, so v3 saved zero positions despite moving views; that metadata was rejected. v4 reads the camera prim directly and preserves stale SDK readings separately. v2 cached-state84 images and v3 native-state84 images are retained as earlier attempts; they are not pooled into the latest84 image claim. The original v1 startup failed before step0 and is retained separately.

Visualcuda0 versus numericalcuda1 initial/tape comparison exact, but q/execution/margins/controller/actuator trajectory exact FAIL. Do not use photos as the numerical holdout counterpart. No new holdout tape is added by any visual development replay.

The original192 physical carry tasks remain19full-gate passes and their original gates/failed attempts are preserved in the prior sealed campaign. No grasp/contact/orientation/hardware success is inferred from this pressure study. Control endpoints are sampled at16.666ms; continuous collision, realtime deployment, impulses/full frictional wrenches and hardware safety remain unverified.

A parallel parent also used the initial shared mechanism folder. PLAN.md there was inadvertently overwritten before concurrent ownership was detected; expected prior hash and collision are in NAMESPACE_NOTE.md. Subsequent work uses the separate causal_obs namespace; foreign sources, plans and processes were not changed. Only owned closed evidence is archived.

## Next frontier

Residual table failures require a controlled experiment on committed queue motion, original versus authority-limited feasibility, nonlinear target-path geometry and tracking response. Use a newly registered development/holdout split for any next candidate, retain strict FIFO unless actual architecture authorization changes. Full16s v2 finite-gate evaluation and object U-grasp retention remain separate unfinished strategy milestones.

Evidence delivery gate can pass while safety strategy remains BLOCKED. Public interactive panel includes every new paired window,class curve,negative outcome,measured exposure and84actual camera images; original main/data panel fields remain intact.
