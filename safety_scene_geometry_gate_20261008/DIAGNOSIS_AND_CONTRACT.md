# Scene geometry hand prevention, 2026-10-08

The V3 fresh reference bank is now development evidence. Two first-cycle U_L
thumb contacts against the NIST board violated the unchanged 0.1 N scalar
normal-force criterion at references 19.6333 and 19.85 s. Exact time/goal
exclusions cannot prevent nearby spatial failures.

Discriminating hypotheses: (1) the exact-context list misses the same collider
at nearby poses; (2) guide-purpose collision geometry was omitted from the old
default-purpose BBoxCache, yielding empty float-max bounds; (3) arm/hand tracking
drift consumes nominal clearance; (4) an incorrect joint/quaternion convention
would invalidate predictive geometry. Inspect guide meshes and compare analytic
hand FK to native recorded poses before using geometry for admission.

This opt-in experiment preserves the 12 exact hand goals, internal-contact
qualification, native scalar monitoring, 3 rad/s bound and latched empty-hand
return. It removes time-index exclusions as a prevention mechanism. A new gate
predicts opening-to-goal-to-opening geometry using USD joint frames and mesh points/physical primitive dimensions, without writing simulated state. Enabled collision geometry includes all purposes. Authored extent is ignored: KLT Cube size is about 1 but stale extent spans 100, producing 100x boxes. The first offline diagnostic using those extents is preserved and rejected. Offline USD without Kit cannot resolve remote FR3 arm references; its FK/fixture diagnostic is U-hand scoped. Native Kit extraction must cover all four robots. Unsupported/empty/nonfinite
geometry or inaccurate FK refuses admission. Collider boxes contain source
geometry and its convex hull; they may cause conservative false refusals.

The scene gate checks tables, fixtures, objects and all other robots. Other U
hand requested paths are bounded by their full swept boxes, so simultaneous
closure cannot evade the check. Hand internal contact still relies on the frozen
exact-goal bank plus native monitoring; this is not a new full internal geometry
proof. No intended object contact exemption exists in this empty-hand experiment.

Freeze before native qualification: 121 interpolation samples; a per-collider
chain-length Lipschitz padding for the space between samples; 20 mm additional
geometric reserve; at least 20 mm contact-offset reserve on each collider when
an explicit native value is absent. These reserves are conservative engineering
assumptions, not a certified stochastic calibration bound. FK must match current
native hand link positions within 0.5 mm and quaternion components within 0.002
(sign invariant). Every first post-initialization physical frame is scored.

Primary oracle is the independent original native point magnitudes summed per
sensor/partner, <=0.1 N throughout each full six-second path. Also require native
tracking <=0.02 rad in closed/returned windows, target reconstruction <=1e-6 rad,
finite/bounded state, complete point counts/identity and unchanged sources.
Refusal, abort, safe return and executed success are separate counts. Full
scientific acceptance fails on any admitted path violation. All attempted paths
and initialization failures are retained. Old inputs are immutable.

First diagnose on archived V3 failures, then execute a matched native V3/geometry
comparison. Freeze runtime and a new OS-random reference bank before its native
run. This remains static four-arm configurations with two hand closures, not
four-arm cooperative manipulation, full joint-space coverage, grasp/load safety,
or formal System0 arm projection validation.
