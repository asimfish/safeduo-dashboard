# Research acceptance still required

This round resolves observed pre-closure scene collisions and tests a frozen static reference bank. It does not close full four-arm System0 acceptance.

Next stage must define the operating domain before sampling: all four arms' joint limits, reachable object placement volumes, six arm-pair overlap regions, manipulator/object/fixture clearances, collision-free startup, grasp constraints and moving carried-object envelopes. Report coverage separately for each arm, six pairs, workspace cells, boundary distance and task/contact phase. Marginal joint spans cannot establish joint-space volume coverage.

Freeze a separate broad campaign with independently randomized task selection, object size/pose/spacing, collision-free arm starts and bounded joint targets. Include full-range uniform and stratified/LHS sampling, narrow collision boundaries, adversarial crossing, four-arm cooperative tasks and paired bypass/System0 trials. Use identical initial states and random draws across ablations. Retain refused/unreachable setups and distinguish valid operating-domain coverage from safety interventions.

Acceptance requires task success alongside contact/clearance/velocity/limit safety, all-partner point measurements, grasp stability and load/release outcomes. Freeze metrics, exclusions, failure taxonomy and stopping rules before qualification. Calibrate perception and prediction uncertainty; include sensing delay/noise and dynamic obstacle disturbance. Native constructor contact measurement and material/asset dependency sealing remain required. Count independent tasks, report paired uncertainty by task-level clusters, and keep separate development/qualification banks.
