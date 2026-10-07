# 原生开手与支撑测量诊断

Observed: 8/8 feedback task windows abort. All 120 waiting states per case fail default-open <=0.2rad. Otherwise contact-free cases remain about 0.27rad from default. Table-only support under-attributes U support in some old windows.
Expected: independently qualify native target versus actual open angle, identify its physical contact constraints, and attribute static support before changing any guard.
Scope: IsaacSim5.1 / IsaacLab0.54.2 / same four arms and collision assets, seed90610751. Six static diagnostics, NOT successful task trials or held-out experiments.

|Hypothesis|Prediction / discriminator|
|---|---|
|Native limits/default ordering disallows zero|Read native limits and named DOF order; zero lies outside range or getter/cache mismatch.|
|Driver supplies a nonzero target when reporting open|Direct PhysX target readback differs from original HandDriver default.|
|Self contacts constrain the default-open pose|Error and contact persist without payload; independently registered self-collision-off counterfactual reduces error.|
|External fixture or gravity/tracking causes offset|Depends on arm pose, external contacts, or remains with self-collision off.|
|Missing instance proxy fixtures support U payload|New disjoint native fixture filters receive forces missing from old table filter; full normal reconstruction/net accounting closes.|

Before evaluation, preserve source SHA, native metadata and all raw chunks. Fixed diagnostic windows and accounting tolerances are in REGISTRATION.json. Capacity, contact identity, normal reconstruction and net/filter sum are measurement qualification gates; failing them is not a physical safety failure or success. No change to old scorer or production guard.
