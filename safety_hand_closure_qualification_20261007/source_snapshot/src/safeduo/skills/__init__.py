"""SafeDuo skill library, organised by the owner's five task categories
(SKILL_LIBRARY_PLAN_20260908.md):

  1 dual_independent   two arms, each on its own object
  2 dual_coordinated   two arms coupled through one object (handover / co-lift / relay)
  3 quad_independent   four arms, the two pairs work independently
  4 quad_cooperative   four arms in one task chain (relay across pairs, assembly)
  5 random             unscripted operator streams -- the safety evidence

The taxonomy module is a registry over the existing task families
(task_library_r24/r26/r34/r35, task_record_s9, the v7 kinematic skills and
the random flows); the spec/compile modules turn declarative task specs into
S9Task programmes for the existing IK gates and physics-grasp recorder.
"""
