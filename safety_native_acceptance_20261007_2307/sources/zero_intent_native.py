MODES=('joint_reference','zero_inclusive','box_admission','adaptive_joint')
def reference_gap(mode):
 if mode not in MODES:raise ValueError('unregistered native acceptance mode')
 return .050 if mode=='joint_reference' else .010
