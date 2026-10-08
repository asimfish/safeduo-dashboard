import copy,numpy as np
from astra_native_audit import synthetic_packet_fixture
from audit_native import packet_check as fixed
from audit_native_before_review import packet_check as old
cell,initial,packets=synthetic_packet_fixture()
for t,p in enumerate(packets):fixed(p,cell,t)
t=7;p=packets[t]
for key in ['pre_F_L_applied_controlled_target','physics_U_L_pending_project_targets','post_F_R_pending_project_targets']:
 changed={k:x.copy() for k,x in p.items()};changed[key].flat[0]+=.125
 old(changed,cell,t)
 try:fixed(changed,cell,t)
 except AssertionError:pass
 else:raise AssertionError('historical omission still escapes '+key)
 print('RED old accepts wrong field; GREEN fixed rejects',key)
print('PASS9 full valid synthetic frames; 3 historical false-positive controls now rejected; no actual physics claimed')
