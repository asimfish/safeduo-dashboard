"""Cheap state/FIFO dependency and known drift counterexamples, not native proof."""
import unittest
import numpy as np
from actual_state_arm_guard_v1 import ARMS,ActualStateArmGuard


def setup():
    sizes={'F_L':19,'F_R':19,'U_L':18,'U_R':18}
    ids={a:np.arange(7 if a.startswith('F') else 6) for a in ARMS}
    bounds={a:np.tile([-2.,2.],(n,1)) for a,n in sizes.items()}
    caps={a:np.ones(n) for a,n in sizes.items()}
    # A plane-like critical pair; remaining9020 gaps fixed positive.
    def gaps(q):
        n=len(q['F_L']);g=np.full((n,9021),.1)
        g[:,0]=.002-q['F_L'][:,0]
        return g
    retreat={a:np.full(len(ids[a]),-.5) for a in ARMS}
    guard=ActualStateArmGuard(gaps,bounds,caps,ids,retreat)
    q={a:np.zeros(n) for a,n in sizes.items()};v={a:np.zeros(n) for a,n in sizes.items()}
    applied={a:np.zeros(len(ids[a])) for a in ARMS}
    pending=[{a:z.copy() for a,z in applied.items()} for _ in range(6)]
    actual=gaps({a:q[a][ids[a]][None] for a in ARMS})[0]
    return guard,q,v,applied,pending,actual


class Contracts(unittest.TestCase):
    def test_accepts_clear_slow_request(self):
        g,q,v,a,p,d=setup();a['F_L'][0]=-.1
        r=g.decide(q,v,a,p,d)
        self.assertEqual(r.mode,'REQUEST');self.assertLess(r.target['F_L'][0],0)
        self.assertFalse(r.prediction_qualified)

    def test_measured_velocity_changes_decision_even_with_same_command(self):
        g,q,v,a,p,d=setup();a['F_L'][0]=.0002
        calm=g.decide(q,v,a,p,d);v['F_L'][0]=.02
        drifting=g.decide(q,v,a,p,d)
        self.assertEqual(calm.mode,'REQUEST')
        self.assertNotEqual(drifting.mode,'REQUEST')
        self.assertLess(drifting.target['F_L'][0],calm.target['F_L'][0])

    def test_queue_changes_approval_even_with_same_actual_and_current_command(self):
        g,q,v,a,p,d=setup();a['F_L'][0]=.0008
        calm=g.decide(q,v,a,p,d)
        for row in p:row['F_L'][0]=1.
        conflict=g.decide(q,v,a,p,d)
        self.assertEqual(calm.mode,'REQUEST');self.assertNotEqual(conflict.mode,'REQUEST')

    def test_native_gap_anchor_can_reject_source_positive(self):
        g,q,v,a,p,d=setup();d[0]=-.00001
        r=g.decide(q,v,a,p,d)
        self.assertEqual(r.mode,'UNQUALIFIED_BRAKE');self.assertLess(r.actual_minimum_m,0.)

    def test_measured_hand_speed_is_not_ignored(self):
        g,q,v,a,p,d=setup();v['U_L'][-1]=1.1
        self.assertEqual(g.decide(q,v,a,p,d).mode,'UNQUALIFIED_BRAKE')

    def test_exact_fifo_six_and_finite_state_required(self):
        g,q,v,a,p,d=setup()
        with self.assertRaises(ValueError):g.decide(q,v,a,p[:5],d)
        q['F_L'][0]=np.nan
        with self.assertRaises(ValueError):g.decide(q,v,a,p,d)

    def test_no_actual_state_or_queue_mutations(self):
        g,q,v,a,p,d=setup();oldq={k:z.copy() for k,z in q.items()};oldp=[{k:z.copy() for k,z in row.items()} for row in p]
        g.decide(q,v,a,p,d)
        for k in q:np.testing.assert_array_equal(q[k],oldq[k])
        for old,new in zip(oldp,p):
            for k in old:np.testing.assert_array_equal(old[k],new[k])


if __name__=='__main__':unittest.main(verbosity=2)
