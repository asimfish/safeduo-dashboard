"""RED original actual-attribute closure / GREEN corrected capture, no simulator."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import observer_runner as original
from observer_runner_v2 import Observer
from safeduo.safety.types import ARM_KEYS
class Tests(unittest.TestCase):
    def exercise(self,kind):
        calls=[];j={'F':torch.zeros(1,9021,14),'U':torch.zeros(1,9021,12)}
        rows=NS(J=j);out=NS(active_idx=torch.arange(9021)[None])
        def raw(a,b):calls.append(a);return rows
        provider=NS(rows_from=raw)
        q={a:torch.zeros(1,7 if a.startswith('F') else 6) for a in ARM_KEYS}
        state=NS(q=q,qd=q,dt=.016666)
        full=NS(dists=torch.zeros(1,9021),closing=torch.zeros(1,9021),full_viol_exempt=torch.zeros(1,9021,dtype=torch.bool))
        env=NS(_provider=provider,scene_state=lambda:state,_delta_src=NS(info={}),_last_out=full,
               _evaluation_actuator_delay=NS(queue=NS(pending=[q])))
        sentinel=object()
        def fake_start(trace,e):
            trace.mode='admission_full';trace.frames={};trace.original_rows=e._provider.rows_from
            def checked(a,b):return trace.original_rows(a,b)
            def safety():checked(out,None);return sentinel
            e._provider.rows_from=checked;e.safety_dist_out=safety
        with tempfile.TemporaryDirectory() as directory:
            trace=kind.__new__(kind);trace.out=Path(directory)
            with patch.object(original.base.GuardTrace,'start',fake_start):trace.start(env)
            result=env.safety_dist_out()
            self.assertIs(result,sentinel);self.assertEqual(len(calls),1)
            self.assertIs(trace.full_J['F'],j['F']);self.assertTrue(torch.equal(trace.pred,torch.zeros(1,9021,4)))
    def test_original_seam_is_red(self):
        with self.assertRaisesRegex(AttributeError,'full_J'):self.exercise(original.Observer)
    def test_captured_actual_callable_is_green(self):self.exercise(Observer)
if __name__=='__main__':unittest.main()
