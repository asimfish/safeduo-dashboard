import unittest
import numpy as np
from random_input import make_recipe,WIDTHS,AMPS,HOLDS
class Tests(unittest.TestCase):
    def test_balanced_real_temporal_exposure(self):
        tape,m=make_recipe(912437)
        self.assertEqual(tape.shape,(962,64,26));self.assertTrue(np.isfinite(tape).all())
        self.assertFalse(tape[:60].any());self.assertLessEqual(float(np.abs(tape).max()),.05+1e-8)
        np.testing.assert_equal(np.bincount(m['order']),[32,32])
        for e in range(64):
            for value in (1,2):self.assertEqual(int((m['regime'][60:960,e]==value).sum()),450)
            off=0
            for arm,w in enumerate(WIDTHS):
                for start in (60,510):
                    stop=start+450;upd=np.flatnonzero(m['updates'][start:stop,e,arm])+start
                    self.assertEqual(upd[0],start)
                    for a,b in zip(upd,np.r_[upd[1:],stop]):
                        np.testing.assert_equal(tape[a:b,e,off:off+w],np.broadcast_to(tape[a,e,off:off+w],(b-a,w)))
                        if m['regime'][a,e]==2:self.assertEqual(b-a,1)
                off+=w
        self.assertTrue(set(np.unique(m['segment_amplitudes'][60:960]))==set(AMPS))
        self.assertTrue(set(np.unique(m['holds'][60:960]))==set(HOLDS))
if __name__=='__main__':unittest.main()
