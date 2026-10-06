import unittest,numpy as np
from coverage import histograms,pair_summary,zero_failure_upper,zero_failure_required
class Tests(unittest.TestCase):
    def test_marginal_full_does_not_mean_pair_full(self):
        q=np.repeat(((np.arange(10)+.5)/10)[:,None],26,axis=1)
        a,b,outs,states=histograms(q,np.array([[0.,1.]]*26))
        np.testing.assert_equal((a>0).sum(-1),[10]*26)
        np.testing.assert_equal((b>0).sum(-1),[10]*325)
        self.assertEqual(pair_summary(b)['mean_percent'],10.)
        self.assertEqual(int(outs.sum())+states,0)
    def test_outside_values_not_clipped_into_coverage(self):
        q=np.repeat(np.array([0.,1.,-1.,2.])[:,None],26,axis=1)
        a,b,outs,states=histograms(q,np.array([[0.,1.]]*26))
        np.testing.assert_equal(a[:,0],[1]*26);np.testing.assert_equal(a[:,-1],[1]*26)
        np.testing.assert_equal(a.sum(-1),[2]*26);np.testing.assert_equal(b.sum(-1),[2]*325)
        np.testing.assert_equal(outs,[2]*26);self.assertEqual(states,2)
    def test_all_pairs_accounted_for(self):
        _,b,_,_=histograms(np.zeros((1,26)),np.array([[-1.,1.]]*26));s=pair_summary(b)
        self.assertEqual(s['within_arm_pairs']+s['cross_arm_pairs'],325)
    def test_each_environment_uses_its_own_bounds(self):
        q=np.repeat(np.array([[.75,1.5],[1.5,1.5]])[...,None],26,axis=-1)
        limits=np.repeat(np.array([[[0.,1.]],[[0.,2.]]]),26,axis=1)
        a,b,outs,states=histograms(q,limits)
        np.testing.assert_equal(a[:,7],[3]*26);np.testing.assert_equal(a.sum(-1),[3]*26)
        np.testing.assert_equal(b[:,77],[3]*325);self.assertEqual(states,1)
    def test_rare_zero_failure_sample_bound(self):
        self.assertAlmostEqual(zero_failure_upper(192),1-.05**(1/192))
        self.assertEqual(zero_failure_required(.001),2995)
        self.assertLessEqual(zero_failure_upper(2995),.001)
        self.assertGreater(zero_failure_upper(2994),.001)
    def test_invalid_sample_rejected(self):
        for n in (0,-1,192.):
            with self.assertRaises(ValueError):zero_failure_upper(n)
if __name__=='__main__':unittest.main()
