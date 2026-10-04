"""Oracle: unchanged pressure tape, exact prefix and missing-quota rejection."""
import unittest
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_random_space_20261004'))
from wide_random import make_tape
from risk_recipe import prepend_zero,validate_bank

class RecipeTests(unittest.TestCase):
    def test_prefix_retains_all_pressure_inputs(self):
        old,info=make_tape(64,900,.05,123,'mixed_hold')
        tape,meta=prepend_zero(old,info)
        self.assertEqual(tape.shape,(962,64,26))
        self.assertFalse(tape[:60].any())
        self.assertTrue(torch.equal(tape[60:],old))
        self.assertEqual(meta['random_steps'],900)
        for key in ['updates','holds','segment_amplitudes']:
            self.assertFalse(meta[key][:60].any())
            np.testing.assert_array_equal(meta[key][60:],info[key])
        self.assertNotEqual(meta['tape_sha256'],info['tape_sha256'])
    def test_missing_quota_or_outcome_selection_rejected(self):
        q=np.zeros((64,26));labels=np.repeat(np.arange(6),8);labels=np.r_[labels,[-1]*16]
        meta=dict(status='complete',selected_count=64,policy_outcomes_used=False)
        validate_bank(q,labels,meta)
        bad=labels.copy();bad[0]=1
        with self.assertRaisesRegex(ValueError,'no padding'):validate_bank(q,bad,meta)
        with self.assertRaisesRegex(ValueError,'pre-policy'):validate_bank(q,labels,{**meta,'policy_outcomes_used':True})

if __name__=='__main__':unittest.main()
