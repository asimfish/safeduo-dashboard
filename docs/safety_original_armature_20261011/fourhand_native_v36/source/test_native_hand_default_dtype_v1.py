import unittest,json
from pathlib import Path
import torch,numpy as np
class NativeDtypeTests(unittest.TestCase):
    def test_original_error_is_reproduced_and_all4_native_hand_defaults_preserve_other_joints(self):
        reg=json.loads((Path(__file__).parent.parent/'fourhand_native_probe_v17/REGISTRATION_V1.json').read_text())
        with np.load(reg['coupled_reference_endpoints']) as z:
            for arm in ['F_L','F_R','U_L','U_R']:
                ids=z[arm+'_hand_indices'].tolist();source=z[arm+'_hand_open_q'];self.assertEqual(source.dtype,np.float64)
                q=torch.full((1,max(ids)+1),-.123,dtype=torch.float32);old=q.clone()
                with self.assertRaisesRegex(RuntimeError,'dtypes match'):q[:,ids]=torch.tensor(source)
                q[:,ids]=torch.tensor(source,dtype=q.dtype)
                np.testing.assert_allclose(q[0,ids].numpy(),source,atol=1e-7,rtol=0)
                other=[i for i in range(q.shape[1]) if i not in ids];self.assertTrue(torch.equal(q[:,other],old[:,other]))
if __name__=='__main__':unittest.main()
