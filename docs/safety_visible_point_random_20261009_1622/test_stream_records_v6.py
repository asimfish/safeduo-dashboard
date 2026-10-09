import json, tempfile, unittest
from pathlib import Path
import numpy as np
from stream_records_v6 import StreamRecords
class Checks(unittest.TestCase):
    def test_exact_export_and_commit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);w=StreamRecords(p/'fields',9)
            records=[]
            for i in range(9):
                r=dict(signed=np.array([-0.,float(i)],dtype=np.float32),index=np.asarray(i,dtype=np.int64),bools=np.array([i%2==0],dtype=bool));records.append(r);w.append(r)
            w.close(p/'export.npz')
            with np.load(p/'export.npz') as z:
                for k in records[0]:
                    expected=np.stack([r[k] for r in records]);self.assertEqual(z[k].dtype,expected.dtype);self.assertEqual(z[k].tobytes(),expected.tobytes())
            self.assertEqual(json.loads((p/'fields/closed.json').read_text())['filled'],9)
    def test_partial_never_closed(self):
        with tempfile.TemporaryDirectory() as t:
            w=StreamRecords(Path(t)/'fields',8);w.append({'x':np.zeros(2)})
            with self.assertRaisesRegex(ValueError,'incomplete'):w.close(Path(t)/'export.npz')
            self.assertFalse((Path(t)/'export.npz').exists())
    def test_schema_change_fails(self):
        for case in [{'x':np.zeros(3)},{'x':np.zeros(2,dtype=np.float32)},{'y':np.zeros(2)}]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                w=StreamRecords(Path(t)/'fields',2);w.append({'x':np.zeros(2)})
                with self.assertRaises(ValueError):w.append(case)
                self.assertEqual(w.filled,1)
    def test_object_and_unsafe_name_fail(self):
        for case in [{'x':np.asarray([object()])},{'../x':np.zeros(2)}]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                w=StreamRecords(Path(t)/'fields',2)
                with self.assertRaises(ValueError):w.append(case)
    def test_excess_denominator_fails(self):
        with tempfile.TemporaryDirectory() as t:
            w=StreamRecords(Path(t)/'fields',1);w.append({'x':np.zeros(2)})
            with self.assertRaisesRegex(ValueError,'denominator'):w.append({'x':np.zeros(2)})
    def test_export_map_pages_reclaimed(self):
        with tempfile.TemporaryDirectory() as t:
            w=StreamRecords(Path(t)/'fields',128);v=np.arange(262144,dtype=np.float32)
            for i in range(128):w.append({'x':v})
            w.close(Path(t)/'export.npz')
            resident=int(Path('/proc/self/statm').read_text().split()[1])*4096
            self.assertLess(resident,160*1024*1024)
    def test_stream_resident_bound(self):
        import resource
        with tempfile.TemporaryDirectory() as t:
            w=StreamRecords(Path(t)/'fields',128);v=np.arange(262144,dtype=np.float32)
            for i in range(128):w.append({'x':v})
            resident=int(Path('/proc/self/statm').read_text().split()[1])*4096
            self.assertLess(resident,160*1024*1024)
            w.close(Path(t)/'export.npz')
            with np.load(Path(t)/'export.npz') as z:self.assertEqual(z['x'][127].tobytes(),v.tobytes())
if __name__=='__main__':unittest.main(verbosity=2)
