import unittest
from clock_contract import control_clock

class ClockContract(unittest.TestCase):
    def test_original_task_has_distinct_native_and_trajectory_grids(self):
        self.assertGreater(abs(.016666-1/60),1e-7)
        self.assertEqual(control_clock(.016666,.016666,1/60),.016666)
    def test_native_clock_changes_are_rejected(self):
        with self.assertRaises(ValueError):control_clock(1/60,.016666,1/60)
        with self.assertRaises(ValueError):control_clock(float('nan'),.016666,1/60)

if __name__=='__main__':unittest.main()
