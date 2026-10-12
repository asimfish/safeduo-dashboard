import unittest
import numpy as np
import torch
from native_common_beam_v1 import pack_friction


class FrictionPacketContract(unittest.TestCase):
    def test_only_valid_native_slots_are_observed_with_original_indices(self):
        force=torch.full((8,3),float('nan'));point=force.clone()
        force[1:3]=torch.tensor([[1.,2.,3.],[4.,5.,6.]]);point[1:3]=0
        packet=pack_friction([force,point,torch.tensor([[2,0]]),torch.tensor([[1,4294967295]])],8)
        np.testing.assert_array_equal(packet['raw_indices'],[1,2])
        np.testing.assert_array_equal(packet['forces'],[[1,2,3],[4,5,6]])

    def test_capacity_boundary_rejected(self):
        with self.assertRaises(AssertionError):pack_friction([torch.zeros(8,3),torch.zeros(8,3),torch.tensor([[8]]),torch.tensor([[0]])],8)

    def test_out_of_range_native_packet_rejected(self):
        with self.assertRaises(AssertionError):pack_friction([torch.zeros(8,3),torch.zeros(8,3),torch.tensor([[2]]),torch.tensor([[7]])],8)


if __name__=='__main__':unittest.main()
