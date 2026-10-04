import numpy as np

from coverage_metrics import ee_occupancy,joint_occupancy,measured_coverage


def test_outside_values_do_not_fill_edge_bins():
    limits=np.broadcast_to(np.array([0.,1.]),(2,26,2))
    q=np.full((1,2,26),2.);q[0,0,0]=.35
    r=joint_occupancy(q,limits)
    assert r['counts'][0][3]==1 and r['outside_soft_limit_samples']==51
    assert sum(map(sum,r['counts']))==1


def test_large_initial_spread_is_separate_from_window_motion():
    limits=np.broadcast_to(np.array([0.,1.]),(2,26,2))
    q0=np.array([[.05]*26,[.95]*26]);q=np.repeat(q0[None],3,axis=0)
    ee0=np.zeros((2,4,3));ee=np.repeat(ee0[None],3,axis=0)
    r=measured_coverage(q0,q,ee0,ee,limits)
    assert r['initial_joint']['visited_bins_per_joint']==[2]*26
    assert r['visited_joint']['visited_bins_per_joint']==[2]*26
    assert r['mean_within_window_joint_range']==0 and r['mean_joint_path_rad']==0


def test_ee_cells_are_visited_points_not_bounding_box_volume():
    ee=np.zeros((2,1,4,3));ee[1,...]=1.
    r=ee_occupancy(ee)
    assert r['visited_voxels_per_arm']==[2]*4
