"""Admission/metadata/capacity contracts against an independent scalar oracle."""
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import unittest
import torch

LEGACY='--legacy' in sys.argv
if LEGACY:
    sys.argv.remove('--legacy')
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_random_space_20261004/dependencies'))
    from critical_rows import merge_critical_rows
    def merge(full,selected,cls,ids,capacity=2048):
        return merge_critical_rows(full,selected,cls,ids,band=.010,capacity=capacity),{}
else:
    from predictive_rows import merge_predictive_rows as merge

@dataclass
class Bundle:
    active_pairs:object=None
    active_mask:object=None
    active_idx:object=None
    active_dmin:object=None
    viol_exempt:object=None
    dists:object=None
    closing:object=None
    full_dmin:object=None
    full_viol_exempt:object=None

def fixture(d,c,cls,chosen=()):
    full=Bundle(dists=torch.tensor([d],dtype=torch.float32),closing=torch.tensor([c],dtype=torch.float32),
                full_dmin=torch.full((1,len(d)),.020),full_viol_exempt=torch.zeros((1,len(d)),dtype=torch.bool))
    selected=Bundle(active_idx=torch.tensor([list(chosen)+[-1]],dtype=torch.long),
                    active_mask=torch.tensor([[True]*len(chosen)+[False]]))
    return full,selected,torch.tensor(cls,dtype=torch.float32),torch.arange(len(d),dtype=torch.float32)

def admitted(out):return out.active_idx[out.active_mask].tolist()

class Contracts(unittest.TestCase):
    def test_closing_far_row_must_be_admitted_before_instant_band(self):
        f,s,c,i=fixture([.080,.070,.090],[.5,-.6,0],[0,1,2],chosen=[2])
        out,_=merge(f,s,c,i)
        self.assertEqual(admitted(out),[0,2])

    def test_class_horizons_current_exemption_and_legacy_rows(self):
        f,s,c,i=fixture([.150,.150,.150,.090],[.35,.35,.35,0],[0,1,2,0],chosen=[3])
        f.full_dmin[0,2]=.001;f.full_viol_exempt[0,2]=True
        out,_=merge(f,s,c,i)
        self.assertEqual(admitted(out),[1,3])
        f.closing[0,2]=.5
        out,_=merge(f,s,c,i)
        self.assertEqual(admitted(out),[1,2,3]);self.assertAlmostEqual(float(out.active_dmin[0,1]),.001,places=7)
        self.assertTrue(bool(out.viol_exempt[0,1]))

    def test_capacity_excess_is_failure_not_truncation(self):
        f,s,c,i=fixture([.080,.080],[.5,.5],[0,0])
        with self.assertRaisesRegex(ValueError,'capacity'):
            merge(f,s,c,i,capacity=1)

    def test_random_scalar_set_oracle_metadata_and_opening(self):
        rng=random.Random(7819307)
        for _ in range(250):
            n=rng.randint(2,160);ds=[rng.uniform(-.1,1.5) for _ in range(n)];cs=[rng.uniform(-2,2) for _ in range(n)]
            cls=[rng.randrange(3) for _ in range(n)];chosen=rng.sample(range(n),rng.randrange(min(12,n)))
            f,s,c,i=fixture(ds,cs,cls,chosen);f.full_dmin[0]=torch.tensor([rng.choice([.001,.005,.015,.02]) for _ in range(n)])
            f.full_viol_exempt[0]=torch.tensor([rng.choice([True,False]) for _ in range(n)])
            out,stats=merge(f,s,c,i)
            expected=[j for j in range(n) if j in chosen or float(f.dists[0,j])<=float(f.full_dmin[0,j])+.010 or
                      float(f.dists[0,j])-[.16,.4,.3][cls[j]]*max(float(f.closing[0,j]),0)<=float(f.full_dmin[0,j])+.010]
            self.assertEqual(admitted(out),expected)
            for k,j in enumerate(expected):
                self.assertEqual(float(out.active_pairs[0,k,0]),float(f.dists[0,j]));self.assertEqual(float(out.active_pairs[0,k,1]),float(f.closing[0,j]))
                self.assertEqual(float(out.active_pairs[0,k,3]),j);self.assertEqual(float(out.active_dmin[0,k]),float(f.full_dmin[0,j]))
                self.assertEqual(bool(out.viol_exempt[0,k]),bool(f.full_viol_exempt[0,j]))
            self.assertEqual(out.active_mask.sum().item(),len(expected))

    def test_missing_or_nonfinite_full_measurements_fail(self):
        f,s,c,i=fixture([.08],[.5],[0]);f.closing[0,0]=math.nan
        with self.assertRaises(ValueError):merge(f,s,c,i)
        f.closing=None
        with self.assertRaises(ValueError):merge(f,s,c,i)

    def test_heterogeneous_batch_padding_cannot_erase_final_valid_row(self):
        f,s,c,i=fixture([.080,.080,.080],[.5,.5,.5],[0,0,0])
        f.dists=f.dists.repeat(3,1);f.closing=torch.tensor([[.5,.5,.5],[0,0,.5],[0,0,0.]])
        f.full_dmin=f.full_dmin.repeat(3,1);f.full_viol_exempt=f.full_viol_exempt.repeat(3,1)
        s.active_idx=s.active_idx.repeat(3,1);s.active_mask=s.active_mask.repeat(3,1)
        out,stats=merge(f,s,c,i)
        self.assertEqual([out.active_idx[e,out.active_mask[e]].tolist() for e in range(3)],[[0,1,2],[2],[]])
        self.assertTrue((out.active_idx[~out.active_mask]==-1).all())
        if not LEGACY:self.assertEqual(stats['predicted_missing'].tolist(),[0,0,0])

if __name__=='__main__':unittest.main()
