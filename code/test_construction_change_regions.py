import unittest
import numpy as np
from construction_change_regions import group_positives


class RegionTests(unittest.TestCase):
    def test_membership_conservation_without_dropping_isolated_or_unknown_points(self):
        xyz=np.array([[0,0,0],[.1,0,0],[.2,0,0],[2,0,0],[3,0,0],[4,0,0]],float)
        prediction=np.array([1,1,1,1,0,-1]);ids=np.arange(6)
        membership,components=group_positives(xyz,prediction,ids,.4,3)
        self.assertEqual(len(components),1);np.testing.assert_array_equal(membership,[0,0,0,-1,-2,-3])
        self.assertEqual(sum(len(x) for x in components)+int((membership==-1).sum()),int((prediction==1).sum()))

    def test_group_identity_is_permutation_and_common_translation_invariant(self):
        xyz=np.array([[0,0,0],[.1,0,0],[.2,0,0],[2,0,0],[2.1,0,0],[2.2,0,0]],float);ids=np.array([4,5,6,1,2,3]);p=np.ones(6,int)
        a,_=group_positives(xyz,p,ids,.4,3);order=np.array([3,0,5,1,4,2])
        b,_=group_positives(xyz[order]+[30,40,5],p[order],ids[order],.4,3)
        np.testing.assert_array_equal(a[order],b)


if __name__=='__main__':unittest.main(verbosity=2)
