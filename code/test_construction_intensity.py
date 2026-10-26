import unittest
import numpy as np
from construction_intensity import gather_intensity,shuffled_within_cloud,local_surface


class IntensityTests(unittest.TestCase):
    def test_original_ids_repeated_rows_and_label_independence(self):
        record=np.zeros(5,dtype=[('intensity','f4'),('cd_type','f4')])
        record['intensity']=[0,1,64,128,255];ids=np.array([4,0,2,2])
        expected=np.array([1.,0.,64/255,64/255],np.float32)
        np.testing.assert_array_equal(gather_intensity(record,ids),expected)
        record['cd_type']=np.nan
        np.testing.assert_array_equal(gather_intensity(record,ids),expected)
        with self.assertRaises(AssertionError):gather_intensity(record,np.array([-1]))
        with self.assertRaises(AssertionError):gather_intensity(record,np.array([5]))

    def test_shuffle_preserves_each_cloud_distribution_and_is_deterministic(self):
        values=np.arange(24,dtype=np.float32);pairs=np.repeat([1,2],12);scans=np.tile(np.repeat([0,1],6),2)
        result=shuffled_within_cloud(values,pairs,scans,7)
        np.testing.assert_array_equal(result,shuffled_within_cloud(values,pairs,scans,7))
        self.assertFalse(np.array_equal(values,result))
        for pair in (1,2):
            for scan in (0,1):
                ids=(pairs==pair)&(scans==scan)
                np.testing.assert_array_equal(np.sort(values[ids]),np.sort(result[ids]))

    def test_flat_horizontal_and_vertical_surfaces_and_invalid_padding(self):
        q=np.linspace(-1,1,8);a,b=np.meshgrid(q,q);plane=np.column_stack((a.ravel(),b.ravel(),np.zeros(64)))
        vertical=plane[:,[0,2,1]];xyz=np.stack([plane,vertical,np.zeros_like(plane)])
        mask=np.ones((3,64),bool);mask[2]=False;xyz[2]=np.nan
        result=local_surface(xyz,mask)
        np.testing.assert_array_equal(result['valid'],[True,True,False])
        np.testing.assert_array_equal(result['locally_horizontal_planar'],[True,False,False])
        shifted=local_surface(xyz+np.array([4.,-6.,2.]),mask)
        np.testing.assert_allclose(result['planarity'],shifted['planarity'],atol=1e-12)
        np.testing.assert_allclose(result['absolute_normal_z'],shifted['absolute_normal_z'],atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
