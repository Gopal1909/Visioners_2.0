import unittest
import numpy as np
from construction_match_distributions import distribution_features


class DistributionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(23)
        self.x=rng.random((4,3,2,64,8),dtype=np.float32)
        self.x[...,7]=(self.x[...,7]>.3).astype(np.float32)
        self.mask=rng.random((4,3,2,64))>.25
        self.geo=rng.normal(size=(4,165)).astype(np.float32)

    def test_padding_and_point_permutation_invariance(self):
        a=distribution_features(self.x,self.mask,self.geo)
        corrupted=self.x.copy();corrupted[~self.mask]=np.nan
        np.testing.assert_array_equal(a,distribution_features(corrupted,self.mask,self.geo))
        permutation=np.random.default_rng(2).permutation(64)
        np.testing.assert_allclose(a,distribution_features(self.x[:,:,:,permutation],self.mask[:,:,:,permutation],self.geo),atol=2e-7)
        np.testing.assert_array_equal(a[:,:165],self.geo)

    def test_empty_support_is_finite_and_not_a_perfect_match(self):
        result=distribution_features(self.x,np.zeros_like(self.mask),self.geo)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:,165:],0.)

    def test_distance_quantiles_and_cdf_agree_with_manual_distribution(self):
        x=np.zeros((1,3,2,64,8),np.float32);mask=np.zeros(x.shape[:-1],bool)
        x[:,:,:,:4,6]=np.array([0.,.04,.16,1.]);x[:,:,:,:3,7]=1;mask[:,:,:,:4]=True
        result=distribution_features(x,mask,np.zeros((1,165),np.float32))
        descriptor=result[:,165:273].reshape(1,3,2,18)[0,0,0]
        self.assertAlmostEqual(float(descriptor[0]),4/64)
        self.assertAlmostEqual(float(descriptor[1]),.3,places=6)
        np.testing.assert_allclose(descriptor[3:8],np.quantile([0.,.04,.16,1.],[.1,.25,.5,.75,.9]),atol=1e-7)
        self.assertAlmostEqual(float(descriptor[8]),.25)
        # Strict distance test: 1cm corresponds to normalized .04, exact edge excluded.
        np.testing.assert_allclose(descriptor[9:15],[.25,.5,.5,.75,.75,.75])

    def test_swapping_sides_reverses_signed_contrast_only(self):
        a=distribution_features(self.x,self.mask,self.geo)
        b=distribution_features(self.x[:,:,::-1],self.mask[:,:,::-1],self.geo)
        np.testing.assert_allclose(a[:,273:327],-b[:,273:327],atol=1e-7)
        np.testing.assert_allclose(a[:,327:],b[:,327:],atol=1e-7)


if __name__=='__main__':unittest.main(verbosity=2)
