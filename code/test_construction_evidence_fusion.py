import unittest
import numpy as np
from construction_evidence_fusion import blend


class FusionTests(unittest.TestCase):
    def test_endpoints_and_interpolation(self):
        a=np.array([.1,.5,.9]);b=np.array([.9,.6,.2])
        np.testing.assert_array_equal(blend(a,b,0),a)
        np.testing.assert_array_equal(blend(a,b,1),b)
        np.testing.assert_allclose(blend(a,b,.25),[.3,.525,.725])
        self.assertTrue(((blend(a,b,.25)>=np.minimum(a,b))&(blend(a,b,.25)<=np.maximum(a,b))).all())

    def test_invalid_scores_weights_and_alignment_rejected(self):
        for a,b,w in [([.1],[.2],1.1),([.1],[np.nan],.5),([-.1],[.2],.5),([.1,.2],[.3],.5)]:
            with self.assertRaises(AssertionError):blend(a,b,w)


if __name__=='__main__':unittest.main(verbosity=2)
