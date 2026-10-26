"""Regression for a visual audit finding: unordered retention levels made a triangle."""
import unittest
import numpy as np
from analyze_study import ordered_curve


class PlotTests(unittest.TestCase):
    def test_retention_clean_and_stress_pairing(self):
        x,y=ordered_curve([100,10,25],[.4,.2,.3])
        np.testing.assert_array_equal(x,[10,25,100])
        np.testing.assert_allclose(y,[.2,.3,.4])
        self.assertTrue((np.diff(x)>0).all())

    def test_noise_order_preserves_measurements(self):
        x,y=ordered_curve([0,7.5,2],[.41,.43,.36])
        np.testing.assert_allclose(x,[0,2,7.5])
        np.testing.assert_allclose(y,[.41,.36,.43])


if __name__=='__main__':unittest.main(verbosity=2)
