"""Analytical geometry, uncertainty, abstention and support fairness regressions.

Run with the existing dl_env: python -m unittest discover -s code -p test_m3c2_synthetic.py -v
These tests never read dataset files or invoke the C2C/Stage 3 experiment entrypoints.
"""
import copy
from dataclasses import replace
import unittest
import numpy as np
from scipy.spatial import cKDTree
from m3c2_baseline import (Config, Reason, estimate_normals, project_direction,
                           classify, spatial_core_indices, score_pair)


def plane(step=.02):
    x, y = np.meshgrid(np.arange(-1, 1.0001, step), np.arange(-1, 1.0001, step))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.p = plane()
        self.c = np.array([[0., 0., 0.], [.10, .10, 0.], [-.10, .10, 0.]])
        self.cfg = Config()

    def measure(self, p, q, c=None, cfg=None, flip=False):
        c = self.c if c is None else c
        cfg = self.cfg if cfg is None else cfg
        t0, t1 = cKDTree(p), cKDTree(q)
        normals = estimate_normals(t0, c, cfg)
        if flip:
            normals['normals'] *= -1
        raw = project_direction(t0, t1, c, normals, cfg)
        return normals, raw, classify(raw, cfg.registration_error_m, cfg.minimum_effect_m)

    def test_identical_planes(self):
        n, raw, d = self.measure(self.p, self.p)
        self.assertTrue(d['valid'].all())
        np.testing.assert_allclose(raw['signed_distance_m'], 0, atol=1e-12)
        np.testing.assert_array_equal(d['prediction'], 0)
        np.testing.assert_allclose(n['normals'], np.tile([0,0,1], (len(self.c),1)), atol=1e-12)

    def test_known_offset_both_directions_and_label_mapping(self):
        offset = np.array([0., 0., .12])
        _, forward, d0 = self.measure(self.p, self.p+offset)
        _, reverse, d1 = self.measure(self.p+offset, self.p, self.c+offset)
        np.testing.assert_allclose(forward['signed_distance_m'], .12, atol=1e-12)
        np.testing.assert_allclose(reverse['signed_distance_m'], -.12, atol=1e-12)
        self.assertTrue(d0['valid'].all() and d1['valid'].all())
        np.testing.assert_array_equal(d0['prediction'], 1)
        np.testing.assert_array_equal(d1['prediction'], 1)
        score = score_pair([np.ones(3), np.ones(3)],
                           [d0['prediction'][:,1], d1['prediction'][:,1]], [np.ones(3,bool)]*2)
        self.assertGreater(score['combined_f1_removed'], .999)
        self.assertGreater(score['combined_f1_added'], .999)

    def test_tangential_translation_interior(self):
        _, raw, d = self.measure(self.p, self.p+[.173, -.127, 0])
        self.assertTrue(d['valid'].all())
        np.testing.assert_allclose(raw['signed_distance_m'], 0, atol=1e-12)
        np.testing.assert_array_equal(d['prediction'], 0)

    def test_normal_sign_reversal(self):
        q = self.p+[.02, .02, .12]
        _, a, da = self.measure(self.p, q)
        _, b, db = self.measure(self.p, q, flip=True)
        np.testing.assert_allclose(a['signed_distance_m'], -b['signed_distance_m'], atol=1e-12)
        np.testing.assert_allclose(da['lod95_m'], db['lod95_m'], atol=1e-12)
        np.testing.assert_array_equal(da['prediction'], db['prediction'])

    def test_common_translation(self):
        shift = np.array([1234.512, -785.127, 91.227])
        _, a, da = self.measure(self.p, self.p+[0,0,.12])
        _, b, db = self.measure(self.p+shift, self.p+shift+[0,0,.12], self.c+shift)
        np.testing.assert_allclose(a['signed_distance_m'], b['signed_distance_m'], atol=1e-10)
        np.testing.assert_array_equal(da['prediction'], db['prediction'])

    def test_wall_normal_and_oblique_plane(self):
        from scipy.spatial.transform import Rotation
        for rotation in (Rotation.from_euler('y', 90, degrees=True),
                         Rotation.from_euler('xyz', [27, 34, 15], degrees=True)):
            p, c = rotation.apply(self.p), rotation.apply(self.c)
            n = rotation.apply([0,0,1])
            if n[np.argmax(np.abs(n))] < 0:
                n = -n
            normals, raw, d = self.measure(p, p+.12*n, c)
            self.assertTrue(d['valid'].all())
            np.testing.assert_allclose(normals['normals'], np.tile(n, (3,1)), atol=1e-10)
            np.testing.assert_allclose(raw['signed_distance_m'], .12, atol=1e-10)

    def test_controlled_noise_known_offset(self):
        rng = np.random.default_rng(17)
        p, q = self.p.copy(), self.p.copy()+[0,0,.08]
        p[:,2] += rng.normal(0,.003,len(p))
        q[:,2] += rng.normal(0,.003,len(q))
        _, raw, d = self.measure(p, q)
        self.assertTrue(d['valid'].all())
        np.testing.assert_allclose(raw['signed_distance_m'], .08, atol=.003)
        self.assertTrue((d['lod95_m'] > 0).all())
        np.testing.assert_array_equal(d['prediction'], 1)

    def test_population_spread_and_registration_formula(self):
        # Analytical distribution: 20 projected values, mean=.015, population SD=.005.
        p = np.column_stack((np.tile([-.01,.01],10), np.zeros(20), np.tile([.01,.02],10)))
        q = p+[0,0,.08]
        normals = dict(normals=np.array([[0.,0.,1.]]), normal_reason=np.array([0], np.uint8))
        raw = project_direction(cKDTree(p), cKDTree(q), np.zeros((1,3)), normals, self.cfg)
        np.testing.assert_array_equal(raw['count_source'], 20)
        np.testing.assert_allclose(raw['spread_source_m'], .005, atol=1e-12)
        np.testing.assert_allclose(raw['signed_distance_m'], .08, atol=1e-12)
        d = classify(raw, registration_error_m=.01, minimum_effect_m=0)
        np.testing.assert_allclose(d['lod95_m'], 1.96*(np.sqrt(2*.005**2/20)+.01), atol=1e-12)

    def test_effect_floor_and_strict_comparison(self):
        _, raw, _ = self.measure(self.p, self.p+[0,0,.01])
        np.testing.assert_array_equal(classify(raw, 0, .02)['prediction'], 0)
        np.testing.assert_array_equal(classify(raw, 0, .005)['prediction'], 1)
        raw['signed_distance_m'][:] = .02
        np.testing.assert_array_equal(classify(raw, 0, .02)['prediction'], 0)

    def test_exact_cylinder_excludes_radial_and_axial_outliers(self):
        support = np.zeros((12,3))
        target = support+[0,0,.1]
        target = np.vstack((target, np.tile([.06,0,.7], (20,1)),
                            np.tile([0,0,1.01], (20,1))))
        normal = dict(normals=np.array([[0.,0.,1.]]), normal_reason=np.array([0],np.uint8))
        raw = project_direction(cKDTree(support), cKDTree(target), np.zeros((1,3)), normal, self.cfg)
        self.assertEqual(raw['count_target'][0,0],12)
        self.assertEqual(raw['count_target'][0,1],32)
        self.assertAlmostEqual(raw['signed_distance_m'][0,0], .1)

    def test_empty_and_insufficient_source_normals(self):
        for p in (np.empty((0,3)), self.p[:5]):
            n, _, d = self.measure(p, self.p)
            np.testing.assert_array_equal(n['normal_reason'], Reason.NORMAL_INSUFFICIENT)
            np.testing.assert_array_equal(d['prediction'], -1)
            self.assertFalse(d['valid'].any())

    def test_degenerate_duplicate_and_line_normals(self):
        line = np.column_stack((np.linspace(-.4,.4,100), np.zeros(100), np.zeros(100)))
        for p in (np.zeros((20,3)), line):
            n, _, d = self.measure(p, p, c=np.zeros((1,3)))
            np.testing.assert_array_equal(n['normal_reason'], Reason.NORMAL_DEGENERATE)
            np.testing.assert_array_equal(d['prediction'], -1)

    def test_missing_target_and_insufficient_projection(self):
        for q in (np.empty((0,3)), self.p[:5], self.p+[0,0,1.5]):
            _, raw, d = self.measure(self.p, q)
            self.assertTrue((raw['reason'] & Reason.TARGET_SUPPORT).all())
            self.assertFalse(d['valid'].any())
            np.testing.assert_array_equal(d['prediction'], -1)
            self.assertTrue(np.isnan(d['lod95_m']).all())

    def test_invalid_uncertainty_is_unknown(self):
        _, raw, _ = self.measure(self.p, self.p)
        for reg in (np.nan, np.inf, -.01):
            d = classify(raw, reg)
            np.testing.assert_array_equal(d['reason'], Reason.INVALID_UNCERTAINTY)
            np.testing.assert_array_equal(d['prediction'], -1)
        damaged = copy.deepcopy(raw)
        damaged['spread_target_m'][:] = np.nan
        d = classify(damaged)
        np.testing.assert_array_equal(d['prediction'], -1)

    def test_invalid_normal_and_resource_guard(self):
        n = dict(normals=np.array([[np.nan,0.,1.]]), normal_reason=np.array([0],np.uint8))
        raw = project_direction(cKDTree(self.p), cKDTree(self.p), self.c[:1], n)
        np.testing.assert_array_equal(classify(raw)['reason'], Reason.INVALID_NORMAL)
        n, raw, d = self.measure(self.p, self.p, cfg=replace(self.cfg, max_candidates=10))
        np.testing.assert_array_equal(n['normal_reason'], Reason.RESOURCE_LIMIT)
        np.testing.assert_array_equal(d['prediction'], -1)

    def test_single_scale_cannot_assert_cross_scale_stability(self):
        n, _, d = self.measure(self.p, self.p, cfg=replace(self.cfg, normal_radii_m=(.2,)))
        np.testing.assert_array_equal(n['normal_reason'], Reason.NORMAL_UNSTABLE)
        np.testing.assert_array_equal(d['prediction'], -1)

    def test_spatial_original_indices_reproducible(self):
        ids, meta = spatial_core_indices(self.p, 100, 42, tile_m=.2)
        again, _ = spatial_core_indices(self.p, 100, 42, tile_m=.2)
        np.testing.assert_array_equal(ids, again)
        self.assertEqual(len(np.unique(ids)),100)
        self.assertGreater(meta['sampled_tiles'],50)
        self.assertGreater(np.ptp(self.p[ids,0]),1.5)
        self.assertGreater(np.ptp(self.p[ids,1]),1.5)

    def test_unknowns_excluded_and_conservative_score_counts_errors(self):
        gt = [np.array([0,1,1]), np.array([0,1,0])]
        pred = [np.array([0,1,-1]), np.array([0,-1,-1])]
        masks = [p >= 0 for p in pred]
        score = score_pair(gt, pred, masks)
        self.assertEqual(score['coverage'], .5)
        self.assertEqual(score['conservative_accuracy_lower_bound'], .5)
        self.assertEqual(score['scan0_TP'],1)
        self.assertEqual(score['scan0_FN'],0)
        self.assertEqual(score['scan0_gt1_coverage'],.5)

    def test_c2c_mask_exclusion_is_not_detector_abstention(self):
        gt = [np.array([0,1]), np.array([0,1])]
        mask = [np.array([True,False])]*2
        score = score_pair(gt, gt, mask)
        self.assertEqual(score['coverage'],.5)
        self.assertEqual(score['support_excluded_rate'],.5)
        self.assertEqual(score['unknown_rate'],0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
