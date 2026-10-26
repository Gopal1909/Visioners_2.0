import tempfile
import unittest
from pathlib import Path
import numpy as np
from learning_v3_support import bounded_voxels,mapped_transform
from learning_v2_data import fixed_voxel_centroids
from learning_v3_data import transformed


class DiskSupportTests(unittest.TestCase):
    def test_light_geometry_kernels_are_ast_identical_to_frozen_functions(self):
        import ast
        root=Path(__file__).resolve().parent
        light={n.name:n for n in ast.parse((root/'learning_v3_kernels.py').read_text()).body if isinstance(n,ast.FunctionDef)}
        for filename,names in [('learning_v2_data.py',['raw_moments','bounded_nearest','coverage_patches','features']),('multipair_data.py',['discover_pair','cell_coordinates','split_mask']),('c2c_robustness.py',['apply_rigid_transformation','apply_point_sparsity'])]:
            original={n.name:n for n in ast.parse((root/filename).read_text()).body if isinstance(n,ast.FunctionDef)}
            for name in names:self.assertEqual(ast.dump(light[name],include_attributes=False),ast.dump(original[name],include_attributes=False))

    def test_partition_centroids_match_original_bit_for_bit(self):
        rng=np.random.default_rng(8);xyz=rng.normal(size=(13000,3))*3+np.array([40,10,-2])
        for size in (.025,.05,.075):
            np.testing.assert_array_equal(bounded_voxels(xyz,size,max_points=301),fixed_voxel_centroids(xyz,size))

    def test_degenerate_massively_repeated_voxel_matches_original(self):
        xyz=np.tile(np.array([[1.0001,2.0002,3.0003]],np.float64),(10003,1))
        np.testing.assert_array_equal(bounded_voxels(xyz,.075,max_points=100),fixed_voxel_centroids(xyz,.075))

    def test_disk_transform_preserves_retention_noise_stream_and_geometry(self):
        xyz=np.random.default_rng(2).normal(size=(120017,3)).astype(np.float32);center=np.array([.2,.3,.1])
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace/'results') as d:
            self.assertTrue(Path(d).resolve().is_relative_to(workspace))
            for params in [dict(translation_m=.08,rotation_deg=1.2,retention=.7,noise_sigma_m=.015),dict(translation_m=0.,rotation_deg=0.,retention=1.,noise_sigma_m=.03)]:
                a,ids,maximum=transformed(xyz,params,31,center)
                b,other,measured=mapped_transform(xyz,params,31,center,d)
                np.testing.assert_array_equal(ids,other);np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
                self.assertAlmostEqual(maximum,measured,places=12)
                np.testing.assert_array_equal(fixed_voxel_centroids(a,.05),bounded_voxels(b,.05,max_points=10000))
                del b,other


if __name__=='__main__':unittest.main(verbosity=2)
