import unittest
import numpy as np
import torch
from learning_v3_model import HierarchicalFusion,anchors,gather
from learning_v3_model import loss_values
from learning_v3_data import balanced_ids,transformed,stream_npz


class HierarchyTests(unittest.TestCase):
    def test_supplementary_three_class_scores_match_independent_point_confusion(self):
        from learning_v3_metrics import supplementary
        from multipair_metrics import pair_scores,directional_three_class_scores
        rng=np.random.default_rng(80)
        for y in [[np.zeros(100,dtype=int),np.zeros(120,dtype=int)],[rng.integers(0,2,100),rng.integers(0,2,120)]]:
            p=[rng.integers(-1,2,len(a)) for a in y]
            expected=directional_three_class_scores(y,p);actual=supplementary(pair_scores(y,p))
            for key in expected:self.assertAlmostEqual(actual[key],expected[key],places=14)

    def test_bounded_atomic_feature_writer_preserves_dtypes_and_values(self):
        import tempfile
        from pathlib import Path
        arrays={'x':np.random.default_rng(7).normal(size=(30,3,2,64,8)).astype(np.float32),'mask':np.arange(30)%2==0,'ids':np.arange(30,dtype=np.int64)}
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace/'results') as d:
            self.assertTrue(Path(d).resolve().is_relative_to(workspace))
            p=Path(d)/'features.npz';stream_npz(p,arrays)
            self.assertFalse(p.with_name('features_writing.npz').exists())
            with np.load(p) as z:
                for k,a in arrays.items():np.testing.assert_array_equal(z[k],a);self.assertEqual(z[k].dtype,a.dtype)

    def test_fresh_training_sampler_cannot_read_excluded_labels(self):
        rng=np.random.default_rng(3);labels=np.r_[np.zeros(80),np.ones(80)].astype(np.int8)
        eligible=np.ones(160,bool);eligible[::3]=False
        a=balanced_ids(labels,eligible,40,np.random.default_rng(42))
        altered=labels.copy();altered[~eligible]=1-altered[~eligible]
        b=balanced_ids(altered,eligible,40,np.random.default_rng(42))
        np.testing.assert_array_equal(a,b);self.assertTrue(eligible[a].all());self.assertEqual(int(labels[a].sum()),20)
        fresh=balanced_ids(labels,eligible,40,np.random.default_rng(43));self.assertFalse(np.array_equal(a,fresh))

    def test_full_scan_thinning_retains_original_coordinate_relationship(self):
        xyz=np.random.default_rng(9).normal(size=(1000,3)).astype(np.float32)
        p,ids,maximum=transformed(xyz,dict(translation_m=0.,rotation_deg=0.,retention=.4,noise_sigma_m=0.),42,np.zeros(3))
        np.testing.assert_array_equal(p,xyz[ids]);self.assertEqual(len(ids),400);self.assertEqual(maximum,0.)
        self.assertTrue((np.diff(ids)>0).all())

    def test_focal_loss_reduces_easy_example_weight_and_has_finite_gradients(self):
        logits=torch.tensor([-8.,0.,8.],requires_grad=True);target=torch.ones(3)
        plain=loss_values(logits,target);focal=loss_values(logits,target,True)
        self.assertGreater(float(focal[0]/plain[0]),float(focal[-1]/plain[-1]))
        focal.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_masked_sampling_uses_distinct_valid_points(self):
        torch.manual_seed(1);xyz=torch.randn(2,64,3);mask=torch.zeros(2,64,dtype=torch.bool);mask[0,:7]=True
        ids,valid=anchors(xyz,mask)
        self.assertEqual(int(valid[0].sum()),7);self.assertEqual(int(valid[1].sum()),0)
        self.assertEqual(len(torch.unique(ids[0,valid[0]])),7)
        self.assertTrue(mask[0,ids[0,valid[0]]].all())

    def test_local_cross_scan_model_masks_permutation_and_finite_gradients(self):
        torch.manual_seed(6);torch.set_num_threads(2);model=HierarchicalFusion().eval()
        x=torch.randn(2,3,2,64,8)*.3;mask=torch.rand(2,3,2,64)>.3;geo=torch.randn(2,165)
        a=model(x,mask,geo);altered=x.clone();altered[~mask]=700;order=torch.randperm(64)
        z=model(altered[:,:,:,order],mask[:,:,:,order],geo)
        torch.testing.assert_close(a,z,rtol=1e-4,atol=1e-5)
        a.sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(torch.isfinite(model(x,torch.zeros_like(mask),geo)).all())


if __name__=='__main__':unittest.main(verbosity=2)
