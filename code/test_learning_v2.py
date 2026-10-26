"""Nontrivial geometry, masking and sampling checks for the model upgrade."""
import unittest
import numpy as np
import torch
from scipy.spatial import cKDTree
from learning_v2_data import features, contexts_for, transform_training, bounded_nearest, SUPPORT_BOUND
from learning_v2_model import MetricFusion, hard_order, batch, batch_arrays


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(14)

    def test_light_legacy_network_and_scores_are_identical_to_frozen_kernels(self):
        import ast
        from pathlib import Path
        root=Path(__file__).resolve().parent
        for original,light,names in [('siamese_pointnet.py','learning_v2_legacy.py',['SiamesePointNet','tensor_batch','predict']),
            ('multipair_evaluate.py','learning_v2_scores.py',['metric_rows'])]:
            a={n.name:n for n in ast.parse((root/original).read_text()).body if hasattr(n,'name')}
            b={n.name:n for n in ast.parse((root/light).read_text()).body if hasattr(n,'name')}
            for name in names:self.assertEqual(ast.dump(a[name],include_attributes=False),ast.dump(b[name],include_attributes=False))

    def test_reusable_host_buffers_match_original_mixed_view_gather(self):
        banks=[]
        for view in range(3):
            banks.append(dict(x=self.rng.normal(size=(6,3,2,8,8)).astype(np.float32),mask=self.rng.random((6,3,2,8))>.2,
                geo=self.rng.normal(size=(6,165)).astype(np.float32),available=np.array([1,0,1,1,0,1],bool)))
        for ids,views in [(np.array([5,1,3,2]),np.array([0,1,2,1])),(np.array([0,4,2,5]),np.array([2,2,0,1]))]:
            expected=batch_arrays(banks,ids,views,reuse=False)
            actual=batch_arrays(banks,ids,views,reuse=True)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_checkpoint_restart_preserves_adamw_and_dropout_trajectory(self):
        import io
        torch.manual_seed(91)
        x=torch.randn(4,3,2,8,8);mask=torch.ones(4,3,2,8,dtype=torch.bool)
        geo=torch.randn(4,165);target=torch.tensor([0.,1.,0.,1.])
        torch.manual_seed(18);model=MetricFusion().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        checkpoint=None
        for step in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,mask,geo),target)
            loss.backward();optimizer.step()
            if step==2:
                stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),stream)
                checkpoint=stream.getvalue()
        expected={k:v.detach().clone() for k,v in model.state_dict().items()}
        resumed=MetricFusion().train();other=torch.optim.AdamW(resumed.parameters(),lr=.001)
        state=torch.load(io.BytesIO(checkpoint),weights_only=False);resumed.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        for step in range(3):
            other.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(resumed(x,mask,geo),target)
            loss.backward();other.step()
        for key,value in resumed.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)

    def test_saturated_neighborhood_retains_wide_radius_context(self):
        grid=np.arange(-10,11)*.075
        x,y=np.meshgrid(grid,grid);xyz=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        trees=[cKDTree(xyz),cKDTree(xyz)]
        f=features(trees,[[*trees] for _ in range(3)],np.zeros((1,3)))
        selected=f['x'][0,2,0,:,:3]*.8
        self.assertEqual(int(f['mask'][0,2,0].sum()),64)
        self.assertGreater(float(np.linalg.norm(selected,axis=1).max()),.75)
        self.assertLessEqual(float(np.linalg.norm(selected,axis=1).max()),.800001)
        self.assertEqual(len(np.unique(selected,axis=0)),64)

    def test_feature_translation_equivariance_and_no_gt_argument(self):
        xyz=self.rng.normal(size=(100,3))*.2
        trees=[cKDTree(xyz),cKDTree(xyz+[0,0,.04])];c=xyz[:4]
        # A fixed context grid avoids voxel-origin roundoff at an exact boundary.
        contexts=[[trees[0],trees[1]] for _ in range(3)]
        a=features(trees,contexts,c)
        shifted=[cKDTree(t.data+[10,-4,2]) for t in trees]
        b=features(shifted,[[shifted[0],shifted[1]] for _ in range(3)],c+[10,-4,2])
        np.testing.assert_array_equal(a['mask'],b['mask'])
        np.testing.assert_allclose(a['x'],b['x'],rtol=0,atol=2e-6)
        np.testing.assert_allclose(a['geo'],b['geo'],rtol=0,atol=2e-6)

    def test_remote_geometry_cannot_change_bounded_features(self):
        a=self.rng.normal(size=(60,3))*.03;b=a+[0,0,.02]
        trees=[cKDTree(a),cKDTree(b)];centers=np.zeros((2,3))
        x=features(trees,contexts_for(trees),centers)
        remote=np.array([[-20,-20,-20],[30,0,0]],float)
        extra=[cKDTree(np.r_[a,remote]),cKDTree(np.r_[b,remote*2])]
        y=features(extra,contexts_for(extra),centers)
        for key in x:np.testing.assert_allclose(x[key],y[key],atol=1e-7)
        self.assertLess(SUPPORT_BOUND,1.2)

    def test_empty_target_has_finite_features_and_explicit_no_match(self):
        trees=[cKDTree(np.zeros((10,3))),cKDTree(np.empty((0,3)))]
        f=features(trees,[[*trees] for _ in range(3)],np.zeros((2,3)))
        self.assertTrue(np.isfinite(f['x']).all() and np.isfinite(f['geo']).all())
        self.assertFalse(f['mask'][:,:,1].any())
        self.assertTrue((f['x'][:,:,:, :,7]==0).all())

    def test_full_scan_transform_keeps_original_id_coordinate_relationship(self):
        xyz=self.rng.normal(size=(100,3)).astype(np.float32)
        p,ids,maximum=transform_training(xyz,'medium',6,[0,0,0])
        q,other,m=transform_training(xyz,'medium',6,[0,0,0])
        np.testing.assert_array_equal(ids,other);np.testing.assert_array_equal(p,q)
        self.assertEqual(len(ids),40);self.assertTrue((np.diff(ids)>0).all())
        self.assertAlmostEqual(maximum,np.linalg.norm(p.astype(float)-xyz[ids],axis=1).max())
        self.assertEqual(maximum,m)

    def test_hard_sampling_preserves_pair_class_counts_and_emphasizes_hard_rows(self):
        y=np.tile([0,0,0,1,1,1],20);pair=np.repeat([1,2],60)
        loss=np.ones(120);loss[2::3]=5
        counts=np.zeros(120,int)
        for seed in range(80):
            ids=hard_order(y,pair,loss,np.random.default_rng(seed));np.add.at(counts,ids,1)
            for code in (1,2):
                for label in (0,1):self.assertEqual(int(((pair[ids]==code)&(y[ids]==label)).sum()),30)
        self.assertGreater(counts[2::3].mean(),counts[0::3].mean()*1.25)

    def test_deleted_training_query_falls_back_to_clean_bank(self):
        def bank(value,available):
            return dict(x=np.full((2,3,2,64,8),value,np.float32),mask=np.ones((2,3,2,64),bool),
                geo=np.full((2,165),value,np.float32),available=np.array(available,bool))
        clean=bank(1,[1,1]);aug=bank(2,[0,1]);medium=bank(3,[1,1])
        x,mask,geo=batch([clean,aug,medium],np.array([0,1]),'cpu',views=[1,1])
        self.assertEqual(float(x[0,0,0,0,0]),1.);self.assertEqual(float(geo[1,0]),2.)

    def test_network_padding_and_permutation_invariance_and_gradients(self):
        torch.manual_seed(4);model=MetricFusion().eval()
        x=torch.randn(2,3,2,64,8);mask=torch.rand(2,3,2,64)>.3;geo=torch.randn(2,165)
        a=model(x,mask,geo)
        altered=x.clone();altered[~mask]=500
        order=torch.randperm(64);b=model(altered[:,:,:,order],mask[:,:,:,order],geo)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        empty=model(x,torch.zeros_like(mask),geo);self.assertTrue(torch.isfinite(empty).all())


if __name__=='__main__':unittest.main(verbosity=2)
