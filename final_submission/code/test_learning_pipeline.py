"""Learning input invariants and trainability, without reading final test labels."""
import unittest
import numpy as np
import torch
from scipy.spatial import cKDTree
from siamese_pointnet import SiamesePointNet,paired_features,setup


class LearningTests(unittest.TestCase):
    def setUp(self):
        setup(42);self.model=SiamesePointNet().eval()

    def test_shared_encoder_permutation_and_padding_invariance(self):
        p=torch.randn(3,12,3);m=torch.ones(3,12,dtype=torch.bool)
        q=torch.randn(3,12,3);aux=torch.rand(3,3)
        with torch.no_grad():
            a=self.model(p,m,q,m,aux)
            order=torch.randperm(12)
            b=self.model(p[:,order],m[:,order],q[:,order],m[:,order],aux)
            ppad=torch.cat([p,torch.randn(3,5,3)],dim=1)
            qpad=torch.cat([q,torch.randn(3,5,3)],dim=1)
            mpad=torch.cat([m,torch.zeros(3,5,dtype=torch.bool)],dim=1)
            c=self.model(ppad,mpad,qpad,mpad,aux)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(a,c,atol=1e-6,rtol=1e-6)

    def test_empty_target_finite_and_no_ground_truth_features(self):
        p=np.random.default_rng(1).normal(size=(30,3))*.1
        f=paired_features(cKDTree(p),cKDTree(np.empty((0,3))),np.zeros((2,3)),k=16)
        self.assertEqual(set(f),{'p0','m0','p1','m1','aux'})
        self.assertFalse(f['m1'].any())
        with torch.no_grad():
            pred=self.model(*[torch.from_numpy(f[k]) for k in ('p0','m0','p1','m1','aux')])
        self.assertTrue(torch.isfinite(pred).all())

    def test_paired_feature_common_translation(self):
        rng=np.random.default_rng(3);p=rng.uniform(-.2,.2,(50,3));q=p+[0,0,.08]
        c=p[:4];shift=np.array([31.273,-4.256,3.111])
        a=paired_features(cKDTree(p),cKDTree(q),c,k=16)
        b=paired_features(cKDTree(p+shift),cKDTree(q+shift),c+shift,k=16)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-6)

    def test_supervised_loss_decreases_on_synthetic_offsets(self):
        # Only checks that encoder/head/optimizer can learn; not a dataset metric.
        model=SiamesePointNet().train();optim=torch.optim.Adam(model.parameters(),lr=.005)
        p=torch.randn(24,16,3)*.1;target=torch.cat([torch.zeros(12),torch.ones(12)])
        q=p.clone();q[12:,:,2]+=.4
        mask=torch.ones(24,16,dtype=torch.bool)
        aux=torch.zeros(24,3);aux[:,:2]=1;aux[12:,2]=.7
        losses=[]
        for _ in range(50):
            optim.zero_grad();pred=model(p,mask,q,mask,aux)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,target)
            loss.backward();optim.step();losses.append(float(loss.detach()))
        self.assertLess(np.mean(losses[-5:]),np.mean(losses[:5])*.5)


if __name__=='__main__':unittest.main(verbosity=2)
