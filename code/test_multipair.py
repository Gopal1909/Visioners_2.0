"""Meaningful safety/integrity tests for six-pair integration and evaluation."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from plyfile import PlyData,PlyElement
from multipair_data import discover_pair,read_cloud,cell_coordinates,split_mask,sample_indices
from multipair_metrics import binary_scores,balanced_training_weights,directional_three_class_scores
from multipair_features import voxel_centroids,extract_features,assemble,paired_features
from scipy.spatial import cKDTree
from multipair_nested import choose_inner_cells
import pandas as pd

TEST_TEMP=Path(__file__).resolve().parents[1]/'results/test_tmp'
TEST_TEMP.mkdir(parents=True,exist_ok=True)


class DataTests(unittest.TestCase):
    def test_pooled_out_of_fold_confusions_preserve_unknown_penalties(self):
        from multipair_analysis import spatial_support_summary
        from multipair_metrics import pair_scores
        labels=[np.array([0,0,1,1]),np.array([0,1,1])]
        prediction=[np.array([0,-1,0,1]),np.array([1,1,-1])]
        row=dict(method='fixed',**pair_scores(labels,prediction))
        result=spatial_support_summary(pd.DataFrame([row])).iloc[0]
        independent=directional_three_class_scores(labels,prediction)
        self.assertAlmostEqual(result.pooled_out_of_fold_three_class_macro_f1,independent['pooled_three_class_macro_f1'])
        self.assertEqual(result.positive_support_directions,2)

    def test_stress_curve_clean_reference_uses_identical_original_ids(self):
        from unittest.mock import patch
        from multipair_analysis import stress_clean_reference
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            root=Path(temp);base=root/'evaluation/primary/pair_01'
            clean=base/'clean_seed42';stress=base/'translation_0.05_seed42'
            clean.mkdir(parents=True);stress.mkdir()
            # Full clean scores are perfect at the first two rows, but the
            # stress study requests only the two missed changed rows.
            values={}
            for scan in (0,1):
                values[f'indices{scan}']=np.array([10,20,30,40])
                values[f'labels{scan}']=np.array([0,0,1,1],np.int8)
                values[f'method_prediction{scan}']=np.zeros(4,np.int8)
            np.savez(clean/'predictions.npz',**values)
            np.savez(stress/'geometry_measurements.npz',indices0=[30,40],indices1=[30,40])
            with patch('multipair_analysis.PAIRS',('01',)):
                row=stress_clean_reference(root).iloc[0]
            self.assertEqual(row.scan0_cores,2)
            self.assertEqual(row.scan1_positives,2)
            self.assertEqual(row.selection_binary_macro_f1,0.)

    def test_nested_discovery_rejects_ambiguous_clouds(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            self.assertEqual(Path(temp).resolve().parent,TEST_TEMP.resolve())
            root=Path(temp);nested=root/'pair_02/pair_02';nested.mkdir(parents=True)
            for i in (0,1):(nested/f'pointCloud{i}.ply').touch()
            self.assertEqual(discover_pair(root,'02')[0],nested/'pointCloud0.ply')
            (nested.parent/'pointCloud0.ply').touch()
            with self.assertRaises(ValueError):discover_pair(root,'02')

    def test_labels_validated_before_integer_cast(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            self.assertEqual(Path(temp).resolve().parent,TEST_TEMP.resolve())
            path=Path(temp)/'bad.ply'
            a=np.zeros(3,dtype=[(k,'f4') for k in ('x','y','z','intensity','cd_type')]);a['cd_type'][1]=.5
            PlyData([PlyElement.describe(a,'params')]).write(str(path))
            with self.assertRaises(ValueError):read_cloud(path)

    def test_cell_guards_and_shared_split_assignment(self):
        protocol=dict(origin_xy_m=[0,0],cell_size_m=8.,core_boundary_guard_m=2.5,
            cells=dict(train=[[0,0]],validation=[[1,0]],test=[[2,0]]))
        xyz=np.array([[2.5,3,0],[5.5,4,0],[5.51,4,0],[10.5,3,0],[18.5,3,0]])
        masks=[split_mask(xyz,protocol,s) for s in ('train','validation','test')]
        self.assertTrue(np.all(np.sum(masks,axis=0)<=1))
        self.assertFalse(np.any(np.array(masks)[:,2]))
        self.assertEqual(np.min(abs(xyz[masks[0],0,None]-xyz[masks[1],0])),5.)
        moved_z=xyz.copy();moved_z[:,2]+=100
        np.testing.assert_array_equal(split_mask(moved_z,protocol,'train'),masks[0])

    def test_uniform_sampling_preserves_original_ids_without_labels(self):
        mask=np.arange(10000)%3==0
        a=sample_indices(mask,500,7);b=sample_indices(mask,500,7)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(np.unique(a)),500);self.assertTrue(mask[a].all())
        with self.assertRaises(ValueError):sample_indices(mask,10000,7)

    def test_abstentions_are_not_unchanged_or_dropped(self):
        y=np.array([0,0,1,1]);p=np.array([-1,0,1,-1])
        s=binary_scores(y,p)
        self.assertEqual(s['accuracy'],.5);self.assertEqual(s['coverage'],.5)
        self.assertEqual(s['unknown_positive'],1);self.assertEqual(s['unknown_negative'],1)
        self.assertAlmostEqual(s['recall'],.5)
        self.assertLess(s['binary_macro_f1'],binary_scores(y,np.array([0,0,1,0]))['binary_macro_f1'])

    def test_balanced_weights_equalize_pairs_and_classes(self):
        y=np.array([0,0,0,1,0,1,1,1]);pairs=np.repeat([1,2],4)
        w=balanced_training_weights(y,pairs)
        for pair in [1,2]:
            for label in [0,1]:self.assertAlmostEqual(float(w[(pairs==pair)&(y==label)].sum()),2.)

    def test_voxel_context_spans_beyond_dense_nearest_points(self):
        q=np.linspace(-.6,.6,121);x,y=np.meshgrid(q,q);p=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        tree=cKDTree(p);voxel=cKDTree(voxel_centroids(p))
        a=extract_features([tree,tree],[voxel,voxel],np.zeros((1,3)),'nearest')
        b=extract_features([tree,tree],[voxel,voxel],np.zeros((1,3)),'voxel')
        self.assertGreater(np.linalg.norm(b['p0'],axis=2).max(),np.linalg.norm(a['p0'],axis=2).max()*3)
        self.assertEqual(set(b),{'p0','m0','p1','m1','aux','geo'})

    def test_empty_patch_geometry_is_finite(self):
        tree=cKDTree(np.zeros((1,3)));empty=cKDTree(np.empty((0,3)))
        f=extract_features([tree,empty],[tree,empty],np.zeros((2,3)),'voxel')
        for v in f.values():self.assertTrue(np.isfinite(v).all())
        self.assertFalse(f['m1'].any())

    def test_streamed_feature_assembly_keeps_pair_scan_and_index_alignment(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            root=Path(temp);self.assertEqual(root.resolve().parent,TEST_TEMP.resolve())
            (root/'features').mkdir()
            for pair in ['01','23']:
                arrays={}
                for scan in (0,1):
                    marker=int(pair)*10+scan
                    for key in ['p0','m0','p1','m1','aux','geo','labels','indices']:
                        arrays[f'train_{key}{scan}']=np.full((2,3),marker,dtype=np.float32) if key in ['p0','p1','aux','geo'] else np.full(2,marker)
                np.savez_compressed(root/f'features/pair_{pair}_voxel.npz',**arrays)
            result=assemble(root,'voxel','train',('01','23'))
            np.testing.assert_array_equal(result['pair'],[1,1,1,1,23,23,23,23])
            np.testing.assert_array_equal(result['scan'],[0,0,1,1,0,0,1,1])
            np.testing.assert_array_equal(result['indices'],[10,10,11,11,230,230,231,231])
            self.assertIsInstance(result['p0'],np.memmap)
            self.assertFalse(result['p0'].flags.writeable)
            for value in result.values():value._mmap.close()

    def test_cpu_pairing_kernel_matches_trained_feature_definition(self):
        from siamese_pointnet import paired_features as legacy
        p=np.random.default_rng(4).normal(0,.1,(300,3));q=p+[0,0,.1]
        trees=[cKDTree(p),cKDTree(q)];centers=p[:17]
        a=paired_features(*trees,centers);b=legacy(*trees,centers)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_three_class_pooling_keeps_unknown_as_error(self):
        y=[np.array([0,1]),np.array([0,1])]
        perfect=directional_three_class_scores(y,y)
        self.assertEqual(perfect['pooled_three_class_macro_f1'],1.)
        partial=directional_three_class_scores(y,[np.array([-1,1]),np.array([0,-1])])
        self.assertEqual(partial['pooled_removed_f1'],1.)
        self.assertEqual(partial['pooled_added_f1'],0.)
        self.assertAlmostEqual(partial['pooled_unchanged_f1'],2/3)

    def test_inner_stratification_cannot_read_outer_test_labels(self):
        rows=[dict(pair=pair,scan=scan,ix=x,iy=0,unchanged=800,changed=200)
            for pair in ['01','02','03','12','13','23'] for scan in [0,1] for x in range(10)]
        stats=pd.DataFrame(rows);cells=[[x,0] for x in range(10)];test=[[8,0],[9,0]]
        a=choose_inner_cells(stats,cells,test)[0]
        stats.loc[stats.ix>=8,['unchanged','changed']]=[1,1000000000]
        b=choose_inner_cells(stats,cells,test)[0]
        self.assertEqual(a,b);self.assertEqual(a['test'],test)
        self.assertFalse(set(map(tuple,a['validation']))&set(map(tuple,test)))

    def test_inadequate_inner_validation_is_rejected(self):
        stats=pd.DataFrame([dict(pair='01',scan=0,ix=x,iy=0,unchanged=10000,changed=0) for x in range(10)])
        with self.assertRaises(ValueError):choose_inner_cells(stats,[[x,0] for x in range(10)],[[8,0],[9,0]])

    def test_deleted_target_cores_are_separate_from_detector_unknowns(self):
        from multipair_evaluate import metric_rows
        y=[np.array([0,1]),np.array([0])];p=[a.copy() for a in y]
        mask=[np.ones(len(a),bool) for a in y]
        rows=metric_rows(y,{'test':p},mask,{},mask,2,[[1,1],[1,1]])
        full=next(row for row in rows if row['support']=='full_requested')
        self.assertEqual(full['detector_unknown_rate'],0.)
        self.assertEqual(full['correct_over_requested'],1.)
        self.assertEqual(full['correct_over_initial_requested_cores'],.75)
        self.assertEqual(full['scan1_requested_gt1'],1)
        self.assertEqual(full['scan1_retained_gt1'],0)

    def test_changed_f1_operating_point_uses_validation_score_with_fixed_tie_break(self):
        from multipair_operating_points import best_changed_threshold
        frame=pd.DataFrame(dict(threshold=[.2,.5,.8],objective=[.9,.7,.6],changed_f1=[.3,.6,.6]))
        self.assertEqual(best_changed_threshold(frame),.8)
        frame['objective']=[0.,1.,0.]
        self.assertEqual(best_changed_threshold(frame),.8)


if __name__=='__main__':unittest.main(verbosity=2)
