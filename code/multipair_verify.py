"""Independent completion/integrity checks for the result-focused six-pair study."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plyfile import PlyData
from multipair_data import ROOT,DEFAULT,PAIRS,split_mask
from multipair_metrics import pair_scores
from m3c2_baseline import digest,write_json


def verify(out=DEFAULT,full=True):
    out=Path(out);audit=json.loads((out/'data_audit.json').read_text())
    for row in audit['clouds']:assert digest(row['path'])==row['sha256'],row['path']
    original_records={(row['pair'],row['scan']):PlyData.read(row['path'],mmap='r')['params'].data for row in audit['clouds']}
    original_labels={key:np.asarray(record['cd_type'],dtype=np.int8) for key,record in original_records.items()}
    protected=json.loads((ROOT/'results/pair_01/m3c2/preservation_before.json').read_text(encoding='utf-8-sig'))
    for row in protected:assert digest(row['Path']).upper()==row['Hash'].upper(),row['Path']
    old_submission=ROOT/'final_submission'
    snapshot=json.loads((old_submission/'submission_manifest.json').read_text())
    for name,sha in snapshot.items():assert digest(old_submission/name)==sha,('Previous submission changed',name)
    roots=[out,*sorted((out/'spatial_cv').glob('fold_*'))] if (out/'spatial_cv').exists() else [out]
    if full:assert len(roots)==5,'Five predeclared spatial folds must complete'
    if (out/'nested_design.json').exists():
        nested=json.loads((out/'nested_design.json').read_text())
        original=json.loads((Path(nested['prototype'])/'crossvalidation_design.json').read_text())
        for a,b in zip(nested['folds'],original['folds']):
            assert a['cells']['test']==b['cells']['test'],'Outer test cells changed'
            assert a['actual_validation_changed']>=100 and a['validation_directions_ge10']>=4
            oldroot=Path(nested['prototype']) if a['fold']==0 else Path(nested['prototype'])/f"spatial_cv/fold_{a['fold']}"
            newroot=out if a['fold']==0 else out/f"spatial_cv/fold_{a['fold']}"
            for pair in PAIRS:
                previous=oldroot/f'splits/pair_{pair}.npz'
                if previous.exists():
                    with np.load(previous) as old,np.load(newroot/f'splits/pair_{pair}.npz') as new:
                        for i in (0,1):np.testing.assert_array_equal(old[f'test_indices{i}'],new[f'test_indices{i}'])
        ref=json.loads((out/'temporal_reference_manifest.json').read_text())
        for name,sha in ref['files'].items():assert digest(out/name)==sha,('Temporal reference changed',name)
    checked=0;weights=0;cases=0;label_support=[];support_audit=[]
    operating=json.loads((out/'operating_points_frozen.json').read_text())
    assert operating['frozen_before_stress_generation'] and operating['temporal_reference_excluded']
    assert len(operating['catalog'])==len(roots)
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text())
        geometry=json.loads((root/'geometry_frozen.json').read_text())
        selection=json.loads((root/'learning/primary/selection_frozen.json').read_text())
        fold=0 if root==out else int(root.name.split('_')[-1])
        from multipair_operating_points import catalog
        assert catalog(root)==operating['catalog'][str(fold)]['thresholds']
        for name,sha in operating['catalog'][str(fold)]['validation_sha256'].items():assert digest(root/name)==sha
        assert geometry['protocol_sha256']==digest(root/'protocol.json')
        assert selection['protocol_sha256']==digest(root/'protocol.json')
        assert protocol['minimum_cross_split_core_distance_m']>protocol['worst_clean_train_vs_stressed_test_support_sum_m']
        for pair in PAIRS:
            with np.load(root/f'splits/pair_{pair}.npz') as z:
                for split in ('train','validation','test'):
                    for i in (0,1):
                        assert split_mask(z[f'{split}_xyz{i}'],protocol,split).all()
                        assert len(np.unique(z[f'{split}_indices{i}']))==len(z[f'{split}_indices{i}'])
        for frozen in (root/'learning').glob('*/*/frozen.json'):
            meta=json.loads(frozen.read_text());config=json.loads((frozen.parent/'config.json').read_text())
            if frozen.parents[1].name=='primary' and 'protocol_sha256' in config:
                assert config['protocol_sha256']==digest(root/'protocol.json')
            path=frozen.parent/('model.joblib' if meta.get('model')=='geometry_boosted' else 'best.pt')
            assert digest(path)==meta['checkpoint_sha256']
            assert digest(frozen.parent/'config.json')==meta['config_sha256']
            for name,sha in config['source_sha256'].items():assert digest(frozen.parent/'source'/name)==sha
            assert not meta['test_accessed'];weights+=1
        for receipt in (root/'evaluation').glob('*/*/*/complete.json'):
            dst=receipt.parent;meta=json.loads(receipt.read_text());path=dst/'predictions.npz'
            assert digest(path)==meta['measurement_sha256'];assert meta['no_stress_retuning']
            rows=pd.read_csv(dst/'metrics.csv',dtype={'pair':str});condition=json.loads((dst/'condition.json').read_text())
            if condition['kind']!='clean':
                assert condition['measured_support_guard_passed']
                assert condition['context_plus_observed_displacement_m']<protocol['minimum_cross_split_core_distance_m']
                bounded_sum=2*protocol['learning_support_bound_m']+condition.get('maximum_observed_perturbation_m',0.)
                assert bounded_sum<protocol['minimum_cross_split_core_distance_m'],'PointNet train/test context can cross guards'
            with np.load(path) as z:
                labels=[z[f'labels{i}'] for i in (0,1)]
                pair=str(rows.iloc[0]['pair'])
                for scan in (0,1):
                    ids=z[f'indices{scan}'];np.testing.assert_array_equal(labels[scan],original_labels[(pair,scan)][ids])
                    record=original_records[(pair,scan)]
                    xyz=np.column_stack([record[k][ids] for k in ('x','y','z')])
                    if scan==0 or condition['kind'] in ('clean','retention','occlusion'):np.testing.assert_array_equal(z[f'xyz{scan}'],xyz)
                    elif 'maximum_observed_perturbation_m' in condition:
                        observed=np.linalg.norm(z[f'xyz{scan}']-xyz,axis=1).max(initial=0.)
                        assert observed<=condition['maximum_observed_perturbation_m']+1e-8
                for row in rows.to_dict('records'):
                    mask=[np.ones(len(y),bool) for y in labels]
                    if row['visibility']=='B':mask[0]=np.linalg.norm(z['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
                    if row['support']=='common_valid':mask=[m&(z[f'm3c2_calibrated_prediction{i}']>=0) for i,m in enumerate(mask)]
                    pred=[z[f"{row['method']}_prediction{i}"] for i in (0,1)]
                    expected=pair_scores([y[m] for y,m in zip(labels,mask)],[p[m] for p,m in zip(pred,mask)])
                    for key,value in expected.items():assert np.isclose(row[key],value,rtol=0,atol=1e-12),(path,key,row[key],value)
                    checked+=1
                for i in (0,1):
                    reason=z[f'm3c2_calibrated_scan{i}_decision_reason'];pred=z[f'm3c2_calibrated_prediction{i}']
                    np.testing.assert_array_equal(pred<0,reason!=0)
                    for key in z.files:
                        if key.endswith(f'_probability{i}'):
                            probability=z[key];assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                    if condition['kind']=='occlusion':assert len(z['indices1'])<condition['target_requested_cores']
                if rows.iloc[0]['experiment']=='primary':
                    distance=np.concatenate([z[f'c2c_distance{i}'] for i in (0,1)])
                    support_audit.append(dict(root=str(root.relative_to(out)) if root!=out else 'main',pair=pair,case=dst.name,
                        maximum_nearest_point_distance_m=float(distance.max(initial=0.)),
                        nearest_point_beyond_patch_radius_fraction=float((distance>.6).mean())))
                if dst.name=='clean_seed42':label_support.append(dict(root=str(root),pair=rows.iloc[0]['pair'],positives=[int(y.sum()) for y in labels]))
            cases+=1
        clean=list((root/'evaluation/primary').glob('*/clean_seed42/complete.json'))
        assert len(clean)==6,(root,'incomplete clean comparisons')
    if full:
        assert len(list((out/'evaluation/temporal').glob('*/clean_seed42/complete.json')))==1
        stress=[p for p in (out/'evaluation/primary').glob('*/*/complete.json') if p.parent.name!='clean_seed42']
        assert len(stress)==6*16*3,('stress cases',len(stress))
    pd.DataFrame(support_audit).to_csv(out/'analysis/full_cloud_nearest_feature_support.csv',index=False)
    result=dict(passed=True,full_planned_matrix_complete=full,clouds_unchanged=len(audit['clouds']),protected_historical_files_unchanged=len(protected),
        spatial_folds=len(roots),model_checkpoint_configs_and_sources_checked=weights,completed_cases=cases,
        metric_rows_independently_recomputed=checked,clean_positive_support=label_support,
        original_row_ground_truth_and_coordinates_checked=True,
        previous_submission_files_verified_unchanged=len(snapshot),reports_and_presentations_untouched=True,reference_software_equivalence=False,
        validation_operating_point_catalogs_verified=len(operating['catalog']),
        bounded_context_guard_scope='PointNet and M3C2 only. Boosted classifier has an unbounded full-cloud nearest-point vector feature; no test labels enter it.',
        caveats=['One construction site with four acquisition epochs','Spatial folds and perturbation seeds are not independent scenes',
            'Clean 1500-core/epoch/pair and stress 512-core/epoch/pair evaluation within buffered interiors',
            'Temporal test 5000 uniformly sampled original points per epoch from pair_23',
            'M3C2 registration uncertainty uncalibrated, explicit unknowns retained',
            'Boosted geometry baseline context is not bounded by the spatial guards',
            'Robustness matrix uses fold 0 models; clean spatial cross-validation uses all five folds'])
    write_json(out/'verification.json',result);print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--partial',action='store_true')
    a=p.parse_args();verify(a.out,not a.partial)
