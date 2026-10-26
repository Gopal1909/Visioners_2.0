"""Validation-only scalar-threshold sensitivity; fitted geometry/models fixed.

The catalog reads validation CSVs, never test labels or test predictions.
It changes neither an encoder, a normal/projection scale nor an unknown mask.
The already inspected temporal-reference experiment is deliberately excluded.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multipair_data import DEFAULT
from multipair_geometry import NORMALS
from multipair_evaluate import metric_rows
from m3c2_baseline import digest,write_json


def best_changed_threshold(frame):
    score='changed_f1' if 'changed_f1' in frame else 'directional_changed_f1'
    return float(frame.sort_values([score,'threshold'],ascending=[False,False]).iloc[0].threshold)


def catalog(root):
    root=Path(root);geometry=json.loads((root/'geometry_frozen.json').read_text())
    validation=pd.read_csv(root/'geometry_calibration/all_candidate_metrics.csv')
    c2c=validation[validation.method=='C2C'].groupby('threshold_m').directional_changed_f1.mean().reset_index().rename(columns={'threshold_m':'threshold'})
    config=geometry['m3c2_config'];normal=next(name for name,radii in NORMALS.items() if tuple(radii)==tuple(config['normal_radii_m']))
    m=validation[(validation.method=='M3C2-style')&(validation.normal_set==normal)&
        np.isclose(validation.half_length_m,config['half_length_m'])&np.isclose(validation.projection_radius_m,config['projection_radii_m'][0])]
    m=m.groupby('effect_floor_m').directional_changed_f1.mean().reset_index().rename(columns={'effect_floor_m':'threshold'})
    result={'c2c_calibrated':dict(threshold=best_changed_threshold(c2c),source='geometry_calibration/all_candidate_metrics.csv'),
        'm3c2_calibrated':dict(threshold=best_changed_threshold(m),source='geometry_calibration/all_candidate_metrics.csv')}
    for path in (root/'learning/primary').glob('*/thresholds.csv'):
        result[path.parent.name]=dict(threshold=best_changed_threshold(pd.read_csv(path)),source=str(path.relative_to(root)).replace('\\','/'))
    return result


def freeze_catalog(out):
    out=Path(out);roots=[out,*sorted((out/'spatial_cv').glob('fold_*'))];catalogs={}
    for root in roots:
        selected=catalog(root);fold=0 if root==out else int(root.name.split('_')[-1])
        catalogs[str(fold)]=dict(thresholds=selected,
            validation_sha256={v['source']:digest(root/v['source']) for v in selected.values()},
            geometry_and_model_parameters_fixed=True,unknown_masks_fixed=True)
    path=out/'operating_points_frozen.json'
    if path.exists():
        saved=json.loads(path.read_text());assert saved['catalog']==catalogs
        return saved
    started=[p for p in (out/'evaluation/primary').glob('*/*') if p.is_dir() and p.name!='clean_seed42']
    state=json.loads((out/'execution_state.json').read_text()) if (out/'execution_state.json').exists() else {}
    result=dict(catalog=catalogs,selection_data='Clean validation CSVs only',temporal_reference_excluded=True,
        frozen_before_stress_generation=len(started)==0,snapshot_stage=state.get('stage'),
        source_sha256=digest(__file__))
    write_json(path,result);return result


def run(out=DEFAULT):
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True)
    roots=[out,*sorted((out/'spatial_cv').glob('fold_*'))]
    rows=[];frozen=freeze_catalog(out);catalogs=frozen['catalog']
    for root in roots:
        fold=0 if root==out else int(root.name.split('_')[-1])
        selected=catalogs[str(fold)]['thresholds'];assert selected==catalog(root)
        for path in (root/'evaluation/primary').glob('*/*/predictions.npz'):
            if not (path.parent/'complete.json').exists():continue
            condition=json.loads((path.parent/'condition.json').read_text());name,seed=path.parent.name.rsplit('_seed',1)
            with np.load(path) as z:
                labels=[z[f'labels{i}'] for i in (0,1)];predictions={}
                for method,setting in selected.items():
                    if f'{method}_prediction0' not in z.files:continue
                    if method=='c2c_calibrated':prediction=[(z[f'c2c_distance{i}']>setting['threshold']).astype(np.int8) for i in (0,1)]
                    elif method=='m3c2_calibrated':
                        prediction=[]
                        for i in (0,1):
                            p=np.full(len(labels[i]),-1,np.int8);valid=z[f'm3c2_calibrated_prediction{i}']>=0
                            distance=np.abs(z[f'm3c2_calibrated_scan{i}_signed_distance_m'][:,0]);lod=z[f'm3c2_calibrated_scan{i}_decision_lod95_m']
                            p[valid]=(distance[valid]>np.maximum(lod[valid],setting['threshold'])).astype(np.int8);prediction.append(p)
                    else:prediction=[(z[f'{method}_probability{i}']>=setting['threshold']).astype(np.int8) for i in (0,1)]
                    predictions[method]=prediction
                common=[z[f'm3c2_calibrated_prediction{i}']>=0 for i in (0,1)]
                for visibility in ('A','B') if condition['kind']=='occlusion' else ('standard',):
                    masks=[np.ones(len(y),bool) for y in labels]
                    if visibility=='B':masks[0]=np.linalg.norm(z['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
                    base=dict(experiment='primary',spatial_fold=fold,pair=path.parents[1].name.split('_')[1],condition=name,
                        family=condition['kind'],severity=condition['severity'],perturbation_seed=int(seed),visibility=visibility,
                        operating_point='validation_changed_f1',geometry_fixed=True)
                    rows.extend(metric_rows(labels,predictions,common,base,masks,condition['target_requested_cores'],condition.get('requested_label_counts')))
    write_json(folder/'changed_f1_operating_points.json',dict(catalog=catalogs,temporal_reference_excluded=True,
        decision_source='Clean validation rows only; no test-score maximization. Architecture and unknown masks stay fixed.',
        frozen_catalog_sha256=digest(out/'operating_points_frozen.json'),source_sha256=digest(__file__)))
    frame=pd.DataFrame(rows);frame.to_csv(folder/'validation_changed_f1_metrics.csv',index=False)
    clean=frame[(frame.condition=='clean')&(frame.support=='full_requested')].copy();chosen=[]
    for fold,part in clean.groupby('spatial_fold'):
        root=out if fold==0 else out/f'spatial_cv/fold_{fold}'
        variant=json.loads((root/'learning/primary/selection_frozen.json').read_text())['selected_variant']
        select=part[part.method.isin(['c2c_calibrated','m3c2_calibrated','geometry_boosted',variant+'_seed42'])].copy()
        select['method']=select.method.replace({variant+'_seed42':'pointnet_validation_selected'});chosen.append(select)
    pd.concat(chosen).groupby('method').agg(changed_f1=('directional_changed_f1','mean'),binary_macro_f1=('selection_binary_macro_f1','mean'),
        detector_unknown_rate=('detector_unknown_rate','mean'),pair_fold_count=('pair','size')).to_csv(folder/'changed_f1_operating_point_spatial_summary.csv')


if __name__=='__main__':run()
