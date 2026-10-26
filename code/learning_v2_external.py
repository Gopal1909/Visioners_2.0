"""Untuned independent Open_space evaluation, with separate CPU/GPU workers."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from learning_v2_runtime import BASE,DEFAULT,fold_dir,digest,write_json,write_csv,MemoryTimer,setup
from multipair_metrics import pair_scores,directional_three_class_scores


def cpu(out=DEFAULT):
    from scipy.spatial import cKDTree
    from learning_v2_data import features,contexts_for
    from multipair_data import read_cloud
    from multipair_features import extract_features,voxel_centroids
    out=Path(out);folder=out/'external_open_space';folder.mkdir(exist_ok=True)
    if (folder/'features_complete.json').exists():return
    assert all((fold_dir(out,i)/'learning/selection.json').exists() for i in range(5)), 'Freeze every model choice before accessing independent labels'
    plan=json.loads((out/'external_test_plan.json').read_text());scene=Path(plan['expected_folder'])
    files=[]
    for scan in (0,1):
        found=list(scene.rglob(f'pointCloud{scan}.ply'));assert len(found)==1,(scan,found);files.append(found[0])
    with MemoryTimer() as timer:
        data=[read_cloud(path) for path in files]
        ids=[np.sort(np.random.default_rng(91260+i).choice(len(d['xyz']),min(5000,len(d['xyz'])),replace=False)) for i,d in enumerate(data)]
        original=[d['xyz'][ids[i]].astype(np.float64) for i,d in enumerate(data)]
        # A common 150 m grid-aligned offset protects centroid precision for
        # large map coordinates, and preserves the 0.025/0.05/0.075 m grid phase.
        largest=max(float(np.abs(d['xyz']).max()) for d in data)
        origin=np.floor(np.min([d['xyz'].min(0).astype(np.float64) for d in data],axis=0)/150.)*150. if largest>1000 else np.zeros(3)
        for d in data:d['xyz']=d['xyz'].astype(np.float64)-origin
        trees=[cKDTree(d['xyz']) for d in data];contexts=contexts_for(trees)
        saved={};legacy={};labels=[d['labels'][ids[i]] for i,d in enumerate(data)]
        for scan in (0,1):
            centers=original[scan]-origin
            new=features([trees[scan],trees[1-scan]],[[c[scan],c[1-scan]] for c in contexts],centers)
            for key,value in new.items():saved[f'{key}{scan}']=value
            saved[f'labels{scan}']=labels[scan];saved[f'indices{scan}']=ids[scan];saved[f'xyz{scan}']=original[scan]
            saved[f'c2c_distance{scan}']=trees[1-scan].query(centers,workers=1)[0]
        del contexts,new
        old_context=[cKDTree(voxel_centroids(t.data,.075)) for t in trees]
        selected=json.loads((BASE/'learning/primary/selection_frozen.json').read_text())['selected_variant']
        for scan in (0,1):
            old=extract_features([trees[scan],trees[1-scan]],[old_context[scan],old_context[1-scan]],original[scan]-origin,
                'nearest' if selected=='nearest' else 'voxel')
            for key in ('p0','m0','p1','m1','aux'):legacy[f'{key}_{scan}']=old[key]
        np.savez_compressed(folder/'features.npz',**saved);np.savez_compressed(folder/'previous_pointnet_features.npz',**legacy)
        audit=dict(accessed_at_utc=datetime.now(timezone.utc).isoformat(),scene_directory=str(scene),
            original_coordinate_offset_m=origin.tolist(),sampling_seed_rule='91260 + scan; uniform without replacement',
            requested_cores_per_scan=5000,model_selection_finished_before_label_access=True,no_external_training_or_calibration=True,
            clouds=[dict(path=str(path),sha256=digest(path),points=len(d['xyz']),sampled_cores=len(ids[i]),
                sample_changed=int(labels[i].sum()),sample_unchanged=int((labels[i]==0).sum())) for i,(path,d) in enumerate(zip(files,data))],
            selection_sha256={str(i):digest(fold_dir(out,i)/'learning/selection.json') for i in range(5)},
            plan_sha256=digest(out/'external_test_plan.json'))
        write_json(folder/'data_audit.json',audit)
    write_json(folder/'features_complete.json',dict(passed=True,features_sha256=digest(folder/'features.npz'),
        previous_features_sha256=digest(folder/'previous_pointnet_features.npz'),timing=timer.stats))
    print('Independent scene features complete',json.dumps(audit['clouds']),flush=True)


def predict_one(out=DEFAULT,method='v2_validation_selected'):
    out=Path(out);folder=out/'external_open_space';marker=folder/f'probability_{method}.json'
    if marker.exists():return
    if method=='previous_pointnet':
        selected=json.loads((BASE/'learning/primary/selection_frozen.json').read_text())['selected_variant']+'_seed42'
        model_folder=BASE/f'learning/primary/{selected}'
    elif method=='bounded_geometry':model_folder=fold_dir(out,0)/'learning/bounded_geometry'
    else:
        selected=json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['primary_variant']+'_seed42'
        model_folder=fold_dir(out,0)/f'learning/{selected}'
    meta=json.loads((model_folder/'frozen.json').read_text())
    model_path=model_folder/('model.joblib' if method=='bounded_geometry' else 'best.pt')
    assert digest(model_path)==meta['checkpoint_sha256']
    result={}
    if method=='bounded_geometry':
        import joblib
        model=joblib.load(model_path)
        with np.load(folder/'features.npz') as z:
            for scan in (0,1):result[f'probability{scan}']=model.predict_proba(z[f'geo{scan}'])[:,1]
    else:
        import torch
        setup(42)
        if method=='previous_pointnet':
            from learning_v2_legacy import SiamesePointNet,predict
            model=SiamesePointNet().cuda()
        else:
            from learning_v2_model import MetricFusion,probabilities
            model=MetricFusion().cuda()
        state=torch.load(model_path,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);model.eval();del state
        path=folder/('previous_pointnet_features.npz' if method=='previous_pointnet' else 'features.npz')
        with np.load(path) as z:
            for scan in (0,1):
                if method=='previous_pointnet':
                    f={k:z[f'{k}_{scan}'] for k in ('p0','m0','p1','m1','aux')}
                    result[f'probability{scan}']=predict(model,f,'cuda')
                else:
                    f={k:z[f'{k}{scan}'] for k in ('x','mask','geo','labels')}
                    result[f'probability{scan}']=probabilities(model,f)
    target=folder/f'probability_{method}.npz';np.savez_compressed(target,**result)
    write_json(marker,dict(folder=str(model_folder),sha256=meta['checkpoint_sha256'],threshold=meta['threshold'],probability_sha256=digest(target)))
    print('Independent frozen-model prediction complete:',method,flush=True)


def inference(out=DEFAULT):
    out=Path(out);folder=out/'external_open_space'
    if (folder/'complete.json').exists():return
    receipt=json.loads((folder/'features_complete.json').read_text())
    assert digest(folder/'features.npz')==receipt['features_sha256']
    assert digest(folder/'previous_pointnet_features.npz')==receipt['previous_features_sha256']
    # Separate model processes bound CUDA module and host-memory lifetimes.
    names=('v2_validation_selected','bounded_geometry','previous_pointnet')
    for method in names:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'predict-one','--out',str(out),'--method',method],check=True)
    geometry=json.loads((BASE/'geometry_frozen.json').read_text());threshold={};prediction={}
    with np.load(folder/'features.npz') as z:
        result={k:z[k] for k in z.files if k.startswith(('labels','indices','xyz','c2c_distance'))}
    provenance={}
    for method in names:
        meta=json.loads((folder/f'probability_{method}.json').read_text());threshold[method]=meta['threshold'];provenance[method]=meta
        path=folder/f'probability_{method}.npz';assert digest(path)==meta['probability_sha256']
        with np.load(path) as z:prob=[z[f'probability{i}'] for i in (0,1)]
        prediction[method]=[(p>=threshold[method]).astype(np.int8) for p in prob]
        for scan in (0,1):result[f'{method}_probability{scan}']=prob[scan]
    threshold.update(c2c_calibrated=geometry['c2c_threshold_m'],c2c_legacy_075=.075)
    for method in ('c2c_calibrated','c2c_legacy_075'):
        prediction[method]=[(result[f'c2c_distance{i}']>threshold[method]).astype(np.int8) for i in (0,1)]
    labels=[result[f'labels{i}'] for i in (0,1)]
    rows=[dict(method=method,**pair_scores(labels,p),**directional_three_class_scores(labels,p)) for method,p in prediction.items()]
    write_csv(folder/'metrics.csv',rows)
    for method,p in prediction.items():
        for scan in (0,1):result[f'{method}_prediction{scan}']=p[scan]
    np.savez_compressed(folder/'predictions.npz',**result)
    write_json(folder/'complete.json',dict(passed=True,predictions_sha256=digest(folder/'predictions.npz'),
        frozen_thresholds=threshold,no_external_training_or_calibration=True,
        new_model=provenance['v2_validation_selected'],previous_model=provenance['previous_pointnet'],geometry_model=provenance['bounded_geometry'],
        primary_model_policy='Predeclared fold0 BikeParkingConstruction validation-selected model, seed42; no external choice among seeds or folds'))
    for row in rows:print({k:row[k] for k in ('method','selection_binary_macro_f1','directional_changed_f1','pooled_three_class_macro_f1')},flush=True)


def verify(out=DEFAULT):
    from plyfile import PlyData
    out=Path(out);folder=out/'external_open_space';audit=json.loads((folder/'data_audit.json').read_text())
    receipt=json.loads((folder/'complete.json').read_text())
    assert receipt['no_external_training_or_calibration'] and audit['model_selection_finished_before_label_access']
    assert digest(folder/'predictions.npz')==receipt['predictions_sha256']
    for fold,sha in audit['selection_sha256'].items():assert digest(fold_dir(out,int(fold))/'learning/selection.json')==sha
    assert digest(out/'external_test_plan.json')==audit['plan_sha256']
    with np.load(folder/'predictions.npz') as z:
        labels=[z[f'labels{i}'] for i in (0,1)]
        for scan,cloud in enumerate(audit['clouds']):
            assert digest(cloud['path'])==cloud['sha256'];record=PlyData.read(cloud['path'],mmap='r')['params'].data
            expected=np.sort(np.random.default_rng(91260+scan).choice(len(record),min(5000,len(record)),replace=False))
            np.testing.assert_array_equal(z[f'indices{scan}'],expected)
            np.testing.assert_array_equal(labels[scan],record['cd_type'][expected])
            np.testing.assert_array_equal(z[f'xyz{scan}'],np.column_stack([record[k][expected] for k in ('x','y','z')]))
        for row in list(csv.DictReader((folder/'metrics.csv').open(newline=''))):
            p=[z[f"{row['method']}_prediction{i}"] for i in (0,1)]
            measured=dict(**pair_scores(labels,p),**directional_three_class_scores(labels,p))
            for key,value in measured.items():assert np.isclose(float(row[key]),value,rtol=0,atol=1e-12)
            for scan in (0,1):
                method=row['method'];threshold=receipt['frozen_thresholds'][method]
                if method.startswith('c2c_'):expected=(z[f'c2c_distance{scan}']>threshold).astype(np.int8)
                else:
                    probability=z[f'{method}_probability{scan}'];assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                    expected=(probability>=threshold).astype(np.int8)
                np.testing.assert_array_equal(p[scan],expected)
    for key in ('new_model','previous_model','geometry_model'):
        meta=receipt[key];path=Path(meta['folder'])/('model.joblib' if key=='geometry_model' else 'best.pt');assert digest(path)==meta['sha256']
        frozen=json.loads((Path(meta['folder'])/'frozen.json').read_text())
        method={'new_model':'v2_validation_selected','previous_model':'previous_pointnet','geometry_model':'bounded_geometry'}[key]
        assert receipt['frozen_thresholds'][method]==frozen['threshold']
    assert receipt['frozen_thresholds']['c2c_calibrated']==json.loads((BASE/'geometry_frozen.json').read_text())['c2c_threshold_m']
    assert receipt['frozen_thresholds']['c2c_legacy_075']==.075
    result=dict(passed=True,independent_scene_evaluated=True,scene='User-supplied MultiChange3D Open_space',
        source_clouds_verified=2,uniform_original_cores_verified=sum(c['sampled_cores'] for c in audit['clouds']),
        metric_rows_independently_recomputed=5,all_thresholds_frozen_on_BikeParkingConstruction=True,
        independent_scene_never_used_for_training_or_selection=True,
        scope='One additional scene, not a multi-scene generalization benchmark; full support with uniformly sampled original evaluation cores.')
    write_json(folder/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['cpu','inference','verify','predict-one']);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--method',choices=['v2_validation_selected','bounded_geometry','previous_pointnet']);a=p.parse_args()
    predict_one(a.out,a.method) if a.stage=='predict-one' else {'cpu':cpu,'inference':inference,'verify':verify}[a.stage](a.out)
