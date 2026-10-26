"""Independent original-row, feature-bank, validation-selection and score audits."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from learning_v3_runtime import (ROOT,BASE,DEFAULT,PAIRS,CANDIDATES,EPOCHS,CORES,THRESHOLDS,
    fold_dir,bank,digest,write_json,write_csv)


def features_audit(out=DEFAULT,fold=0):
    from learning_v3_data import balanced_ids
    from learning_v3_support import load_mapped_pair,mapped_transform
    from learning_v3_kernels import split_mask
    out=Path(out);root=fold_dir(out,fold);protocol=json.loads((root/'protocol.json').read_text());probes=0;views=0;unique=[]
    for pair in PAIRS:
        data=load_mapped_pair(out,pair);eligible=[split_mask(d['xyz'],protocol,'train') for d in data];seen=[set(),set()]
        for epoch in range(EPOCHS):
            folder=root/f'epochs/epoch_{epoch:02d}';meta=json.loads((folder/f'pair_{pair}.json').read_text())
            assert meta['training_cells_only'] and meta['full_scan_transformed_before_rebuilding'] and meta['total_support_guard_m']<5
            assert digest(folder/f'pair_{pair}.npz')==meta['sha256']
            points,retained,maximum=mapped_transform(data[1]['xyz'],meta['parameters'],meta['seed'],np.array(protocol['stress_rotation_center_m']),out/'disk_support'/f'audit_f{fold}_p{pair}')
            assert np.isclose(maximum,meta['maximum_observed_training_displacement_m'],rtol=0,atol=1e-10)
            retained_mask=np.zeros(len(data[1]['xyz']),bool);retained_mask[retained]=True
            assembled=bank(out,fold,epoch)
            with np.load(folder/f'pair_{pair}.npz') as z:
                for scan in (0,1):
                    ids=z[f'indices{scan}'];seen[scan].update(ids.tolist())
                    expected=[]
                    for augmented,count in ((False,CORES//3),(True,CORES-CORES//3)):
                        mask=eligible[scan] if not augmented or scan==0 else eligible[scan]&retained_mask
                        expected.append(balanced_ids(data[scan]['labels'],mask,count,np.random.default_rng(meta['seed']+100000*scan+1000*augmented)))
                    np.testing.assert_array_equal(ids,np.concatenate(expected));assert eligible[scan][ids].all()
                    np.testing.assert_array_equal(z[f'labels{scan}'],data[scan]['labels'][ids]);assert z[f'available{scan}'].all()
                    xyz=data[scan]['xyz'][ids].astype(np.float64)
                    if scan==1:
                        aug=z['augmented1'];pos=np.searchsorted(retained,ids[aug]);np.testing.assert_array_equal(retained[pos],ids[aug]);xyz[aug]=points[pos]
                    np.testing.assert_allclose(z[f'xyz{scan}'],xyz,rtol=0,atol=1e-10)
                    x=z[f'x{scan}'];mask=z[f'mask{scan}'];assert np.isfinite(x).all() and np.isfinite(z[f'geo{scan}']).all()
                    assert (x[~mask]==0).all();assert (np.linalg.norm(x[...,:3],axis=-1)[mask]<=1.00001).all()
                    ids_probe=np.random.default_rng(99100+epoch+scan).choice(CORES,8,replace=False);offset=(PAIRS.index(pair)*2+scan)*CORES
                    for key in ('x','mask','geo','labels','indices','available'):
                        np.testing.assert_array_equal(assembled[key][offset+ids_probe],z[f'{key}{scan}'][ids_probe])
                    assert (assembled['pair'][offset+ids_probe]==int(pair)).all() and (assembled['scan'][offset+ids_probe]==scan).all();probes+=8
            del assembled,points,retained,retained_mask;views+=1
        unique.extend(dict(pair=pair,scan=i,unique_original_cores=len(seen[i]),epoch_observations=EPOCHS*CORES) for i in (0,1))
        print('V3 fresh-row audit',fold,pair,flush=True)
    assembled_manifest={str(p.relative_to(root)):digest(p) for p in (root/'epochs').glob('*/assembled/*.npy')}
    validation_manifest={str(p.relative_to(BASE)):digest(p) for p in (BASE/f'fold_{fold}/assembled').glob('validation_*/*.npy')}
    assert len(assembled_manifest)==EPOCHS*8 and len(validation_manifest)==24
    write_json(root/'assembled_manifest.json',assembled_manifest);write_json(root/'reused_validation_manifest.json',validation_manifest)
    result=dict(passed=True,fold=fold,epoch_pair_caches_verified=views,original_row_observations_verified=views*2*CORES,
        assembly_alignment_probes=probes,fresh_core_inventory=unique,test_cells_never_sampled=True,
        full_scan_transform_and_original_id_mapping_recomputed=True,
        assembled_manifest_sha256=digest(root/'assembled_manifest.json'),reused_validation_manifest_sha256=digest(root/'reused_validation_manifest.json'))
    write_json(root/'fresh_features_verification.json',result)


def preserved(out):
    receipt=json.loads((out/'baseline_preservation.json').read_text())
    for p,sha in receipt['files'].items():assert digest(BASE/p)==sha,('Baseline changed',p)
    original=json.loads((out/'data_audit.json').read_text())
    for row in original['clouds']:assert digest(row['path'])==row['sha256']
    manifest=json.loads((ROOT/'final_submission/submission_manifest.json').read_text())
    for p,sha in manifest.items():assert digest(ROOT/'final_submission'/p)==sha
    sources=BASE/'execution_logs/final_sources/manifest.json'
    assert digest(sources)==json.loads((BASE/'final_review.json').read_text())['final_sources_manifest_sha256']
    source_manifest=json.loads(sources.read_text())
    for name,sha in source_manifest.items():assert digest(ROOT/'code'/name)==sha,('Previous source changed',name)
    return dict(previous_result_files_preserved=len(receipt['files']),previous_source_files_preserved=len(source_manifest),original_clouds_preserved=12,submission_files_preserved=len(manifest))


def model_audit(out,fold):
    from multipair_metrics import choose_threshold,binary_scores
    root=fold_dir(out,fold);selection=json.loads((root/'learning/selection.json').read_text());models={}
    for manifest,base in [('assembled_manifest.json',root),('reused_validation_manifest.json',BASE)]:
        for p,sha in json.loads((root/manifest).read_text()).items():assert digest(base/p)==sha
    candidates=[]
    for path in (root/'learning').glob('*/frozen.json'):
        meta=json.loads(path.read_text());folder=path.parent;assert not meta['test_accessed']
        assert digest(folder/'best.pt')==meta['checkpoint_sha256'] and digest(folder/'config.json')==meta['config_sha256']
        config=json.loads((folder/'config.json').read_text())
        assert digest(root/'fresh_features_verification.json')==config['fresh_features_verification_sha256']
        for name,sha in config['source_sha256'].items():assert digest(folder/'source'/name)==sha
        with np.load(folder/'validation_clean.npz') as z:
            threshold,_=choose_threshold(z['labels'],z['probability'],z['pair']*2+z['scan'],THRESHOLDS)
        assert threshold['threshold']==meta['threshold'] and np.isclose(threshold['objective'],meta['validation_objective'],atol=1e-12,rtol=0)
        for view in ('clean','mild','medium'):
            with np.load(folder/f'validation_{view}.npz') as z:
                keep=z['available'];code=z['pair']*2+z['scan'];values=[]
                for group in np.unique(code):
                    mask=keep&(code==group);values.append(binary_scores(z['labels'][mask],(z['probability'][mask]>=meta['threshold']).astype(np.int8))['binary_macro_f1'])
                assert np.isclose(np.mean(values),meta['validation_by_view'][view]['binary_macro_f1'],atol=1e-12,rtol=0)
        models[folder.name]=meta
        if meta['seed']==42:candidates.append(meta)
    best=max(candidates,key=lambda x:(x['robustness_validation_objective'],-CANDIDATES.index(x['candidate'])))
    assert best['candidate']==selection['candidate'];return models


def pilot(out=DEFAULT):
    out=Path(out);root=fold_dir(out,0);selection=json.loads((root/'learning/selection.json').read_text())
    models=model_audit(out,0);assert len(models)==3 and not selection['pilot_gate_passed']
    assert not list(root.glob('evaluation/*/*/complete.json'))
    from learning_v3_runtime import validation
    counts={}
    for view in ('clean','mild','medium'):
        d=validation(0,view);counts[view]=int(d['available'].sum())
    folder=out/'analysis';folder.mkdir(exist_ok=True);rows=[]
    for name,meta in models.items():
        rows.append(dict(method=name,clean_validation_macro=meta['validation_objective'],clean_validation_changed_f1=meta['validation_changed_f1'],
            three_view_validation_macro=meta['robustness_validation_objective'],best_epoch=meta['best_epoch']))
    old_choice=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    comparisons=[(name,root/'learning'/name,meta) for name,meta in models.items()]
    for key,alias in [('primary_variant','v2_clean_validation_selected'),('secondary_robust_variant','v2_robust_validation_selected')]:
        old_folder=BASE/f"fold_0/learning/{old_choice[key]}_seed42";meta=json.loads((old_folder/'frozen.json').read_text())
        rows.append(dict(method=alias,clean_validation_macro=meta['validation_objective'],clean_validation_changed_f1=meta['validation_changed_f1'],
            three_view_validation_macro=meta['robustness_validation_objective'],best_epoch=meta['best_epoch']))
        comparisons.append((alias,old_folder,meta))
    write_csv(folder/'pilot_validation_comparison.csv',rows)
    from multipair_metrics import binary_scores,pair_scores
    from learning_v3_metrics import supplementary
    direction_rows=[];pair_rows=[]
    for name,model_folder,meta in comparisons:
        for view in ('clean','mild','medium'):
            with np.load(model_folder/f'validation_{view}.npz') as z:
                reference=validation(0,view)
                for key in ('labels','pair','scan','indices','available'):np.testing.assert_array_equal(z[key],reference[key])
                for pair in PAIRS:
                    labels=[];predictions=[]
                    for scan in (0,1):
                        mask=(z['pair']==int(pair))&(z['scan']==scan)&z['available'];y=z['labels'][mask];p=z['probability'][mask];estimate=(p>=meta['threshold']).astype(np.int8)
                        labels.append(y);predictions.append(estimate)
                        direction_rows.append(dict(method=name,view=view,pair=pair,scan=scan,threshold=meta['threshold'],
                            mean_probability_unchanged=float(p[y==0].mean()) if (y==0).any() else None,
                            mean_probability_changed=float(p[y==1].mean()) if (y==1).any() else None,
                            predicted_changed_fraction=float(estimate.mean()),**binary_scores(y,estimate)))
                    values=pair_scores(labels,predictions);pair_rows.append(dict(method=name,view=view,pair=pair,**values,**supplementary(values)))
                del reference
    write_csv(folder/'validation_direction_error_breakdown.csv',direction_rows);write_csv(folder/'validation_pair_metrics.csv',pair_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for name in models:
        history=list(csv.DictReader((root/'learning'/name/'history.csv').open()))
        ax.plot([int(r['epoch']) for r in history],[float(r['clean_validation_bce']) for r in history],label=name)
    ax.set_xlabel('Epoch (fresh construction cores each epoch)');ax.set_ylabel('Clean validation BCE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'pilot_validation_curves.png',dpi=170);plt.close(fig)
    result=dict(passed=True,pilot_gate_passed=False,new_test_cases_scored=0,models_verified=3,
        reason='Predeclared validation gate not met; full five-fold/stress expansion withheld',validation_core_counts=counts,
        selection=selection,validation_direction_error_rows=len(direction_rows),validation_pair_metric_rows=len(pair_rows),**preserved(out))
    write_json(out/'pilot_verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='selection'},indent=2),flush=True)


def verify(out=DEFAULT):
    from multipair_metrics import pair_scores,directional_three_class_scores
    from learning_v3_metrics import supplementary
    out=Path(out);cases=rows=models_count=0
    assert json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['pilot_gate_passed']
    for fold in range(5):
        root=fold_dir(out,fold);models=model_audit(out,fold);models_count+=len(models)
        assert json.loads((root/'fresh_features_verification.json').read_text())['passed']
        for receipt in root.glob('evaluation/*/*/complete.json'):
            dst=receipt.parent;meta=json.loads(receipt.read_text());assert meta['no_test_retuning']
            assert digest(dst/'predictions.npz')==meta['predictions_sha256']
            assert digest(root/'learning/selection.json')==meta['selection_sha256'];assert set(models)==set(meta['model_sha256'])
            for name,model in models.items():assert meta['model_sha256'][name]==model['checkpoint_sha256']
            src=BASE/f'fold_{fold}/evaluation'/dst.parent.name/dst.name
            condition=json.loads((dst/'condition.json').read_text());assert condition['v3_total_support_guard_m']<5
            with np.load(dst/'predictions.npz') as z,np.load(src/'predictions.npz') as old:
                for scan in (0,1):
                    for key in ('indices','labels','xyz','common_valid'):np.testing.assert_array_equal(z[f'{key}{scan}'],old[f'{key}{scan}'])
                    for name,model in models.items():
                        p=z[f'{name}_probability{scan}'];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                        np.testing.assert_array_equal(z[f'{name}_prediction{scan}'],(p>=model['threshold']).astype(np.int8))
                for row in csv.DictReader((dst/'metrics.csv').open()):
                    masks=[np.ones(len(z[f'labels{i}']),bool) for i in (0,1)]
                    if row['visibility']=='B':masks[0]=np.linalg.norm(z['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
                    if row['support']=='common_valid':masks=[m&z[f'common_valid{i}'] for i,m in enumerate(masks)]
                    calculated=pair_scores([z[f'labels{i}'][m] for i,m in enumerate(masks)],[z[f"{row['method']}_prediction{i}"][m] for i,m in enumerate(masks)])
                    for k,v in calculated.items():assert np.isclose(float(row[k]),v,atol=1e-12,rtol=0)
                    direct=directional_three_class_scores([z[f'labels{i}'][m] for i,m in enumerate(masks)],[z[f"{row['method']}_prediction{i}"][m] for i,m in enumerate(masks)])
                    derived=supplementary(row)
                    for k,v in direct.items():assert np.isclose(derived[k],v,atol=1e-12,rtol=0)
                    rows+=1
            cases+=1
    assert cases==318 and models_count==17,(cases,models_count)
    write_json(out/'verification.json',dict(passed=True,cases_verified=cases,metric_rows_independently_recomputed=rows,models_verified=models_count,
        same_original_test_cores_as_v2=True,validation_only_selection=True,construction_scene_only=True,
        progress_completion_percentages_claimed=False,independent_construction_site=False,**preserved(out)))
    print('V3 full verification passed',cases,rows,models_count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','pilot','verify']);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);a=p.parse_args()
    if a.stage=='features':features_audit(a.out,a.fold)
    elif a.stage=='pilot':pilot(a.out)
    else:verify(a.out)
