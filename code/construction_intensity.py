"""Construction sensor inventory, validation error atlas, and intensity ablation.

Reuses the completed 381-column fresh geometry model as its unchanged control.
Only original-core intensity is added. No full-cloud neighborhoods are rebuilt.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from plyfile import PlyData

from learning_v3_control import ROOT, BASE, digest, write_json
from learning_v2_runtime import THRESHOLDS, MemoryTimer, write_csv
from learning_v3_kernels import discover_pair
from multipair_metrics import balanced_training_weights
from construction_evidence_followup import VIEWS, FIELDS, summarize, summarize_views
from construction_match_distributions import DEFAULT as CONTROL, feature_bank

DEFAULT=ROOT/'results/construction_intensity_2026-10-26_v1'
RAW=ROOT/'dataset/MultiChange3D/BikeParkingConstruction'
CANDIDATES=('aligned_intensity','shuffled_intensity')
PAIRS=('01','02','03','12','13','23')


def gather_intensity(record,indices):
    """Attribute extraction from original IDs; never accesses the label field."""
    ids=np.asarray(indices)
    assert ids.ndim==1 and ids.dtype.kind in 'iu' and (ids>=0).all() and (ids<len(record)).all()
    values=np.asarray(record['intensity'][ids],np.float32)
    assert np.isfinite(values).all() and ((values>=0)&(values<=255)).all()
    return values/np.float32(255.)


def shuffled_within_cloud(intensity,pairs,scans,seed):
    """Negative control keeps each cloud's intensity distribution, without labels."""
    values=np.asarray(intensity);pairs=np.asarray(pairs);scans=np.asarray(scans)
    assert values.shape==pairs.shape==scans.shape
    shuffled=values.copy();rng=np.random.default_rng(seed)
    for pair in sorted(np.unique(pairs)):
        for scan in (0,1):
            ids=np.flatnonzero((pairs==pair)&(scans==scan))
            shuffled[ids]=values[rng.permutation(ids)]
    return shuffled


def local_surface(xyz,mask):
    """Local shape descriptor, not a ground/occlusion/semantic label."""
    xyz=np.asarray(xyz,np.float64);mask=np.asarray(mask,bool)
    assert xyz.shape==(*mask.shape,3)
    safe=np.where(mask[...,None],xyz,0.);counts=mask.sum(-1);den=counts.clip(1)
    mean=safe.sum(-2)/den[...,None]
    centered=np.where(mask[...,None],safe-mean[...,None,:],0.)
    covariance=np.einsum('nki,nkj->nij',centered,centered)/den[:,None,None]
    eigenvalues,eigenvectors=np.linalg.eigh(covariance);eigenvalues=np.maximum(eigenvalues,0.)
    valid=(counts>=3)&(eigenvalues[:,1]>1e-10)
    planarity=(eigenvalues[:,1]-eigenvalues[:,0])/np.maximum(eigenvalues[:,2],1e-12)
    normal_z=np.abs(eigenvectors[:,2,0])
    return dict(valid=valid,planarity=planarity,absolute_normal_z=normal_z,
                locally_horizontal_planar=valid&(planarity>=.5)&(normal_z>=.9))


def prepare(out=DEFAULT):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'design.json').exists():return
    assert json.loads((CONTROL/'verification.json').read_text())['passed']
    inventory=[];raw_manifest={}
    for pair in PAIRS:
        for scan,path in enumerate(discover_pair(RAW,pair)):
            ply=PlyData.read(path,mmap='r');record=ply['params'].data
            ids=np.unique(np.linspace(0,len(record)-1,20000,dtype=np.int64))
            intensity=gather_intensity(record,ids)*255.
            sha=digest(path);raw_manifest[str(path)]=sha
            inventory.append(dict(pair=pair,scan=scan,path=str(path),sha256=sha,points=len(record),
                elements=[dict(name=e.name,count=e.count,properties=[dict(name=p.name,dtype=str(p.val_dtype)) for p in e.properties]) for e in ply.elements],
                comments=ply.comments,obj_info=ply.obj_info,intensity_probe=dict(rows=len(ids),
                    quantiles=np.quantile(intensity,[0,.01,.25,.5,.75,.99,1]).tolist(),distinct=len(np.unique(intensity)))))
    archives={}
    for path in sorted(RAW.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:archives[str(path)]=[entry.filename for entry in archive.infolist() if not entry.is_dir()]
    sensor_inventory=dict(clouds=inventory,downloaded_files=[str(p.relative_to(RAW)) for p in sorted(RAW.rglob('*')) if p.is_file()],
        archive_members=archives,available_attributes=['x','y','z','intensity','cd_type'],
        rgb_available=False,sensor_origins_available=False,timestamps_available=False,calibration_available=False,
        scope='Downloaded BikeParkingConstruction six-pair package only; this does not assert that the source acquisition has no poses elsewhere',
        visibility='These files cannot establish ray-based free-space visibility or distinguish occlusion from absent geometry',
        intensity='A radiometric input, not an object/material/construction-event annotation; scan intensity distributions differ')
    write_json(out/'sensor_inventory.json',sensor_inventory)
    inputs=raw_manifest.copy()
    for source in (CONTROL/'features').glob('*/*.npy'):inputs[str(source)]=digest(source)
    for source in (CONTROL/'models/fresh_match_distributions').glob('*'):
        if source.is_file():inputs[str(source)]=digest(source)
    for view in VIEWS:
        folder=BASE/f'fold_0/assembled/validation_{view}'
        for field in ('x','mask','geo',*FIELDS):
            source=folder/f'{field}.npy';inputs[str(source)]=digest(source)
    selected=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    reference=BASE/'fold_0/learning'/(selected['secondary_robust_variant']+'_seed42')
    for kind in ('primary_variant','secondary_robust_variant'):
        folder=BASE/'fold_0/learning'/(selected[kind]+'_seed42')
        for name in ('frozen.json',*[f'validation_{view}.npz' for view in VIEWS]):
            source=folder/name;inputs[str(source)]=digest(source)
    write_json(out/'input_manifest.json',inputs)
    protected={str(p):digest(p) for p in (ROOT/'code').glob('*.py')}
    for previous in (CONTROL,ROOT/'results/construction_evidence_fusion_2026-10-26_v1'):
        for p in previous.glob('*.json'):protected[str(p)]=digest(p)
    write_json(out/'protected_manifest.json',protected)
    sources=out/'sources';sources.mkdir(exist_ok=True)
    for name in ('construction_intensity.py','construction_evidence_followup.py','construction_match_distributions.py',
                 'learning_v2_runtime.py','learning_v3_control.py','multipair_metrics.py'):
        shutil.copy2(ROOT/'code'/name,sources/name)
    estimator=json.loads((CONTROL/'design.json').read_text())['estimator']
    write_json(out/'design.json',dict(
        purpose='Test whether unused original-core intensity complements cross-scan geometry and explains a portion of validation errors',
        source_control=str(CONTROL),reference=str(reference),fold=0,seed=42,training_rows=288000,
        candidates=list(CANDIDATES),new_columns=1,total_columns=382,
        added_feature='Original raw intensity of the core divided by 255; no labels, pair identity, epoch, absolute position, per-scan fitted normalizer, or nearest-neighbor queries',
        negative_control='Deterministically shuffle intensity within each pair/scan among training observations, and among validation observations with a separate seed; preserve marginal scan distributions without reading labels',
        shuffle_seed_train=920042,shuffle_seed_validation=930042,
        augmentation='Reuse all 16 completed fresh geometry banks. Geometric perturbations preserve core identity and intensity; unavailable validation points remain excluded by the original mask.',
        estimator=estimator,thresholds=THRESHOLDS.tolist(),
        threshold_selection='One threshold shared by all pairs and views, maximum mean clean/mild/medium validation macro under clean/FPR constraints; ties changed-F1, clean macro, higher threshold',
        clean_floor='V2 robust clean macro minus .005',fpr_ceiling='V2 robust clean FPR plus .005',
        pilot_gate='Aligned-intensity mean validation macro >= V2 robust +.01, mean changed-F1 noninferior, mean macro >= shuffled-intensity selected/unconstrained best +.005; clean/FPR constraints required',
        negative_control_policy='Shuffled intensity is diagnostic only and can never be selected for deployment',
        fusion_policy='No new blend or parameter search if this ablation fails; retain V2 and preserve all outputs',
        heldout_policy='Only a passing aligned-intensity validation pilot may proceed to frozen fold0 clean/stress comparison; no global five-fold promotion from one fold',
        error_atlas='Clean validation only; four high-confidence FP, four high-confidence FN, two TP and two TN, diversify pair/scan within category; existing bounded 64-point largest-scale patches',
        causality='Local shape and radiometry can suggest error patterns but do not prove occlusion, temporary-object identity or construction-event semantics',
        exposure='Repeated same-site validation development after previously inspected tests; no untouched benchmark claim',
        support='Core intensity adds no geometric support radius; previous feature guard remains applicable',
        input_manifest_sha256=digest(out/'input_manifest.json'),source_sha256=digest(__file__),
        data_policy='BikeParkingConstruction only; no new downloads',reports_or_presentations=False))
    write_json(out/'execution_state.json',dict(stage='prepared',complete=False))
    print('Prepared intensity ablation; RGB/poses/calibration absent from downloaded pair archives',flush=True)


def extract(out=DEFAULT):
    out=Path(out)
    if (out/'feature_complete.json').exists():return
    plan=json.loads((out/'design.json').read_text());assert plan['source_sha256']==digest(__file__)
    raw_records={(int(pair),scan):PlyData.read(path,mmap='r')['params'].data
                 for pair in PAIRS for scan,path in enumerate(discover_pair(RAW,pair))}
    hashes={}
    for split in ('train',*[f'validation_{view}' for view in VIEWS]):
        bank=feature_bank(CONTROL,split);n=len(bank['labels']);dst=out/'features'/split;dst.mkdir(parents=True,exist_ok=True)
        matrix=np.lib.format.open_memmap(dst/'features.npy',mode='w+',dtype=np.float32,shape=(n,382))
        for begin in range(0,n,4096):matrix[begin:begin+4096,:381]=bank['features'][begin:begin+4096]
        for pair in np.unique(bank['pair']):
            for scan in (0,1):
                ids=np.flatnonzero((bank['pair']==pair)&(bank['scan']==scan))
                matrix[ids,381]=gather_intensity(raw_records[(int(pair),scan)],bank['indices'][ids])
        np.testing.assert_array_equal(matrix[:,:381],bank['features']);matrix.flush()
        seed=plan['shuffle_seed_train'] if split=='train' else plan['shuffle_seed_validation']
        np.save(dst/'shuffled_intensity.npy',shuffled_within_cloud(matrix[:,381],bank['pair'],bank['scan'],seed))
        for field in FIELDS:np.save(dst/f'{field}.npy',bank[field])
        del matrix,bank;gc.collect()
        hashes[split]={p.name:digest(p) for p in dst.glob('*.npy')}
        print('Mapped original intensity',split,n,'rows',flush=True)
    write_json(out/'feature_complete.json',dict(passed=True,hashes=hashes,original_381_features_preserved=True,
        new_feature_is_original_core_intensity_only=True,original_metadata_preserved=True))


def diagnostic(out=DEFAULT):
    out=Path(out)
    if (out/'validation_diagnosis.json').exists():return
    bank=feature_bank(CONTROL,'validation_clean');selection=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    folder=BASE/'fold_0/learning'/(selection['primary_variant']+'_seed42');meta=json.loads((folder/'frozen.json').read_text())
    with np.load(folder/'validation_clean.npz') as z:probability=z['probability']
    for field in FIELDS:
        with np.load(folder/'validation_clean.npz') as z:np.testing.assert_array_equal(bank[field],z[field])
    original=BASE/'fold_0/assembled/validation_clean'
    x=np.load(original/'x.npy',mmap_mode='r');mask=np.load(original/'mask.npy',mmap_mode='r')
    surface=local_surface(x[:,2,0,:,:3],mask[:,2,0])
    intensity=np.load(out/'features/validation_clean/features.npy',mmap_mode='r')[:,381]*255.
    distance=bank['features'][:,50]*.25;prediction=probability>=meta['threshold'];labels=bank['labels']
    outcomes=np.where(labels==1,np.where(prediction,'TP','FN'),np.where(prediction,'FP','TN'))
    rows=[]
    for pair in np.unique(bank['pair']):
        for scan in (0,1):
            for outcome in ('FP','FN','TP','TN'):
                chosen=bank['available']&(bank['pair']==pair)&(bank['scan']==scan)&(outcomes==outcome)
                n=int(chosen.sum())
                rows.append(dict(pair=f'{int(pair):02d}',scan=scan,outcome=outcome,count=n,
                    median_intensity=float(np.median(intensity[chosen])) if n else None,
                    median_nn_gap_m=float(np.median(distance[chosen])) if n else None,
                    gap_at_least_8cm=int((distance[chosen]>=.08).sum()),
                    local_shape_valid=int(surface['valid'][chosen].sum()),
                    locally_horizontal_planar=int(surface['locally_horizontal_planar'][chosen].sum()),
                    median_opposite_patch_points=float(np.median(mask[chosen,2,1].sum(-1))) if n else None))
    write_csv(out/'validation_error_attributes.csv',rows)
    selected=[]
    # Confidence-based examples are explicitly illustrative, not a prevalence estimate.
    for outcome,count,descending in [('FP',4,True),('FN',4,False),('TP',2,True),('TN',2,False)]:
        candidates=np.flatnonzero(bank['available']&(outcomes==outcome))
        order=np.lexsort((candidates, -probability[candidates] if descending else probability[candidates]))
        seen=set()
        for row in candidates[order]:
            key=(int(bank['pair'][row]),int(bank['scan'][row]))
            if key in seen:continue
            seen.add(key);selected.append(dict(row=int(row),pair=f'{key[0]:02d}',scan=key[1],outcome=outcome,
                original_index=int(bank['indices'][row]),model_score=float(probability[row]),threshold=meta['threshold'],
                label=int(labels[row]),prediction=int(prediction[row]),intensity=float(intensity[row]),nn_gap_m=float(distance[row]),
                locally_horizontal_planar=bool(surface['locally_horizontal_planar'][row])))
            if len(seen)==count:break
    write_csv(out/'validation_error_examples.csv',selected)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,11))
    for index,example in enumerate(selected):
        row=example['row'];ax=fig.add_subplot(3,4,index+1,projection='3d')
        for side,color,name in ((0,'#2563eb','Core scan'),(1,'#f59e0b','Opposite scan')):
            points=x[row,2,side,mask[row,2,side],:3]*.8
            ax.scatter(*points.T,c=color,s=8,alpha=.8,label=name,depthshade=False)
        ax.scatter([0],[0],[0],c='#dc2626',s=35,marker='x',label='Evaluated core')
        ax.set_xlim(-.85,.85);ax.set_ylim(-.85,.85);ax.set_zlim(-.85,.85);ax.set_box_aspect((1,1,1));ax.view_init(25,-55)
        ax.set_xticks([-.8,0,.8]);ax.set_yticks([-.8,0,.8]);ax.set_zticks([-.8,0,.8]);ax.tick_params(labelsize=6,pad=-1)
        ax.set_title(f"{example['outcome']} | pair {example['pair']} scan {example['scan']}\nscore {example['model_score']:.3f} | gap {example['nn_gap_m']:.3f} m | I {example['intensity']:.0f}",fontsize=8,pad=1)
        if index==0:ax.legend(fontsize=6,loc='lower left')
    fig.suptitle('Construction validation error atlas | original primary V2 decisions\nLargest-scale sampled support centered on each core; no object, material, or occlusion annotations',fontsize=12,y=.99)
    fig.subplots_adjust(left=.02,right=.97,bottom=.03,top=.89,wspace=.05,hspace=.3)
    fig.savefig(out/'validation_error_atlas.png',dpi=170);plt.close(fig)
    summary={}
    for outcome in ('FP','FN','TP','TN'):
        chosen=bank['available']&(outcomes==outcome);n=int(chosen.sum())
        summary[outcome]=dict(observations=n,median_intensity=float(np.median(intensity[chosen])) if n else None,
            locally_horizontal_planar=int(surface['locally_horizontal_planar'][chosen].sum()),
            gap_at_least_8cm=int((distance[chosen]>=.08).sum()))
    write_json(out/'validation_diagnosis.json',dict(scope='Fold0 clean validation, original primary model',outcomes=summary,
        examples=len(selected),example_selection='Confidence extremes, diversify pair/scan; illustrative, not randomly sampled',
        shape_definition='At least three noncollinear source support points; (middle-smallest)/largest covariance eigenvalue >=.5 and |normal_z|>=.9',
        limitations='Horizontal support may include ground, roofs, or other surfaces. Neither intensity nor patch shape verifies object identity, occlusion, or construction activity.'))
    print('Validation error atlas exported',len(selected),'examples',flush=True)


def train(out=DEFAULT,candidate=CANDIDATES[0]):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    out=Path(out);dst=out/'models'/candidate;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'fit_complete.json').exists():return
    plan=json.loads((out/'design.json').read_text());assert candidate in plan['candidates']
    folder=out/'features/train';matrix=np.load(folder/'features.npy',mmap_mode='r')
    labels=np.load(folder/'labels.npy');pairs=np.load(folder/'pair.npy')
    features=matrix if candidate=='aligned_intensity' else np.array(matrix)
    if candidate=='shuffled_intensity':features[:,381]=np.load(folder/'shuffled_intensity.npy')
    assert len(labels)==plan['training_rows'] and np.load(folder/'available.npy').all()
    weights=balanced_training_weights(labels,pairs);model=HistGradientBoostingClassifier(**plan['estimator'])
    print('Intensity fit',candidate,len(labels),'rows',flush=True)
    with MemoryTimer() as timer:model.fit(features,labels,sample_weight=weights)
    assert model.n_iter_==200
    joblib.dump(model,dst/'model.joblib');hashes={}
    for view in VIEWS:
        folder=out/f'features/validation_{view}';val=np.load(folder/'features.npy',mmap_mode='r')
        f=val if candidate=='aligned_intensity' else np.array(val)
        if candidate=='shuffled_intensity':f[:,381]=np.load(folder/'shuffled_intensity.npy')
        p=model.predict_proba(f)[:,1]
        np.savez_compressed(dst/f'validation_{view}.npz',probability=p,**{k:np.load(folder/f'{k}.npy') for k in FIELDS})
        hashes[view]=digest(dst/f'validation_{view}.npz')
    write_json(dst/'fit_complete.json',dict(passed=True,candidate=candidate,checkpoint_sha256=digest(dst/'model.joblib'),
        probability_sha256=hashes,iterations=model.n_iter_,columns=382,timing=timer.stats,test_accessed=False,
        design_sha256=digest(out/'design.json')))
    print('Intensity fit complete',candidate,round(timer.stats['seconds'],1),'seconds',flush=True)


def analyse(out=DEFAULT):
    out=Path(out)
    if (out/'frozen_selection.json').exists():return
    plan=json.loads((out/'design.json').read_text());reference=Path(plan['reference'])
    meta=json.loads((reference/'frozen.json').read_text());reference_views={}
    for view in VIEWS:
        with np.load(reference/f'validation_{view}.npz') as z:reference_views[view]=summarize(z,(z['probability']>=meta['threshold']).astype(np.int8))[0]
    ref=summarize_views(reference_views);rows=[];directions=[];choices={}
    for candidate,folder in [('cached_geometry_control',CONTROL/'models/fresh_match_distributions'),
                             *[(c,out/'models'/c) for c in CANDIDATES]]:
        data={}
        for view in VIEWS:
            with np.load(folder/f'validation_{view}.npz') as z:data[view]={k:z[k] for k in z.files}
            for field in FIELDS:np.testing.assert_array_equal(data[view][field],np.load(out/f'features/validation_{view}/{field}.npy'))
        local=[]
        for threshold in plan['thresholds']:
            metrics={}
            for view in VIEWS:
                metrics[view],groups=summarize(data[view],(data[view]['probability']>=threshold).astype(np.int8))
                directions.extend(dict(candidate=candidate,threshold=threshold,view=view,**r) for r in groups)
            row=dict(candidate=candidate,threshold=threshold,**summarize_views(metrics))
            row['eligible']=(row['clean_binary_macro_f1']>=ref['clean_binary_macro_f1']-.005-1e-12
                             and row['clean_false_positive_rate']<=ref['clean_false_positive_rate']+.005+1e-12)
            rows.append(row);local.append(row)
        rank=lambda r:(r['mean_binary_macro_f1'],r['mean_changed_f1'],r['clean_binary_macro_f1'],r['threshold'])
        valid=[r for r in local if r['eligible']]
        choices[candidate]=dict(selected=max(valid,key=rank) if valid else None,unconstrained_diagnostic=max(local,key=rank))
    selected=choices['aligned_intensity']['selected']
    shuffled=choices['shuffled_intensity']['selected'] or choices['shuffled_intensity']['unconstrained_diagnostic']
    passed=(selected is not None and selected['mean_binary_macro_f1']>=ref['mean_binary_macro_f1']+.01-1e-12
            and selected['mean_changed_f1']>=ref['mean_changed_f1']-1e-12
            and selected['mean_binary_macro_f1']>=shuffled['mean_binary_macro_f1']+.005-1e-12)
    write_csv(out/'validation_thresholds.csv',rows);write_csv(out/'validation_per_direction.csv',directions)
    write_json(out/'frozen_selection.json',dict(pilot_gate_passed=bool(passed),reference=ref,choices=choices,
        selected=selected,negative_control_never_deployable=True,source_control_reused_without_refitting=True,
        frozen_before_any_new_heldout_evaluation=True,new_test_cases_scored=0,design_sha256=digest(out/'design.json')))
    write_json(out/'execution_state.json',dict(stage='validation_frozen',pilot_gate_passed=bool(passed),new_test_cases_scored=0,complete=False))
    print('Intensity pilot gate',passed,flush=True)
    for candidate,choice in choices.items():
        row=choice['selected'] or choice['unconstrained_diagnostic']
        print(candidate,'eligible',choice['selected'] is not None,'clean',row['clean_binary_macro_f1'],'mean',row['mean_binary_macro_f1'],'threshold',row['threshold'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','extract','diagnostic','train','analyse'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--candidate',choices=CANDIDATES,default=CANDIDATES[0]);a=p.parse_args()
    if a.stage=='train':train(a.out,a.candidate)
    else:{'prepare':prepare,'extract':extract,'diagnostic':diagnostic,'analyse':analyse}[a.stage](a.out)
