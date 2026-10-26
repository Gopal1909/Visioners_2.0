"""Controlled fresh-data geometry classifier with cross-scan match distributions.

Uses only the previously computed, bounded construction patch caches. Training
cores and augmented views are exactly the same in the two controlled fits.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

from learning_v3_control import ROOT, BASE, digest, write_json
from learning_v2_runtime import THRESHOLDS, MemoryTimer, write_csv
from multipair_metrics import balanced_training_weights
from construction_evidence_followup import VIEWS, FIELDS, summarize, summarize_views

FRESH = ROOT/'results/learning_v3_construction_2026-10-25_v1'
DEFAULT = ROOT/'results/construction_match_distributions_2026-10-26_v1'
MODELS = ('fresh_geometry', 'fresh_match_distributions')
MATCH_CDF_M = (.01,.02,.04,.08,.16,.25)
QUANTILES = (.1,.25,.5,.75,.9)


def distribution_features(x, mask, geo):
    """Permutation-invariant descriptors; invalid padding has no contribution.

    18 descriptors/scale/side: valid fraction, capped NN mean/std, five NN
    quantiles, unmatched fraction, six within-distance match fractions, and
    three mean absolute displacement channels. Contrasts preserve side order.
    Empty supports get zero descriptors. No absolute XYZ, IDs, labels or epochs.
    """
    x=np.asarray(x);mask=np.asarray(mask,bool);geo=np.asarray(geo)
    assert x.shape[:-1]==mask.shape and x.shape[1:3]==(3,2) and x.shape[-1]==8
    assert geo.shape==(len(x),165)
    # Float64 reductions avoid order-sensitive float32 cancellation in contrasts.
    safe=np.where(mask[...,None],x,0.).astype(np.float64)
    assert np.isfinite(safe).all() and np.isfinite(geo).all()
    n=mask.sum(-1);denominator=n.clip(1)
    d=safe[...,6]
    assert ((d>=0)&(d<=1)).all()
    matched=(safe[...,7]>.5)&mask
    mean=d.sum(-1)/denominator
    variance=((d-mean[...,None])**2*mask).sum(-1)/denominator
    ordered=np.sort(np.where(mask,d,np.inf),axis=-1)
    quantiles=[]
    for q in QUANTILES:
        rank=(denominator-1)*q
        lo=np.floor(rank).astype(np.int64);hi=np.ceil(rank).astype(np.int64)
        lower=np.take_along_axis(ordered,lo[...,None],axis=-1)[...,0]
        upper=np.take_along_axis(ordered,hi[...,None],axis=-1)[...,0]
        # Replace all-empty rows before arithmetic; inf*0 would otherwise NaN.
        lower=np.where(n>0,lower,0.);upper=np.where(n>0,upper,0.)
        quantiles.append(lower+(upper-lower)*(rank-lo))
    columns=[n/mask.shape[-1],mean,np.sqrt(variance),*quantiles,
             (mask&~matched).sum(-1)/denominator]
    # Compare in the original channel precision so exact stored boundary values
    # are excluded consistently after the float64 accumulation conversion.
    columns.extend((matched&(d < float(np.asarray(threshold/.25,dtype=x.dtype)))).sum(-1)/denominator
                   for threshold in MATCH_CDF_M)
    columns.extend((np.abs(safe[...,channel]).sum(-1)/denominator) for channel in (3,4,5))
    descriptors=np.stack(columns,-1)
    assert descriptors.shape==(len(x),3,2,18)
    contrast=descriptors[:,:,1]-descriptors[:,:,0]
    result=np.concatenate((geo,descriptors.reshape(len(x),-1),contrast.reshape(len(x),-1),
                           np.abs(contrast).reshape(len(x),-1)),axis=1).astype(np.float32)
    assert result.shape==(len(x),381) and np.isfinite(result).all()
    return result


def array_folder(epoch):
    return FRESH/f'fold_0/epochs/epoch_{epoch:02d}/assembled'


def prepare(out=DEFAULT):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'design.json').exists():return
    assert json.loads((FRESH/'pilot_verification.json').read_text())['passed']
    inputs={}
    for epoch in range(16):
        source=array_folder(epoch)
        assert json.loads((source/'complete.json').read_text())['passed']
        for name in ('complete.json',*[field+'.npy' for field in ('x','mask','geo',*FIELDS)]):
            path=source/name;inputs[str(path)]=digest(path)
    for view in VIEWS:
        source=BASE/f'fold_0/assembled/validation_{view}'
        for name in ('complete.json',*[field+'.npy' for field in ('x','mask','geo',*FIELDS)]):
            path=source/name;inputs[str(path)]=digest(path)
    selection=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    reference=BASE/'fold_0/learning'/(selection['secondary_robust_variant']+'_seed42')
    for name in ('frozen.json',*[f'validation_{view}.npz' for view in VIEWS]):
        path=reference/name;inputs[str(path)]=digest(path)
    write_json(out/'input_manifest.json',inputs)
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for name in ('construction_match_distributions.py','construction_evidence_followup.py','multipair_metrics.py',
                 'learning_v2_runtime.py','learning_v3_control.py'):
        shutil.copy2(ROOT/'code'/name,snapshot/name)
    sources={p.name:digest(p) for p in snapshot.glob('*.py')}
    write_json(out/'design.json',dict(
        purpose='Test whether explicitly summarized cross-scan match distributions reduce false positives beyond fresh geometric moments',
        candidates=list(MODELS),fold=0,seed=42,training_epochs_reused=list(range(16)),
        training_rows=288000,training='Same 16 precomputed fresh training banks for both candidates; no new cloud preprocessing',
        weighting='Equal pair and present-class weights within each pair over training data only',
        features={'fresh_geometry':165,'fresh_match_distributions':381},
        match_cdf_m=list(MATCH_CDF_M),nn_quantiles=list(QUANTILES),
        estimator=dict(max_iter=200,max_leaf_nodes=15,learning_rate=.05,l2_regularization=10.,
                       min_samples_leaf=100,early_stopping=False,random_state=42),
        estimator_family='HistGradientBoostingClassifier; a learned geometry model, not a neural network',
        threshold_grid=THRESHOLDS.tolist(),single_threshold_for_all_views=True,
        threshold_selection='Maximize mean clean/mild/medium validation binary macro F1 subject to clean macro and false-positive guards; tie break changed F1 then clean macro then higher threshold',
        reference=str(reference),
        clean_macro_floor='Original v2 robust-selected clean macro minus .005',
        clean_fpr_ceiling='Original v2 robust-selected clean false-positive rate plus .005',
        pilot_gate='Mean three-view validation macro >= reference plus .01; mean changed-F1 noninferior. Reject if no threshold meets clean/FPR constraints.',
        expanded_policy='Only if the fold0 pilot passes, fit the identical winning design in remaining folds using their existing old training caches; fresh-feature expansion would require a separate protocol.',
        heldout_policy='No new test scoring if pilot fails. Any successful pilot is provisional and needs matched-fold validation before promotion.',
        exposure='Same-site follow-up after inspecting earlier results; validation has been reused, so no untouched-generalization claim.',
        support='All descriptors derived inside the existing bounded per-core support. No neighboring core labels, positions, pair identity or scan epoch features.',
        masks='Original validation availability, IDs, labels and support stay unchanged',
        data='BikeParkingConstruction only; no downloads',reports_or_presentations=False,
        input_manifest_sha256=digest(out/'input_manifest.json'),source_sha256=sources))
    write_json(out/'execution_state.json',dict(stage='prepared',complete=False))
    print('Frozen match-distribution pilot: two fixed-budget CPU fits',flush=True)


def extract(out=DEFAULT):
    out=Path(out)
    if (out/'feature_complete.json').exists():return
    plan=json.loads((out/'design.json').read_text())
    assert digest(__file__)==plan['source_sha256'][Path(__file__).name]
    hashes={};probes=0
    for split,folders in [('train',[array_folder(epoch) for epoch in range(16)]),
                          *[(f'validation_{v}',[BASE/f'fold_0/assembled/validation_{v}']) for v in VIEWS]]:
        destination=out/'features'/split;destination.mkdir(parents=True,exist_ok=True)
        counts=[len(np.load(folder/'labels.npy',mmap_mode='r')) for folder in folders]
        n=sum(counts)
        matrix=np.lib.format.open_memmap(destination/'features.npy',mode='w+',dtype=np.float32,shape=(n,381))
        metadata={}
        offset=0
        for folder,count in zip(folders,counts):
            source={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ('x','mask','geo',*FIELDS)}
            for begin in range(0,count,512):
                end=min(count,begin+512)
                matrix[offset+begin:offset+end]=distribution_features(source['x'][begin:end],source['mask'][begin:end],source['geo'][begin:end])
            for field in FIELDS:
                if field not in metadata:
                    metadata[field]=np.lib.format.open_memmap(destination/f'{field}.npy',mode='w+',dtype=source[field].dtype,shape=(n,))
                metadata[field][offset:offset+count]=source[field]
            # Cached first 165 features must remain bit-identical to the old bank.
            np.testing.assert_array_equal(matrix[offset:offset+count,:165],source['geo'])
            sample=np.linspace(0,count-1,7,dtype=int)
            np.testing.assert_array_equal(matrix[offset+sample],distribution_features(source['x'][sample],source['mask'][sample],source['geo'][sample]))
            probes+=len(sample)
            offset+=count;del source
            print('Extracted match descriptors',split,offset,'/',n,flush=True)
        matrix.flush();del matrix
        for mapping in metadata.values():mapping.flush()
        metadata.clear();del mapping;gc.collect()
        hashes[split]={p.name:digest(p) for p in destination.glob('*.npy')}
    write_json(out/'feature_complete.json',dict(passed=True,hashes=hashes,rows=plan['training_rows'],descriptor_columns=381,
        original_geometry_columns_preserved=True,probe_rows=probes,original_metadata_preserved=True))
    write_json(out/'execution_state.json',dict(stage='features_complete',complete=False))


def feature_bank(out,split):
    folder=Path(out)/'features'/split
    return {k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ('features',*FIELDS)}


def train(out=DEFAULT,candidate=MODELS[0]):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    out=Path(out);dst=out/'models'/candidate;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'fit_complete.json').exists():return
    plan=json.loads((out/'design.json').read_text())
    assert candidate in plan['candidates']
    data=feature_bank(out,'train');assert len(data['labels'])==plan['training_rows']
    assert data['available'].all()
    columns=plan['features'][candidate]
    weights=balanced_training_weights(np.asarray(data['labels']),np.asarray(data['pair']))
    model=HistGradientBoostingClassifier(**plan['estimator'])
    print('Training match pilot',candidate,'rows',len(weights),'columns',columns,flush=True)
    with MemoryTimer() as timer:
        model.fit(data['features'][:,:columns],data['labels'],sample_weight=weights)
    assert model.n_iter_==plan['estimator']['max_iter']
    joblib.dump(model,dst/'model.joblib')
    probability_hashes={}
    for view in VIEWS:
        val=feature_bank(out,f'validation_{view}')
        probability=model.predict_proba(val['features'][:,:columns])[:,1]
        np.savez_compressed(dst/f'validation_{view}.npz',probability=probability,**{k:val[k] for k in FIELDS})
        probability_hashes[view]=digest(dst/f'validation_{view}.npz')
    write_json(dst/'fit_complete.json',dict(passed=True,candidate=candidate,iterations=model.n_iter_,columns=columns,
        checkpoint_sha256=digest(dst/'model.joblib'),probability_sha256=probability_hashes,
        design_sha256=digest(out/'design.json'),timing=timer.stats,training_only_labels_used=True,
        test_accessed=False))
    print('Finished match pilot',candidate,round(timer.stats['seconds'],1),'seconds',flush=True)


def analyse(out=DEFAULT):
    out=Path(out)
    if (out/'frozen_selection.json').exists():return
    plan=json.loads((out/'design.json').read_text())
    reference=Path(plan['reference']);meta=json.loads((reference/'frozen.json').read_text())
    reference_views={}
    for view in VIEWS:
        with np.load(reference/f'validation_{view}.npz') as z:
            reference_views[view]=summarize(z,(z['probability']>=meta['threshold']).astype(np.int8))[0]
    ref=summarize_views(reference_views);rows=[];chosen=[];directions=[]
    for candidate in MODELS:
        folder=out/'models'/candidate;data={}
        receipt=json.loads((folder/'fit_complete.json').read_text())
        assert digest(folder/'model.joblib')==receipt['checkpoint_sha256']
        for view in VIEWS:
            assert digest(folder/f'validation_{view}.npz')==receipt['probability_sha256'][view]
            with np.load(folder/f'validation_{view}.npz') as z:data[view]={k:z[k] for k in z.files}
            bank=feature_bank(out,f'validation_{view}')
            for field in FIELDS:np.testing.assert_array_equal(data[view][field],bank[field])
        local=[]
        for threshold in plan['threshold_grid']:
            scores={}
            for view in VIEWS:
                metrics,per_direction=summarize(data[view],(data[view]['probability']>=threshold).astype(np.int8))
                scores[view]=metrics
                directions.extend(dict(candidate=candidate,threshold=threshold,view=view,**r) for r in per_direction)
            row=dict(candidate=candidate,threshold=threshold,**summarize_views(scores))
            row['eligible']=(row['clean_binary_macro_f1']>=ref['clean_binary_macro_f1']-.005-1e-12
                             and row['clean_false_positive_rate']<=ref['clean_false_positive_rate']+.005+1e-12)
            local.append(row);rows.append(row)
        valid=[r for r in local if r['eligible']]
        selected=max(valid,key=lambda r:(r['mean_binary_macro_f1'],r['mean_changed_f1'],r['clean_binary_macro_f1'],r['threshold'])) if valid else None
        # Also retain unconstrained best as a transparent diagnostic, never deploy it.
        unconstrained=max(local,key=lambda r:(r['mean_binary_macro_f1'],r['mean_changed_f1'],r['threshold']))
        chosen.append(dict(candidate=candidate,eligible=selected is not None,selected=selected,unconstrained_diagnostic=unconstrained))
    valid=[c['selected'] for c in chosen if c['selected'] is not None]
    best=max(valid,key=lambda r:(r['mean_binary_macro_f1'],r['mean_changed_f1'],-MODELS.index(r['candidate']))) if valid else None
    passed=(best is not None and best['mean_binary_macro_f1']>=ref['mean_binary_macro_f1']+.01-1e-12
            and best['mean_changed_f1']>=ref['mean_changed_f1']-1e-12)
    write_csv(out/'validation_thresholds.csv',rows)
    write_csv(out/'validation_per_direction.csv',directions)
    result=dict(pilot_gate_passed=bool(passed),reference=ref,candidates=chosen,selected=best,
        next_stage='provisional_pilot_needs_matched_fold_confirmation' if passed else 'retain_existing_v2',
        new_test_cases_scored=0,thresholds_frozen_before_any_test=True,design_sha256=digest(out/'design.json'),
        source_sha256=plan['source_sha256'])
    write_json(out/'frozen_selection.json',result)
    write_json(out/'execution_state.json',dict(stage='pilot_complete',complete=True,pilot_gate_passed=bool(passed),new_test_cases_scored=0))
    print('Match-distribution pilot gate',passed,flush=True)
    for row in chosen:
        value=row['selected'] or row['unconstrained_diagnostic']
        print(row['candidate'],'eligible',row['eligible'],'clean',value['clean_binary_macro_f1'],'mean',value['mean_binary_macro_f1'],'threshold',value['threshold'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','extract','train','analyse'])
    parser.add_argument('--candidate',choices=MODELS,default=MODELS[0])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.stage=='train':train(args.out,args.candidate)
    else:{'prepare':prepare,'extract':extract,'analyse':analyse}[args.stage](args.out)


if __name__=='__main__':main()
