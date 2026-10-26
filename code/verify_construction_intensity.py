"""Independent row, attribute, score, and artifact checks for intensity ablation."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from learning_v3_control import ROOT, BASE, digest, write_json
from construction_intensity import DEFAULT, RAW, CANDIDATES, PAIRS
from construction_match_distributions import DEFAULT as CONTROL
from construction_evidence_followup import FIELDS, VIEWS
from verify_construction_evidence import independent_metrics, read_rows, verify_manifest


def verify(out=DEFAULT):
    import joblib
    out=Path(out)
    if (out/'verification.json').exists():return
    plan=json.loads((out/'design.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    assert digest(out/'design.json')==frozen['design_sha256']
    assert digest(ROOT/'code/construction_intensity.py')==plan['source_sha256']
    source_files=verify_manifest(out/'input_manifest.json');protected_files=verify_manifest(out/'protected_manifest.json')
    receipt=json.loads((out/'feature_complete.json').read_text())
    raw={}
    for pair in PAIRS:
        for scan in (0,1):
            inventory=json.loads((out/'sensor_inventory.json').read_text())
            item=next(r for r in inventory['clouds'] if r['pair']==pair and r['scan']==scan)
            ply=PlyData.read(item['path'],mmap='r');record=ply['params'].data
            assert record.dtype.names==('x','y','z','intensity','cd_type') and len(ply.elements)==1
            raw[(int(pair),scan)]=record
    attribute_rows=0;shuffle_groups=0
    for split,hashes in receipt['hashes'].items():
        folder=out/'features'/split;previous=CONTROL/'features'/split
        for name,sha in hashes.items():assert digest(folder/name)==sha
        features=np.load(folder/'features.npy',mmap_mode='r');old=np.load(previous/'features.npy',mmap_mode='r')
        np.testing.assert_array_equal(features[:,:381],old)
        metadata={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in FIELDS}
        for field in FIELDS:np.testing.assert_array_equal(metadata[field],np.load(previous/f'{field}.npy',mmap_mode='r'))
        permuted=np.load(folder/'shuffled_intensity.npy',mmap_mode='r')
        seed=plan['shuffle_seed_train'] if split=='train' else plan['shuffle_seed_validation']
        rng=np.random.default_rng(seed)
        for pair in sorted(np.unique(metadata['pair'])):
            for scan in (0,1):
                rows=np.flatnonzero((metadata['pair']==pair)&(metadata['scan']==scan));indices=metadata['indices'][rows]
                # Independent extraction from the PLY attribute, with labels
                # checked separately against their frozen original IDs.
                record=raw[(int(pair),scan)]
                expected=record['intensity'][indices].astype(np.float32)/np.float32(255.)
                np.testing.assert_array_equal(features[rows,381],expected)
                np.testing.assert_array_equal(record['cd_type'][indices],metadata['labels'][rows])
                ordered=rng.permutation(rows)
                np.testing.assert_array_equal(permuted[rows],features[ordered,381])
                np.testing.assert_array_equal(np.sort(permuted[rows]),np.sort(expected))
                attribute_rows+=len(rows);shuffle_groups+=1
    direction_rows=read_rows(out/'validation_per_direction.csv');groups=defaultdict(list)
    for row in direction_rows:groups[(row['candidate'],float(row['threshold']),row['view'])].append(row)
    scores={key:independent_metrics(rows) for key,rows in groups.items()}
    threshold_rows=read_rows(out/'validation_thresholds.csv')
    selected={}
    for candidate in ('cached_geometry_control',*CANDIDATES):
        rows=[r for r in threshold_rows if r['candidate']==candidate]
        for row in rows:
            for view in VIEWS:
                for key,value in scores[(candidate,float(row['threshold']),view)].items():
                    assert np.isclose(float(row[f'{view}_{key}']),value,atol=1e-12,rtol=0)
            for key in ('binary_macro_f1','changed_f1','changed_precision','changed_recall','false_positive_rate'):
                assert np.isclose(float(row['mean_'+key]),np.mean([float(row[f'{v}_{key}']) for v in VIEWS]),atol=1e-12,rtol=0)
            eligible=(float(row['clean_binary_macro_f1'])>=frozen['reference']['clean_binary_macro_f1']-.005-1e-12
                      and float(row['clean_false_positive_rate'])<=frozen['reference']['clean_false_positive_rate']+.005+1e-12)
            assert eligible==(row['eligible']=='True')
        valid=[r for r in rows if r['eligible']=='True']
        rank=lambda r:(float(r['mean_binary_macro_f1']),float(r['mean_changed_f1']),float(r['clean_binary_macro_f1']),float(r['threshold']))
        best=max(valid,key=rank) if valid else None
        saved=frozen['choices'][candidate]['selected']
        assert (best is None)==(saved is None)
        if saved:
            for key,value in saved.items():
                if key not in ('candidate','eligible'):assert np.isclose(float(best[key]),value,atol=1e-12,rtol=0)
        selected[candidate]=best
    ref=frozen['reference'];aligned=frozen['selected']
    shuffled=frozen['choices']['shuffled_intensity']['selected'] or frozen['choices']['shuffled_intensity']['unconstrained_diagnostic']
    passed=(aligned is not None and aligned['mean_binary_macro_f1']>=ref['mean_binary_macro_f1']+.01-1e-12
            and aligned['mean_changed_f1']>=ref['mean_changed_f1']-1e-12
            and aligned['mean_binary_macro_f1']>=shuffled['mean_binary_macro_f1']+.005-1e-12)
    assert bool(passed)==frozen['pilot_gate_passed']
    probability_probes=0
    for candidate in CANDIDATES:
        folder=out/'models'/candidate;fit=json.loads((folder/'fit_complete.json').read_text())
        assert digest(folder/'model.joblib')==fit['checkpoint_sha256']
        model=joblib.load(folder/'model.joblib');assert model.n_iter_==plan['estimator']['max_iter']==200
        for view in VIEWS:
            source=folder/f'validation_{view}.npz';assert digest(source)==fit['probability_sha256'][view]
            features=np.load(out/f'features/validation_{view}/features.npy',mmap_mode='r')
            ids=np.linspace(0,len(features)-1,48,dtype=int);probe=np.array(features[ids])
            if candidate=='shuffled_intensity':probe[:,381]=np.load(out/f'features/validation_{view}/shuffled_intensity.npy',mmap_mode='r')[ids]
            actual=model.predict_proba(probe)[:,1]
            with np.load(source) as z:
                np.testing.assert_array_equal(actual,z['probability'][ids]);probability_probes+=len(ids)
                for field in FIELDS:np.testing.assert_array_equal(z[field],np.load(out/f'features/validation_{view}/{field}.npy',mmap_mode='r'))
    # Independently check example selection labels and recorded original IDs.
    examples=read_rows(out/'validation_error_examples.csv')
    original=BASE/'fold_0/assembled/validation_clean'
    y=np.load(original/'labels.npy');pairs=np.load(original/'pair.npy');scans=np.load(original/'scan.npy');indices=np.load(original/'indices.npy')
    selection=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    model_folder=BASE/'fold_0/learning'/(selection['primary_variant']+'_seed42')
    threshold=json.loads((model_folder/'frozen.json').read_text())['threshold']
    with np.load(model_folder/'validation_clean.npz') as z:p=z['probability']
    for example in examples:
        row=int(example['row']);truth=int(y[row]);prediction=int(p[row]>=threshold)
        assert truth==int(example['label']) and prediction==int(example['prediction'])
        assert int(pairs[row])==int(example['pair']) and int(scans[row])==int(example['scan']) and int(indices[row])==int(example['original_index'])
        assert np.isclose(float(example['intensity']),float(raw[(int(pairs[row]),int(scans[row]))]['intensity'][indices[row]]),atol=2e-5,rtol=0)
    # Comparison chart keeps failed eligibility visible instead of silently
    # treating an unconstrained diagnostic as an accepted configuration.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5));positions=np.arange(3);width=.2
    rows=[('V2 robust reference',ref,True)]
    for candidate in ('cached_geometry_control',*CANDIDATES):
        choice=frozen['choices'][candidate]
        rows.append((candidate.replace('_',' '),choice['selected'] or choice['unconstrained_diagnostic'],choice['selected'] is not None))
    for index,(name,row,eligible) in enumerate(rows):
        for ax,metric in zip(axes,('binary_macro_f1','changed_f1')):
            ax.bar(positions+(index-1.5)*width,[row[f'{view}_{metric}'] for view in VIEWS],width,
                label=name+('' if eligible else ' (fails guards)'),hatch='' if eligible else '//')
    for ax,title in zip(axes,('Binary macro F1','Mean directional changed F1')):
        ax.set_xticks(positions,VIEWS);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Construction intensity ablation | fold0 validation only\nSame geometry and training observations; original geometry control reused without refitting',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'validation_comparison.png',dpi=170);plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,original_input_files_preserved=source_files,
        previous_sources_and_results_preserved=protected_files,original_intensity_observations_verified=attribute_rows,
        shuffle_groups_verified=shuffle_groups,original_381_geometry_columns_unchanged=True,
        original_labels_indices_availability_preserved=True,validation_threshold_rows_independently_recomputed=len(threshold_rows),
        model_probability_probes=probability_probes,atlas_example_original_rows_verified=len(examples),
        pilot_gate_passed=bool(passed),source_control_not_refit=True,new_test_cases_scored=0,
        design_sha256=digest(out/'design.json'),frozen_selection_sha256=digest(out/'frozen_selection.json')))
    print('Intensity ablation independently verified; pilot gate',passed,flush=True)


def render_annotated_atlas(out=DEFAULT):
    """Explicitly mark clipped NN gaps; preserve the initial atlas unchanged."""
    out=Path(out);destination=out/'validation_error_atlas_annotated.png'
    if destination.exists():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=BASE/'fold_0/assembled/validation_clean'
    x=np.load(original/'x.npy',mmap_mode='r');mask=np.load(original/'mask.npy',mmap_mode='r')
    examples=read_rows(out/'validation_error_examples.csv');fig=plt.figure(figsize=(16,12))
    handles=None
    for index,example in enumerate(examples):
        row=int(example['row']);ax=fig.add_subplot(3,4,index+1,projection='3d')
        for side,color,label in ((0,'#2563eb','Core scan support'),(1,'#f59e0b','Opposite scan support')):
            points=x[row,2,side,mask[row,2,side],:3]*.8
            ax.scatter(*points.T,c=color,s=9,alpha=.85,label=label,depthshade=False)
        ax.scatter([0],[0],[0],c='#dc2626',s=40,marker='x',label='Evaluated core',depthshade=False)
        gap=float(example['nn_gap_m']);gap_label='≥0.250' if gap>=.25 else f'{gap:.3f}'
        ax.set_title(f"{example['outcome']} | pair {example['pair']} scan {example['scan']}\nscore {float(example['model_score']):.3f} | NN gap {gap_label} m | I {float(example['intensity']):.0f}",fontsize=9,pad=0)
        ax.set_xlim(-.85,.85);ax.set_ylim(-.85,.85);ax.set_zlim(-.85,.85);ax.set_box_aspect((1,1,1));ax.view_init(25,-55)
        ax.set_xticks([-.8,0,.8]);ax.set_yticks([-.8,0,.8]);ax.set_zticks([-.8,0,.8]);ax.tick_params(labelsize=6,pad=0)
        if handles is None:handles=ax.get_legend_handles_labels()
    fig.suptitle('Construction validation error atlas | original V2 primary decisions\nLocal XYZ offsets in metres; NN gaps are capped at 0.25 m. Confidence-extreme examples, diversified by pair/scan.\nSampled voxel support does not establish object identity, material, or occlusion.',fontsize=12,y=.99)
    fig.legend(*handles,loc='lower center',ncol=3,fontsize=9,frameon=False)
    fig.subplots_adjust(left=.015,right=.98,bottom=.06,top=.87,wspace=.08,hspace=.32)
    fig.savefig(destination,dpi=170);plt.close(fig)
    print('Annotated atlas exported; original figure preserved',flush=True)


def finalize(out=DEFAULT):
    import shutil
    from datetime import datetime,timezone
    from learning_v3_audit import preserved
    out=Path(out)
    if (out/'final_review.json').exists():return
    receipt=json.loads((out/'verification.json').read_text());assert receipt['passed']
    frozen=json.loads((out/'frozen_selection.json').read_text());assert not frozen['pilot_gate_passed']
    tests=(out/'postflight_tests.log').read_text(encoding='utf-8-sig');assert 'Ran 83 tests' in tests and '\nOK' in tests
    verify_manifest(out/'protected_manifest.json')
    preservation=preserved(ROOT/'results/learning_v3_construction_2026-10-25_v1')
    snapshot=out/'final_sources';snapshot.mkdir(exist_ok=True);manifest={}
    for source in (ROOT/'code').glob('*.py'):
        destination=snapshot/source.name
        if destination.exists():assert digest(destination)==digest(source)
        else:shutil.copy2(source,destination)
        manifest[source.name]=digest(destination)
    write_json(snapshot/'manifest.json',manifest)
    scores=[]
    for candidate in ('cached_geometry_control',*CANDIDATES):
        choice=frozen['choices'][candidate];row=choice['selected'] or choice['unconstrained_diagnostic']
        scores.append(dict(candidate=candidate,eligible=choice['selected'] is not None,clean_macro=row['clean_binary_macro_f1'],
            mean_three_view_macro=row['mean_binary_macro_f1'],threshold=row['threshold'],
            interpretation='Unconstrained validation diagnostic if eligibility is false; not an accepted deployment'))
    figures={}
    for name in ('validation_error_atlas.png','validation_error_atlas_annotated.png','validation_comparison.png'):
        path=out/name
        if path.exists():figures[name]=digest(path)
    summary=dict(passed=True,status='complete_no_model_promotion',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        sensor_inventory='sensor_inventory.json',available_input='XYZ and variable intensity; cd_type is ground truth only',
        unavailable_in_downloaded_package=['RGB','sensor origins/poses','timestamps','calibration'],
        visibility_limit='No ray-based occlusion/free-space claim is supported by these pair files',
        validation_diagnosis=json.loads((out/'validation_diagnosis.json').read_text()),
        new_model_fits=2,existing_control_refit=False,training_observations=288000,
        model='Fixed-budget learned geometry classifier with one added original-core intensity feature; shuffled intensity is a negative control',
        scores=scores,reference_v2_robust_mean_validation_macro=frozen['reference']['mean_binary_macro_f1'],
        pilot_gate_passed=False,acceptance_guards_not_relaxed=True,new_test_cases_scored=0,
        retained_workflow=str(BASE),retained_main_heldout_binary_macro_f1=.6536591206208738,
        scientific_conclusion='Aligned intensity gives a small validation gain over identical cached geometry and shuffled-intensity controls, but clean quality remains insufficient versus V2. It does not establish improved held-out detection or explain individual error causes.',
        scope='One fold of repeatedly reused same-site validation; six pairs from four shared epochs, not independent construction sites',
        unit_tests_passed=83,postflight_tests_sha256=digest(out/'postflight_tests.log'),
        verification_sha256=digest(out/'verification.json'),final_source_manifest_sha256=digest(snapshot/'manifest.json'),
        figures=figures,no_additional_data_downloads=True,reports_and_presentations_untouched=True,
        continuation='Do not rerun the completed intensity fits or lower the gate. Exact visibility attribution needs additional source metadata; semantics would require explicit construction/work-item annotations.',
        **preservation)
    write_json(out/'final_review.json',summary)
    write_json(out/'execution_state.json',dict(stage='complete_no_promotion',complete=True,no_active_workers=True,
        pilot_gate_passed=False,new_test_cases_scored=0,verification='final_review.json'))
    write_json(ROOT/'results/active_experiment.json',dict(study=str(out.relative_to(ROOT)),status='complete_no_promotion',
        state=str((out/'execution_state.json').relative_to(ROOT)),verification=str((out/'final_review.json').relative_to(ROOT)),
        retained_best_workflow=str(BASE.relative_to(ROOT)),report_and_presentation_authoring=False))
    print('Intensity follow-up complete: 83 tests; V2 retained;',preservation,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',choices=['verify','render-atlas','finalize'],default='verify')
    parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    {'verify':verify,'render-atlas':render_annotated_atlas,'finalize':finalize}[args.stage](args.out)
