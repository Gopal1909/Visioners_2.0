"""Independent saved-output audits for the construction evidence follow-ups."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from learning_v3_control import ROOT, BASE, digest, write_json
from construction_evidence_followup import DEFAULT as EVIDENCE
from construction_match_distributions import DEFAULT as DISTRIBUTIONS
from construction_evidence_fusion import DEFAULT as FUSION


def read_rows(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_manifest(path):
    values=json.loads(Path(path).read_text())
    for filename,sha in values.items():
        assert digest(filename)==sha,filename
    return len(values)


def independent_metrics(rows):
    """Derive scalar scores directly from saved directional confusion counts."""
    sums={key:sum(int(r[key]) for r in rows) for key in ('tp','fp','fn','tn')}
    tp,fp,fn,tn=(sums[k] for k in ('tp','fp','fn','tn'))
    changed=[];macro=[]
    for row in rows:
        a,b,c,d=(int(row[k]) for k in ('tp','fp','fn','tn'))
        f1=2*a/max(1,2*a+b+c);f0=2*d/max(1,2*d+b+c)
        changed.append(f1);macro.append((f1+f0)/2)
    return dict(binary_macro_f1=float(np.mean(macro)),changed_f1=float(np.mean(changed)),
                changed_precision=tp/max(1,tp+fp),changed_recall=tp/max(1,tp+fn),
                false_positive_rate=fp/max(1,fp+tn),available_cores=sum(int(r['cores']) for r in rows),**sums)


def verify_evidence(out=EVIDENCE):
    out=Path(out)
    if (out/'verification.json').exists():return
    frozen=json.loads((out/'frozen_selection.json').read_text())
    inputs=verify_manifest(out/'input_manifest.json');protected=verify_manifest(out/'protected_manifest.json')
    assert frozen['design_sha256']==digest(out/'design.json')
    assert frozen['candidate_summary_sha256']==digest(out/'validation_candidate_summary.csv')
    assert frozen['frozen_before_any_new_test_application']
    rows=read_rows(out/'validation_per_direction.csv');groups=defaultdict(list)
    for row in rows:groups[(int(row['fold']),row['view'],row['candidate'])].append(row)
    calculated={key:independent_metrics(part) for key,part in groups.items()}
    folds=read_rows(out/'validation_per_fold.csv')
    for row in folds:
        for view in ('clean','mild','medium'):
            metrics=calculated[(int(row['fold']),view,row['name'])]
            assert len(groups[(int(row['fold']),view,row['name'])])==12
            for key,value in metrics.items():
                assert np.isclose(float(row[f'{view}_{key}']),value,atol=1e-12,rtol=0),(row['name'],key)
    summaries=read_rows(out/'validation_candidate_summary.csv')
    for row in summaries:
        part=[r for r in folds if r['name']==row['name']]
        assert len(part)==5
        for key in row:
            if key.startswith(('clean_','mild_','medium_','mean_')):
                assert np.isclose(float(row[key]),np.mean([float(r[key]) for r in part]),atol=1e-12,rtol=0),key
    by_name={r['name']:r for r in summaries}
    for lane,choice in frozen['choices'].items():
        assert not choice['promoted']
        assert choice['reference']==choice['selected']
        r=by_name[choice['selected']['name']]
        for key,value in choice['selected'].items():
            if key.startswith(('clean_','mild_','medium_','mean_')):assert np.isclose(float(r[key]),value,atol=1e-12,rtol=0)
    slices=read_rows(out/'validation_error_slices.csv')
    fp=defaultdict(int)
    for row in slices:
        if row['model']=='primary' and row['view']=='clean' and row['outcome']=='FP':
            fp[row['distance_bin']]+=int(row['count'])
    total=sum(fp.values());near=sum(fp[key] for key in ('0 to 0.01 m','0.01 to 0.02 m','0.02 to 0.04 m'))
    diagnostics=dict(primary_clean_validation_false_positive_observations=total,
        within_4cm=near,within_4cm_fraction=near/max(1,total),at_least_8cm=fp['0.08 to 0.25 m']+fp['0.25 to inf m'],
        distance_bin_counts=dict(fp),scope='Validation point observations across five overlapping-development folds; not independent objects',
        causality='NN gaps alone cannot distinguish actual changes from occlusion, incomplete coverage, registration errors, or label semantics')
    write_json(out/'diagnosis.json',diagnostics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    labels=['<1 cm','1–2 cm','2–4 cm','4–8 cm','8–25 cm','≥25 cm']
    axes[0].bar(labels,list(fp.values()),color=['#60a5fa']*3+['#f59e0b']*3)
    axes[0].set_ylabel('False-positive validation observations')
    axes[0].set_title('Most false alarms have a substantial NN gap')
    axes[0].tick_params(axis='x',rotation=25)
    for index,value in enumerate(fp.values()):axes[0].text(index,value+15,str(value),ha='center',fontsize=9)
    names=['primary_d000mm','robust_d000mm','agreement_d000mm','mean_margin_d000mm']
    for name,label in zip(names,['Original primary','Original robust','Model agreement','Mean decision margin']):
        r=by_name[name]
        axes[1].scatter(float(r['clean_changed_recall']),float(r['clean_changed_precision']),s=60,label=label)
    axes[1].set_xlabel('Mean-fold clean validation changed recall')
    axes[1].set_ylabel('Mean-fold clean validation changed precision')
    axes[1].set_title('Agreement trades recall for precision')
    axes[1].legend(fontsize=8,loc='best');axes[1].grid(alpha=.2)
    fig.suptitle('Construction evidence diagnostic | validation only; original point masks preserved',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'validation_diagnosis.png',dpi=170);plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,validation_direction_rows=len(rows),candidate_summary_rows=len(summaries),
        original_input_files_preserved=inputs,previous_sources_and_result_receipts_preserved=protected,
        independent_confusion_to_score_recomputation=True,no_test_scoring=True,clean_gate_passed=False,robust_gate_passed=False,
        existing_v2_retained=True,design_sha256=digest(out/'design.json'),frozen_selection_sha256=digest(out/'frozen_selection.json')))
    write_json(out/'execution_state.json',dict(stage='complete_no_promotion',complete=True,verification='verification.json',new_test_cases_scored=0))
    print('Evidence-follow-up independently verified; originals retained',flush=True)


def verify_distributions(out=DISTRIBUTIONS):
    import joblib
    out=Path(out)
    if (out/'verification.json').exists():return
    plan=json.loads((out/'design.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    inputs=verify_manifest(out/'input_manifest.json')
    assert frozen['design_sha256']==digest(out/'design.json')
    for name,sha in plan['source_sha256'].items():
        assert digest(out/'sources'/name)==sha and digest(ROOT/'code'/name)==sha
    receipt=json.loads((out/'feature_complete.json').read_text())
    for split,hashes in receipt['hashes'].items():
        for name,sha in hashes.items():assert digest(out/'features'/split/name)==sha
    # Reproduce features for selected rows via an independent NumPy quantile/CDF
    # implementation, rather than only reusing the descriptor function.
    source=ROOT/'results/learning_v3_construction_2026-10-25_v1/fold_0/epochs/epoch_00/assembled'
    x=np.load(source/'x.npy',mmap_mode='r');mask=np.load(source/'mask.npy',mmap_mode='r')
    exported=np.load(out/'features/train/features.npy',mmap_mode='r');probes=0
    for core in (0,57,1299,17999):
        all_descriptors=np.zeros((3,2,18),np.float32)
        for scale in range(3):
            for side in (0,1):
                local=x[core,scale,side,mask[core,scale,side]].astype(np.float64)
                if not len(local):continue
                distances=local[:,6];matched=local[:,7]>.5
                descriptor=[len(local)/64,float(np.mean(distances)),float(np.std(distances)),
                    *np.quantile(distances,[.1,.25,.5,.75,.9]).tolist(),float(np.mean(~matched))]
                descriptor.extend(float(np.mean(matched&(distances<float(np.float32(t/.25))))) for t in (.01,.02,.04,.08,.16,.25))
                descriptor.extend(np.abs(local[:,3:6]).mean(0).tolist())
                all_descriptors[scale,side]=descriptor;probes+=1
        expected=np.concatenate((all_descriptors.ravel(),(all_descriptors[:,1]-all_descriptors[:,0]).ravel(),
                                 np.abs(all_descriptors[:,1]-all_descriptors[:,0]).ravel()))
        np.testing.assert_allclose(exported[core,165:],expected,atol=1.5e-7,rtol=1e-6)
    direction=read_rows(out/'validation_per_direction.csv');groups=defaultdict(list)
    for row in direction:groups[(row['candidate'],float(row['threshold']),row['view'])].append(row)
    metrics={key:independent_metrics(part) for key,part in groups.items()}
    thresholds=read_rows(out/'validation_thresholds.csv')
    for row in thresholds:
        for view in ('clean','mild','medium'):
            for key,value in metrics[(row['candidate'],float(row['threshold']),view)].items():
                assert np.isclose(float(row[f'{view}_{key}']),value,atol=1e-12,rtol=0)
    probability_probes=0
    for candidate in plan['candidates']:
        folder=out/'models'/candidate;fit=json.loads((folder/'fit_complete.json').read_text())
        assert digest(folder/'model.joblib')==fit['checkpoint_sha256']
        model=joblib.load(folder/'model.joblib')
        assert model.n_iter_==200
        for view in ('clean','mild','medium'):
            p=folder/f'validation_{view}.npz';assert digest(p)==fit['probability_sha256'][view]
            val=out/f'features/validation_{view}'
            with np.load(p) as z:
                for field in ('labels','indices','available','pair','scan'):
                    np.testing.assert_array_equal(z[field],np.load(val/f'{field}.npy',mmap_mode='r'))
                ids=np.linspace(0,len(z['probability'])-1,32,dtype=int)
                features=np.load(val/'features.npy',mmap_mode='r')
                actual=model.predict_proba(features[ids,:fit['columns']])[:,1]
                np.testing.assert_array_equal(actual,z['probability'][ids]);probability_probes+=len(ids)
    ref=frozen['reference']
    selected=frozen['selected']
    passed=(selected is not None and selected['clean_binary_macro_f1']>=ref['clean_binary_macro_f1']-.005-1e-12
            and selected['clean_false_positive_rate']<=ref['clean_false_positive_rate']+.005+1e-12
            and selected['mean_binary_macro_f1']>=ref['mean_binary_macro_f1']+.01-1e-12
            and selected['mean_changed_f1']>=ref['mean_changed_f1']-1e-12)
    assert bool(passed)==frozen['pilot_gate_passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    views=('clean','mild','medium');positions=np.arange(3);width=.25
    displayed=[('V2 robust reference',ref,True)]
    for candidate in frozen['candidates']:
        displayed.append((candidate['candidate'].replace('fresh_','Fresh ').replace('_',' '),
                          candidate['selected'] or candidate['unconstrained_diagnostic'],candidate['eligible']))
    colors=('#2563eb','#9ca3af','#f59e0b')
    for index,(name,row,is_eligible) in enumerate(displayed):
        suffix='' if is_eligible else ' (fails guards)'
        axes[0].bar(positions+(index-1)*width,[row[f'{v}_binary_macro_f1'] for v in views],width,
                    color=colors[index],label=name+suffix,hatch='' if is_eligible else '//')
        axes[1].bar(positions+(index-1)*width,[row[f'{v}_changed_f1'] for v in views],width,
                    color=colors[index],hatch='' if is_eligible else '//')
    for ax,title in zip(axes,['Binary macro F1','Mean directional changed F1']):
        ax.set_xticks(positions,views);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Fresh construction match-distribution pilot | fold 0 validation only\nSame training observations and fixed fit budget; one threshold shared across conditions',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'validation_comparison.png',dpi=170);plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,original_input_files_preserved=inputs,
        validation_threshold_rows_independently_recomputed=len(thresholds),model_probability_probes=probability_probes,
        independent_descriptor_neighborhood_probes=probes,original_validation_ids_labels_availability_preserved=True,
        fixed_200_iteration_budget_verified=True,pilot_gate_passed=bool(passed),new_test_cases_scored=0,
        design_sha256=digest(out/'design.json'),frozen_selection_sha256=digest(out/'frozen_selection.json')))
    print('Match-distribution outputs independently verified',flush=True)


def verify_fusion(out=FUSION):
    out=Path(out)
    if (out/'verification.json').exists():return
    plan=json.loads((out/'design.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    inputs=verify_manifest(out/'input_manifest.json')
    assert digest(ROOT/'code/construction_evidence_fusion.py')==plan['source_sha256']
    assert digest(out/'design.json')==frozen['design_sha256']
    assert digest(out/'validation_grid.csv')==frozen['validation_grid_sha256']
    groups=defaultdict(list)
    for row in read_rows(out/'validation_per_direction.csv'):
        groups[(float(row['weight']),float(row['threshold']),row['view'])].append(row)
    independent={key:independent_metrics(value) for key,value in groups.items()}
    rows=read_rows(out/'validation_grid.csv')
    reference=frozen['reference']
    eligible=[]
    for row in rows:
        for view in ('clean','mild','medium'):
            for key,value in independent[(float(row['weight']),float(row['threshold']),view)].items():
                assert np.isclose(float(row[f'{view}_{key}']),value,atol=1e-12,rtol=0)
        for key in ('binary_macro_f1','changed_f1','changed_precision','changed_recall','false_positive_rate'):
            assert np.isclose(float(row['mean_'+key]),np.mean([float(row[v+'_'+key]) for v in ('clean','mild','medium')]),atol=1e-12,rtol=0)
        is_eligible=(float(row['clean_binary_macro_f1'])>=reference['clean_binary_macro_f1']-.005-1e-12
                     and float(row['clean_false_positive_rate'])<=reference['clean_false_positive_rate']+.005+1e-12)
        assert is_eligible==(row['eligible']=='True')
        if is_eligible:eligible.append(row)
    selected=max(eligible,key=lambda r:(float(r['mean_binary_macro_f1']),float(r['mean_changed_f1']),
        float(r['clean_binary_macro_f1']),-float(r['weight']),float(r['threshold'])))
    for key,value in frozen['selected'].items():
        if key!='eligible':assert np.isclose(float(selected[key]),value,atol=1e-12,rtol=0)
    passed=(float(selected['mean_binary_macro_f1'])>=reference['mean_binary_macro_f1']+.01-1e-12
            and float(selected['mean_changed_f1'])>=reference['mean_changed_f1']-1e-12)
    assert bool(passed)==frozen['pilot_gate_passed'] and not passed
    # Independently recreate the selected validation decisions from the saved
    # source probabilities; do not use the blend helper for these probes.
    for view in ('clean','mild','medium'):
        with np.load(Path(plan['neural'])/f'validation_{view}.npz') as a, np.load(Path(plan['geometry'])/f'validation_{view}.npz') as b:
            for field in ('labels','indices','available','pair','scan'):np.testing.assert_array_equal(a[field],b[field])
            scores=a['probability'].astype(np.float64)*(1-float(selected['weight']))+b['probability'].astype(np.float64)*float(selected['weight'])
            decisions=scores>=float(selected['threshold'])
            keep=a['available']
            truth=a['labels']
            for key,count in [('tp',np.sum(keep&(truth==1)&decisions)),('fp',np.sum(keep&(truth==0)&decisions)),
                              ('fn',np.sum(keep&(truth==1)&~decisions)),('tn',np.sum(keep&(truth==0)&~decisions))]:
                assert int(count)==int(float(selected[f'{view}_{key}']))
    gain=float(selected['mean_binary_macro_f1'])-reference['mean_binary_macro_f1']
    write_json(out/'verification.json',dict(passed=True,original_input_files_preserved=inputs,
        validation_grid_rows_independently_recomputed=len(rows),source_probability_confusions_reproduced=True,
        mean_validation_macro_gain=gain,required_gain=.01,pilot_gate_passed=False,gate_not_relaxed=True,
        new_test_cases_scored=0,original_v2_retained=True,design_sha256=digest(out/'design.json'),
        frozen_selection_sha256=digest(out/'frozen_selection.json')))
    write_json(out/'execution_state.json',dict(stage='complete_no_promotion',complete=True,new_test_cases_scored=0,verification='verification.json'))
    print('Frozen fusion independently verified; gain',gain,'below required .01',flush=True)


def finalize(out=FUSION):
    from datetime import datetime, timezone
    import shutil
    from learning_v3_audit import preserved
    out=Path(out)
    if (out/'final_review.json').exists():return
    tests=(out/'postflight_tests.log').read_text(encoding='utf-8-sig')
    assert 'Ran 80 tests' in tests and '\nOK' in tests
    receipts={}
    for folder in (EVIDENCE,DISTRIBUTIONS,FUSION):
        path=folder/'verification.json';assert json.loads(path.read_text())['passed']
        receipts[str(path)]=digest(path)
    assert not json.loads((DISTRIBUTIONS/'frozen_selection.json').read_text())['pilot_gate_passed']
    fusion=json.loads((FUSION/'frozen_selection.json').read_text())
    assert not fusion['pilot_gate_passed']
    verify_manifest(EVIDENCE/'protected_manifest.json')
    preservation=preserved(ROOT/'results/learning_v3_construction_2026-10-25_v1')
    snapshot=out/'final_sources';snapshot.mkdir(exist_ok=True)
    manifest={}
    for source in (ROOT/'code').glob('*.py'):
        destination=snapshot/source.name
        if destination.exists():assert digest(source)==digest(destination)
        else:shutil.copy2(source,destination)
        manifest[source.name]=digest(destination)
    write_json(snapshot/'manifest.json',manifest)
    model_selection=json.loads((DISTRIBUTIONS/'frozen_selection.json').read_text())
    scores=[dict(method='existing_v2_robust_reference',clean_macro=fusion['reference']['clean_binary_macro_f1'],
                 mean_three_view_macro=fusion['reference']['mean_binary_macro_f1'],eligible=True)]
    for row in model_selection['candidates']:
        selected=row['selected'] or row['unconstrained_diagnostic']
        scores.append(dict(method=row['candidate'],clean_macro=selected['clean_binary_macro_f1'],
            mean_three_view_macro=selected['mean_binary_macro_f1'],eligible=row['eligible'],
            interpretation='Unconstrained diagnostic only if eligibility is false'))
    scores.append(dict(method='neural_geometry_fusion',clean_macro=fusion['selected']['clean_binary_macro_f1'],
        mean_three_view_macro=fusion['selected']['mean_binary_macro_f1'],eligible=True,
        required_gain=.01,actual_gain=fusion['selected']['mean_binary_macro_f1']-fusion['reference']['mean_binary_macro_f1'],
        weight=fusion['selected']['weight'],threshold=fusion['selected']['threshold'],gate_passed=False))
    for folder in (EVIDENCE,DISTRIBUTIONS):
        state=json.loads((folder/'execution_state.json').read_text())
        state.update(complete=True,status='complete_no_promotion',final_consolidated_review=str(out/'final_review.json'))
        write_json(folder/'execution_state.json',state)
    summary=dict(passed=True,status='complete_no_model_promotion',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        unit_tests_passed=80,postflight_tests_sha256=digest(out/'postflight_tests.log'),new_model_fits=2,
        fit_budget='200 boosting iterations per model, same 288000 cached fresh training observations; no new neural training',
        validation_scope='New learned geometry and fusion: fold0, all six pairs, clean/mild/medium; geometric veto/agreement: all five folds. These are reused same-site development data, not untouched held-out results.',
        scores=scores,new_test_cases_scored=0,retained_workflow=str(BASE),
        diagnostic=json.loads((EVIDENCE/'diagnosis.json').read_text()),
        decision='Simple geometric veto, frozen-model agreement, standalone match distributions and score fusion did not meet predeclared acceptance gates. Retain original v2 workflows and held-out scores.',
        meaningful_complement='Richer match-distribution features improved over the matched fresh-geometry control in validation, and their fusion gave a small validation gain. Neither establishes improved held-out performance.',
        causality='Missing coverage, occlusion, residual registration, and change-label semantics remain hypotheses; NN gaps do not identify the cause.',
        next_research_target='Coverage and visibility evidence, and construction-specific semantics; preserve failed candidates and do not repeat these completed fits or lower gates after inspecting results.',
        verification_receipts=receipts,final_sources_manifest_sha256=digest(snapshot/'manifest.json'),
        visually_inspected_figures=[str(EVIDENCE/'validation_diagnosis.png'),str(DISTRIBUTIONS/'validation_comparison.png')],
        no_new_data_downloads=True,reports_and_presentations_untouched=True,**preservation)
    write_json(out/'final_review.json',summary)
    write_json(out/'execution_state.json',dict(stage='complete_no_promotion',complete=True,new_test_cases_scored=0,
        no_active_workers=True,final_review='final_review.json'))
    write_json(ROOT/'results/active_experiment.json',dict(study=str(out.relative_to(ROOT)),status='complete_no_promotion',
        state=str((out/'execution_state.json').relative_to(ROOT)),verification=str((out/'final_review.json').relative_to(ROOT)),
        retained_best_workflow=str(BASE.relative_to(ROOT)),report_and_presentation_authoring=False))
    print('Final evidence follow-up verified: 80 tests, two fits, no promotion;',preservation,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evidence','distributions','fusion','finalize'])
    args=parser.parse_args()
    {'evidence':verify_evidence,'distributions':verify_distributions,'fusion':verify_fusion,'finalize':finalize}[args.stage]()
