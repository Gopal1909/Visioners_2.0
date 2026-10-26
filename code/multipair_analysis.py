"""Result tables, measured robustness figures and integrity checks; no report/deck."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multipair_data import ROOT,DEFAULT,PAIRS
from multipair_metrics import pair_scores,directional_three_class_scores
from m3c2_baseline import digest,write_json


def enrich(frame,root):
    extras=[]
    for key,part in frame.groupby(['experiment','pair','condition','perturbation_seed'],sort=False):
        experiment,pair,condition,seed=key
        folder=Path(root)/f'evaluation/{experiment}/pair_{pair}/{condition}_seed{int(seed)}'
        condition_data=json.loads((folder/'condition.json').read_text())
        with np.load(folder/'predictions.npz') as z:
            for index,row in part.iterrows():
                masks=[np.ones(len(z[f'labels{i}']),bool) for i in (0,1)]
                if row.visibility=='B':masks[0]=np.linalg.norm(z['xyz0'][:,:2]-np.array(condition_data['blind_center_m'])[:2],axis=1)>condition_data['radius_m']
                if row.support=='common_valid':masks=[m&(z[f'm3c2_calibrated_prediction{i}']>=0) for i,m in enumerate(masks)]
                values=directional_three_class_scores([z[f'labels{i}'][m] for i,m in enumerate(masks)],
                    [z[f'{row.method}_prediction{i}'][m] for i,m in enumerate(masks)])
                extras.append(dict(index=index,**values))
    return frame.join(pd.DataFrame(extras).set_index('index'))


def null_controls(root):
    rows=[]
    for path in (Path(root)/'evaluation').glob('*/*/clean_seed42/predictions.npz'):
        with np.load(path) as z:
            labels=[z[f'labels{i}'] for i in (0,1)]
            for value,name in [(0,'always_unchanged'),(1,'always_changed')]:
                rows.append(dict(experiment=path.parents[2].name,pair=path.parents[1].name,method=name,
                    **pair_scores(labels,[np.full(len(y),value,np.int8) for y in labels])))
    return rows


def stress_clean_reference(out):
    """Score clean predictions on the identical 512 original stress cores.

    Using the full 1500 clean cores at zero severity would mix a sampling
    change with the effect of a perturbation in the plotted curves.
    """
    rows=[]
    for pair in PAIRS:
        clean=Path(out)/f'evaluation/primary/pair_{pair}/clean_seed42/predictions.npz'
        stress=clean.parent.parent/'translation_0.05_seed42/geometry_measurements.npz'
        with np.load(clean) as z,np.load(stress) as requested:
            positions=[]
            for scan in (0,1):
                ids=requested[f'indices{scan}'];pos=np.searchsorted(z[f'indices{scan}'],ids)
                np.testing.assert_array_equal(z[f'indices{scan}'][pos],ids)
                positions.append(pos)
            labels=[z[f'labels{i}'][positions[i]] for i in (0,1)]
            for key in z.files:
                if key.endswith('_prediction0'):
                    method=key[:-len('_prediction0')]
                    prediction=[z[f'{method}_prediction{i}'][positions[i]] for i in (0,1)]
                    rows.append(dict(pair=pair,method=method,**pair_scores(labels,prediction)))
    return pd.DataFrame(rows)


def spatial_support_summary(cv):
    """Supplementary confusion pooling, retaining all unchanged-only regions.

    This is a different weighting from the predeclared pair/direction macro
    score. Positive-support F1 is accompanied by all-region false alarms.
    """
    rows=[]
    for method,part in cv.groupby('method'):
        def total(scan,key):return int(part[f'scan{scan}_{key}'].sum())
        tn=sum(total(i,'tn') for i in (0,1))
        unchanged_fn=sum(total(i,'fp')+total(i,'unknown_negative') for i in (0,1))
        unchanged_fp=sum(total(i,'fn')-total(i,'unknown_positive') for i in (0,1))
        f1=[2*tn/max(1,2*tn+unchanged_fn+unchanged_fp)]
        for scan in (0,1):
            tp=total(scan,'tp');f1.append(2*tp/max(1,2*tp+total(scan,'fp')+total(scan,'fn')))
        defined=[];absent_fp=0;absent_n=0
        for scan in (0,1):
            present=part[f'scan{scan}_positives']>0
            defined.extend(part.loc[present,f'scan{scan}_f1_changed'].tolist())
            absent_fp+=int(part.loc[~present,f'scan{scan}_fp'].sum())
            absent_n+=int(part.loc[~present,f'scan{scan}_cores'].sum())
        rows.append(dict(method=method,pooled_out_of_fold_unchanged_f1=f1[0],pooled_out_of_fold_removed_f1=f1[1],
            pooled_out_of_fold_added_f1=f1[2],pooled_out_of_fold_three_class_macro_f1=float(np.mean(f1)),
            positive_support_direction_changed_f1=float(np.mean(defined)),positive_support_directions=len(defined),
            unchanged_only_directions=2*len(part)-len(defined),false_alarms_in_unchanged_only_directions=absent_fp,
            cores_in_unchanged_only_directions=absent_n,unchanged_only_false_alarm_rate=absent_fp/max(1,absent_n)))
    return pd.DataFrame(rows)


def stress_condition_summary(frame):
    rows=[]
    stress=frame[(frame.experiment=='primary')&(frame.condition!='clean')&(frame.support=='full_requested')]
    metrics=('selection_binary_macro_f1','directional_changed_f1','detector_unknown_rate',
        'correct_over_initial_requested_cores','evaluation_coverage_initial_requested_cores')
    for (condition,family,method,visibility),part in stress.groupby(['condition','family','method','visibility']):
        row=dict(condition=condition,family=family,severity=float(part.severity.iloc[0]),method=method,visibility=visibility,
            pair_count=int(part.pair.nunique()),perturbation_seeds=int(part.perturbation_seed.nunique()))
        for metric in metrics:
            seed_means=part.groupby('perturbation_seed')[metric].mean()
            row[metric+'_mean']=float(seed_means.mean())
            row[metric+'_perturbation_seed_sd']=float(seed_means.std())
        rows.append(row)
    return pd.DataFrame(rows)


def summary(out=DEFAULT):
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True)
    frame=pd.read_csv(out/'evaluated_metrics.csv',dtype={'pair':str})
    frame=enrich(frame,out)
    for i in (0,1):
        frame[f'scan{i}_changed_f1_defined']=frame[f'scan{i}_positives']>0
        frame[f'scan{i}_false_positive_rate']=frame[f'scan{i}_fp']/(frame[f'scan{i}_cores']-frame[f'scan{i}_positives']).clip(lower=1)
    defined=np.column_stack([np.where(frame[f'scan{i}_positives']>0,frame[f'scan{i}_f1_changed'],np.nan) for i in (0,1)])
    count=np.isfinite(defined).sum(1)
    frame['positive_support_directions']=count
    frame['changed_f1_positive_support_only']=np.divide(np.nansum(defined,axis=1),count,out=np.full(len(frame),np.nan),where=count>0)
    frame.to_csv(folder/'metrics_with_class_support.csv',index=False)
    stress_condition_summary(frame).to_csv(folder/'stress_condition_summary.csv',index=False)
    clean=frame[(frame.condition=='clean')&(frame.support=='full_requested')].copy()
    clean.to_csv(folder/'clean_per_pair.csv',index=False)
    keys=['experiment','method']
    table=clean.groupby(keys).agg(binary_macro_f1=('selection_binary_macro_f1','mean'),
        changed_f1=('directional_changed_f1','mean'),unknown_rate=('detector_unknown_rate','mean'),
        accuracy=('correct_over_requested','mean'),pair_count=('pair','nunique'),
        changed_f1_positive_directions=('changed_f1_positive_support_only','mean'),
        pooled_three_class_macro_f1=('pooled_three_class_macro_f1','mean'),
        pooled_three_class_miou=('pooled_three_class_miou','mean')).reset_index()
    table.to_csv(folder/'clean_summary.csv',index=False)
    null_rows=null_controls(out)
    if (out/'spatial_cv').exists():
        for root in sorted((out/'spatial_cv').glob('fold_*')):
            null_rows.extend([dict(row,spatial_fold=int(root.name.split('_')[-1])) for row in null_controls(root)])
    pd.DataFrame(null_rows).to_csv(folder/'constant_prediction_controls.csv',index=False)
    print(table.to_string(index=False),flush=True)
    # Paired changes always intersect the same retained original row IDs and
    # clean/current M3C2 support when reporting a common-support delta.
    deltas=[];audited=0
    for experiment in ('primary','temporal'):
        base=out/f'evaluation/{experiment}'
        if not base.exists():continue
        for pair_folder in base.glob('pair_*'):
            clean_file=pair_folder/'clean_seed42/predictions.npz'
            if not clean_file.exists():continue
            with np.load(clean_file) as z:reference={k:z[k] for k in z.files}
            for path in pair_folder.glob('*/predictions.npz'):
                if not (path.parent/'complete.json').exists():continue
                assert digest(path)==json.loads((path.parent/'complete.json').read_text())['measurement_sha256']
                with np.load(path) as z:
                    methods=[k[:-len('_prediction0')] for k in z.files if k.endswith('_prediction0')]
                    for method in methods:
                        if f'{method}_prediction0' not in reference:continue
                        current_labels=[];current_predictions=[];clean_predictions=[]
                        for scan in (0,1):
                            original=z[f'indices{scan}'];positions=np.searchsorted(reference[f'indices{scan}'],original)
                            np.testing.assert_array_equal(reference[f'indices{scan}'][positions],original)
                            np.testing.assert_array_equal(reference[f'labels{scan}'][positions],z[f'labels{scan}'])
                            valid=(reference[f'm3c2_calibrated_prediction{scan}'][positions]>=0)&(z[f'm3c2_calibrated_prediction{scan}']>=0)
                            current_labels.append(z[f'labels{scan}'][valid]);current_predictions.append(z[f'{method}_prediction{scan}'][valid])
                            clean_predictions.append(reference[f'{method}_prediction{scan}'][positions][valid])
                        a=pair_scores(current_labels,clean_predictions);b=pair_scores(current_labels,current_predictions)
                        deltas.append(dict(experiment=experiment,pair=pair_folder.name,case=path.parent.name,method=method,
                            same_original_cores=sum(len(y) for y in current_labels),clean_changed_f1=a['directional_changed_f1'],
                            same_positive_scan0=int(current_labels[0].sum()),same_positive_scan1=int(current_labels[1].sum()),
                            perturbed_changed_f1=b['directional_changed_f1'],delta_changed_f1=b['directional_changed_f1']-a['directional_changed_f1'],
                            clean_binary_macro_f1=a['selection_binary_macro_f1'],perturbed_binary_macro_f1=b['selection_binary_macro_f1'],
                            delta_binary_macro_f1=b['selection_binary_macro_f1']-a['selection_binary_macro_f1']))
                        audited+=1
            del reference;gc.collect()
    pd.DataFrame(deltas).to_csv(folder/'same_index_common_support_deltas.csv',index=False)
    selected=json.loads((out/'learning/primary/selection_frozen.json').read_text())['selected_variant']
    seed_rows=[]
    for experiment in ('primary','temporal'):
        selection_file=out/f'learning/{experiment}/selection_frozen.json'
        if not selection_file.exists():continue
        choice=json.loads(selection_file.read_text())
        for seed in choice['seeds']:
            method=f"{choice['selected_variant']}_seed{seed}"
            values=clean[(clean.experiment==experiment)&(clean.method==method)]
            seed_rows.append(dict(experiment=experiment,variant=choice['selected_variant'],training_seed=seed,
                binary_macro_f1=float(values.selection_binary_macro_f1.mean()),
                changed_f1=float(values.directional_changed_f1.mean()),
                pooled_three_class_macro_f1=float(values.pooled_three_class_macro_f1.mean())))
    pd.DataFrame(seed_rows).to_csv(folder/'training_seed_variation.csv',index=False)
    names=['c2c_calibrated','m3c2_calibrated','geometry_boosted',selected+'_seed42']
    colors=['#2563eb','#ea580c','#059669','#7c3aed']
    families=[('translation','Translation (m)'),('rotation','Rotation (degrees)'),('retention','Retained fraction'),
        ('noise','Gaussian sigma (m)'),('occlusion','Requested test-core deletion fraction')]
    matched_clean=stress_clean_reference(out)
    matched_clean.to_csv(folder/'stress_initial_core_clean_metrics.csv',index=False)
    clean_base=matched_clean.groupby('method').selection_binary_macro_f1.mean().to_dict()
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(family,label) in zip(axes.ravel(),families):
        rows=frame[(frame.experiment=='primary')&(frame.family==family)&(frame.support=='full_requested')&(frame.visibility!='B')]
        for name,color in zip(names,colors):
            values=rows[rows.method==name].groupby(['severity','perturbation_seed']).selection_binary_macro_f1.mean().groupby('severity').agg(['mean','std'])
            if values.empty:continue
            x=values.index.to_numpy();y=values['mean'].to_numpy();sd=values['std'].fillna(0).to_numpy()
            base_x=1. if family=='retention' else 0.
            x=np.r_[base_x,x];y=np.r_[clean_base[name],y];sd=np.r_[0.,sd]
            order=np.argsort(x);x,y,sd=x[order],y[order],sd[order]
            ax.plot(x,y,'o-',label=name,color=color);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12)
        ax.set_xlabel(label);ax.set_ylabel('Binary macro F1; abstentions count as errors');ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.set_title(family.capitalize())
    axes.ravel()[-1].axis('off');axes.ravel()[-1].text(0,1,
        'Fold 0; mean across six same-scene pairs\nClean baseline: same 512 original cores/scan\nShading: variation across perturbation seeds\n(not independent-scene confidence intervals)\n\nOcclusion uses strict A scoring here.\nDeleted target cores have no prediction;\ninitial-core coverage is saved separately.\n\nGeometry classifier includes a full-cloud\nnearest-point vector; its context is unbounded.',va='top',fontsize=10)
    handles,labels=axes.ravel()[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(folder/'measured_robustness.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    ablations=['nearest_seed42','voxel_seed42','voxel_aug_seed42']
    reference=matched_clean.groupby('method').directional_changed_f1.mean().to_dict()
    for ax,(family,label) in zip(axes.ravel(),families):
        source=frame[(frame.experiment=='primary')&(frame.family==family)&(frame.support=='full_requested')&(frame.visibility!='B')]
        for method,color in zip(ablations,['#2563eb','#059669','#7c3aed']):
            points=source[source.method==method].groupby(['severity','perturbation_seed']).directional_changed_f1.mean().groupby('severity').agg(['mean','std'])
            if points.empty:continue
            x=np.r_[1. if family=='retention' else 0.,points.index.to_numpy()]
            y=np.r_[reference[method],points['mean']];sd=np.r_[0.,points['std'].fillna(0)]
            order=np.argsort(x);x,y,sd=x[order],y[order],sd[order]
            ax.plot(x,y,'o-',color=color,label=method);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12)
        ax.set_xlabel(label);ax.set_ylabel('Mean removal/addition F1');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes.ravel()[-1].axis('off');axes.ravel()[-1].text(0,1,
        'Fold 0; all controls use training seed42.\nClean baseline: same 512 original cores/scan\nShading is perturbation-seed variation.\n\nAugmentation is local patch rotation,\njitter and dropout; it is not a\nfull-scan registration augmentation.',va='top',fontsize=10)
    handles,labels=axes.ravel()[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(folder/'learning_robustness_ablation.png',dpi=180);plt.close(fig)
    if (out/'spatial_cv').exists():
        frames=[frame[frame.condition=='clean'].assign(spatial_fold=0)]
        for root in sorted((out/'spatial_cv').glob('fold_*')):
            if (root/'evaluated_metrics.csv').exists():
                a=pd.read_csv(root/'evaluated_metrics.csv',dtype={'pair':str})
                a=enrich(a,root)
                frames.append(a.assign(spatial_fold=int(root.name.split('_')[-1])))
        cv=pd.concat(frames,ignore_index=True);cv=cv[cv.experiment=='primary'];cv.to_csv(folder/'spatial_crossvalidation_all.csv',index=False)
        controls=cv[(cv.support=='full_requested')&cv.method.isin(['c2c_calibrated','c2c_legacy_075',
            'm3c2_calibrated','m3c2_legacy','geometry_boosted','nearest_seed42','voxel_seed42','voxel_aug_seed42'])]
        controls.groupby('method').agg(binary_macro_f1=('selection_binary_macro_f1','mean'),
            changed_f1=('directional_changed_f1','mean'),unknown_rate=('detector_unknown_rate','mean'),
            pooled_three_class_macro_f1=('pooled_three_class_macro_f1','mean'),
            pooled_three_class_miou=('pooled_three_class_miou','mean'),pair_fold_count=('pair','size')).to_csv(folder/'spatial_crossvalidation_controls_summary.csv')
        # The selected neural model can differ by fold; its selection uses only
        # that fold's validation data, and the comparison uses seed42 consistently.
        selected_rows=[]
        for fold,part in cv.groupby('spatial_fold'):
            root=out if fold==0 else out/f'spatial_cv/fold_{fold}'
            variant=json.loads((root/'learning/primary/selection_frozen.json').read_text())['selected_variant']
            keep=part[part.method.isin(['c2c_calibrated','m3c2_calibrated','geometry_boosted',variant+'_seed42'])].copy()
            keep['method']=keep.method.replace({variant+'_seed42':'pointnet_validation_selected'})
            selected_rows.append(keep)
        cv=pd.concat(selected_rows,ignore_index=True);cv.to_csv(folder/'spatial_crossvalidation_selected.csv',index=False)
        cv=cv[cv.support=='full_requested']
        spatial_support_summary(cv).to_csv(folder/'spatial_class_support_and_pooled_confusions.csv',index=False)
        changed=np.column_stack([np.where(cv[f'scan{i}_positives']>0,cv[f'scan{i}_f1_changed'],np.nan) for i in (0,1)])
        present=np.isfinite(changed).sum(1)
        cv['changed_f1_positive_support_only']=np.divide(np.nansum(changed,axis=1),present,out=np.full(len(cv),np.nan),where=present>0)
        cv.groupby('method').agg(binary_macro_f1=('selection_binary_macro_f1','mean'),changed_f1=('directional_changed_f1','mean'),
            unknown_rate=('detector_unknown_rate','mean'),pair_fold_count=('pair','size'),
            pooled_three_class_macro_f1=('pooled_three_class_macro_f1','mean'),
            pooled_three_class_miou=('pooled_three_class_miou','mean'),
            changed_f1_positive_support_only=('changed_f1_positive_support_only','mean')).to_csv(folder/'spatial_crossvalidation_summary.csv')
        per_fold=cv.groupby(['spatial_fold','method']).agg(binary_macro_f1=('selection_binary_macro_f1','mean'),
            changed_f1=('directional_changed_f1','mean'),unknown_rate=('detector_unknown_rate','mean'))
        per_fold.to_csv(folder/'spatial_crossvalidation_per_fold.csv')
        per_fold.groupby('method').agg(['mean','std']).to_csv(folder/'between_spatial_fold_variation.csv')
    write_json(folder/'analysis_audit.json',dict(passed=True,paired_metric_rows_recomputed=audited,
        immutable_measurement_hashes_checked=True,scope='Uniform cores within buffered cell interiors; not full-cloud accuracy',
        figures_are_measured_not_schematic=True))
    write_json(folder/'metric_definitions.json',dict(
        selection_binary_macro_f1='Mean of changed/unchanged F1 in each direction; averaged equally across directions/pairs. Unknown is FN for its actual class.',
        directional_changed_f1='Mean of removal and addition F1. Zero convention when a class is absent; positive counts and a positive-support-only summary are also provided.',
        pooled_three_class_macro_f1='Project directional pooling: unchanged, removed, added. Unknown remains a fourth prediction column, penalizing recall of the true class.',
        full_requested='All available requested source0 cores and retained requested target1 cores. Artificially deleted target cores have no prediction; initial requested counts are recorded separately.',
        common_valid='Same M3C2-calibrated-valid available original IDs for every method. Detector abstention and evaluation exclusions are different.',
        correct_over_initial_requested_cores='Conservative bound: available correct predictions divided by the initial requested cores, counting detector abstentions and deleted target cores as errors.',
        uncertainty='Training-seed, perturbation-seed and spatial-fold variation are distinct. None is independent-scene confidence.',
        context_scope='PointNet and M3C2 have bounded neighborhoods. The boosted classifier also uses a full-cloud nearest-point vector clipped by component; its geometric receptive field is unbounded and is not covered by the buffer guarantee. No test labels enter this feature.',
        robustness_scope='Stress study uses fold 0 models only. Clean curve endpoints use the same 512 original cores per scan as the stress design.',
        supplementary_confusion_pooling='Sum all out-of-fold confusion counts across pairs before computing unchanged/removed/added F1. Different weighting from mean pair/fold scores; no model or threshold is chosen from this descriptive summary.',
        positive_support_direction_changed_f1='Mean changed F1 over scan directions containing changed GT. Always accompanied by unchanged-only false-alarm counts; not a replacement for the primary all-region metric.'))
    from multipair_diagnostics import diagnose
    diagnose(out)
    from multipair_operating_points import run as operating_points
    operating_points(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);summary(p.parse_args().out)
