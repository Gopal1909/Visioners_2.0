"""Audit and summarize all measured study scopes, preserving support distinctions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_pipeline import ROOT,DEFAULT,CONDITIONS
from m3c2_baseline import score_pair,write_json,digest

COLORS={'C2C common':'#2563eb','M3C2-style':'#ea580c','Siamese PointNet common':'#0f766e'}
NAMES={'C2C common':'C2C','M3C2-style':'M3C2-style','Siamese PointNet common':'Siamese PointNet + NN'}


def ordered_curve(x,y):
    """Connect quantitative severity values in order, including clean retention=100%."""
    order=np.argsort(np.asarray(x),kind='stable')
    return np.asarray(x)[order],np.asarray(y)[order]


def raw_preds(measurements,learned):
    return {'M3C2-style':[measurements[f'decision{i}_prediction'] for i in (0,1)],
            'C2C common':[measurements[f'c2c_prediction{i}'] for i in (0,1)],
            'Siamese PointNet common':[learned[f'prediction{i}'] for i in (0,1)]}


def analyze(out=DEFAULT):
    out=Path(out);dest=out/'analysis';dest.mkdir(exist_ok=True)
    geo=pd.read_csv(out/'geometry_metrics.csv')
    learn=pd.read_csv(out/'learned/evaluation_metrics.csv')
    frame=pd.concat([geo,learn],ignore_index=True)
    frame['evaluation_group']='buffered spatial test; same scene/epochs; historic threshold exposure'
    frame.to_csv(dest/'all_spatial_test_metrics.csv',index=False)
    common=frame[frame.method.isin(COLORS)].copy()
    common.to_csv(dest/'matched_condition_metrics.csv',index=False)
    primary=common[common.condition=='clean']
    primary.to_csv(dest/'primary_comparison.csv',index=False)
    allframes=[frame]
    supplement=ROOT/'results/occlusion_supplement_2026-10-25_v1'
    supp=pd.read_csv(supplement/'geometry_metrics.csv')
    supp['evaluation_group']='exploratory all-scene geometric occlusion supplement'
    allframes.append(supp)
    pilot=pd.read_csv(ROOT/'results/pair_01/m3c2/pilot_2026-10-25_v2/primary_comparison.csv')
    pilot['evaluation_group']='historical 2000-core/epoch all-scene pilot; exploratory'
    pilot['condition']='clean';allframes.append(pilot)
    hist=pd.read_csv(ROOT/'results/pair_01/c2c_baseline/threshold_metrics.csv')
    hist=hist[np.isclose(hist.threshold_m,.075)].copy()
    hist=hist.rename(columns={'comb_macro_f1':'combined_macro_f1','comb_macro_mIoU':'combined_macro_mIoU'})
    hist['method']='C2C full pair';hist['condition']='clean';hist['coverage']=1.;hist['unknown_rate']=0.
    hist['evaluation_group']='historical full-pair; threshold selected on same pair'
    allframes.append(hist)
    pd.concat(allframes,ignore_index=True).to_csv(dest/'all_evaluated_results.csv',index=False)
    clean=dict(np.load(out/'geometry/clean/measurements.npz'))
    clearn=dict(np.load(out/'learned/evaluation/clean/predictions.npz'))
    center=np.array(json.loads((out/'perturbation_geometry.json').read_text())['center_used_float32_m'])
    longitudinal=[];checked=0
    for name,kind,value in CONDITIONS:
        m=dict(np.load(out/f'geometry/{name}/measurements.npz'))
        l=dict(np.load(out/f'learned/evaluation/{name}/predictions.npz'))
        preds=raw_preds(m,l);base=raw_preds(clean,clearn)
        gt=[m[f'gt{i}'] for i in (0,1)]
        for i in (0,1):
            assert np.array_equal(m[f'indices{i}'],l[f'indices{i}'])
            assert np.array_equal(m[f'decision{i}_valid'],m[f'decision{i}_reason']==0)
            mask=m[f'decision{i}_valid']
            assert np.all(m[f'decision{i}_prediction'][~mask]==-1)
            assert np.all(m[f'raw{i}_count_source'][mask,0]>=10)
            assert np.all(m[f'raw{i}_count_target'][mask,0]>=10)
        for protocol in ('A','B') if kind=='occlusion' else ('standard',):
            eligible=[np.ones(len(g),bool) for g in gt]
            if protocol=='B':eligible[0]=np.linalg.norm(m['xyz0'][:,:2]-center[:2],axis=1)>value
            for method,pp in preds.items():
                mm=[m[f'decision{i}_valid'][eligible[i]] for i in (0,1)]
                result=score_pair([gt[i][eligible[i]] for i in (0,1)],
                                 [pp[i][eligible[i]] for i in (0,1)],mm)
                row=common[(common.condition==name)&(common.protocol==protocol)&(common.method==method)].iloc[0]
                for key in ('combined_macro_f1','combined_macro_mIoU','coverage','directional_macro_f1'):
                    assert abs(result[key]-row[key])<1e-10,(name,method,key)
                checked+=1
            # Longitudinal comparison intersects clean AND current validity and
            # retained original indices. Delta is not confounded by changing masks.
            positions=[np.searchsorted(clean[f'indices{i}'],m[f'indices{i}']) for i in (0,1)]
            for i in (0,1):
                np.testing.assert_array_equal(clean[f'indices{i}'][positions[i]],m[f'indices{i}'])
                np.testing.assert_array_equal(clean[f'gt{i}'][positions[i]],gt[i])
            masks=[m[f'decision{i}_valid']&clean[f'decision{i}_valid'][positions[i]]&eligible[i] for i in (0,1)]
            for method in preds:
                current=score_pair(gt,preds[method],masks)
                baseline=score_pair(gt,[base[method][i][positions[i]] for i in (0,1)],masks)
                longitudinal.append(dict(condition=name,family=kind,severity=value,protocol=protocol,method=method,
                    common_core_count=sum(int(x.sum()) for x in masks),
                    baseline_macro_f1=baseline['combined_macro_f1'],perturbed_macro_f1=current['combined_macro_f1'],
                    delta_macro_f1=current['combined_macro_f1']-baseline['combined_macro_f1'],
                    baseline_macro_mIoU=baseline['combined_macro_mIoU'],perturbed_macro_mIoU=current['combined_macro_mIoU'],
                    delta_macro_mIoU=current['combined_macro_mIoU']-baseline['combined_macro_mIoU']))
    delta=pd.DataFrame(longitudinal);delta.to_csv(dest/'matched_clean_deltas.csv',index=False)
    # Independent full-pair C2C cache comparison at new clean original cores.
    cachecheck={}
    with np.load(ROOT/'results/pair_01/c2c_baseline/c2c_predictions.npz') as z:
        for i,key in enumerate(('distances_01','distances_10')):
            ids=clean[f'indices{i}']
            discrepancy=float(np.max(np.abs(z[key][ids]-clean[f'c2c_distance{i}_m'])))
            assert discrepancy<2e-6
            np.testing.assert_array_equal(z[f'gt{i}'][ids],clean[f'gt{i}'])
            cachecheck[f'scan{i}_max_distance_difference_m']=discrepancy
    write_json(dest/'metric_audit.json',dict(passed=True,matched_rows_recomputed=checked,
        all_conditions=len(CONDITIONS),predictions_at_identical_indices=True,
        unknowns_explicit=True,complete_cylinder_minimum_counts_verified=True,c2c_cache=cachecheck))
    # Compact transferable bundle of real measurements and predictions for demo.
    arrays={}
    for i in (0,1):
        for key in ('xyz','indices','gt','c2c_prediction'):
            arrays[f'{key}{i}']=clean[f'{key}{i}']
        arrays[f'm3c2_prediction{i}']=clean[f'decision{i}_prediction']
        arrays[f'learned_prediction{i}']=clearn[f'prediction{i}']
        arrays[f'signed_distance{i}_m']=clean[f'raw{i}_signed_distance_m'][:,0]
        arrays[f'lod95_{i}_m']=clean[f'decision{i}_lod95_m']
    np.savez_compressed(dest/'demo_measurements.npz',**arrays)
    plots(dest,common,supp,delta,clean,clearn,out)
    split=json.loads((out/'split_config.json').read_text())
    frozen=json.loads((out/'learned/model_frozen.json').read_text())
    train=json.loads((out/'learned/training_runtime.json').read_text())
    summary=dict(primary=primary[['method','combined_macro_f1','combined_macro_mIoU','directional_macro_f1','coverage','unknown_rate','conservative_accuracy_lower_bound','geometry_seconds','sampled_process_peak_rss_mib']].to_dict('records'),
        learned_spatial_test_all_macro_f1=float(frame[(frame.condition=='clean')&(frame.method=='Siamese PointNet all')].iloc[0].combined_macro_f1),
        split=split,learned=frozen,training=train,
        supplementary_occlusion=supp[['condition','protocol','method','combined_macro_f1','combined_macro_mIoU','coverage','unknown_rate']].to_dict('records'),
        main_occlusion_note='Fixed spatial test stripe is outside both deletion cylinders; identical scores are not occlusion robustness evidence.',
        numerical_center_note='Stage 3 float32 mean center preserved exactly. Its X is 38.895813 m versus float64 mean 39.165547 m, about 0.27 m apart. This affects the defined stress location, not the equality of method inputs.',
        protocol='Same-scene spatial holdout for learning; full pair threshold selection history and prior pilot exposure prevent independent benchmark claims.')
    write_json(dest/'submission_data.json',summary)
    print(primary[['method','combined_macro_f1','combined_macro_mIoU','coverage','unknown_rate']].to_string(index=False))
    print('Audited',checked,'matched rows')


def plots(dest,common,supp,delta,clean,learn,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(family,label,scale) in zip(axes.flat,[('translation','Translation magnitude (cm)',100),('rotation','Rotation (degrees)',1),('sparsity','Scan 1 retained (%)',100),('noise','Per-axis noise SD (cm)',100)]):
        for method,color in COLORS.items():
            base=common[(common.condition=='clean')&(common.method==method)].iloc[0]
            group=common[(common.family==family)&(common.method==method)].sort_values('severity')
            xx=[100 if family=='sparsity' else 0]+(group.severity*scale).tolist()
            yy=[base.combined_macro_f1]+group.combined_macro_f1.tolist()
            xx,yy=ordered_curve(xx,yy)
            ax.plot(xx,yy,'o-',label=NAMES[method],color=color,lw=2)
        ax.set(xlabel=label,ylabel='Pooled 3-class macro F1',ylim=(0,1));ax.grid(alpha=.2)
        if family=='sparsity':ax.invert_xaxis()
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Buffered spatial-test cores: fixed parameters, condition-specific common valid support')
    fig.tight_layout();fig.savefig(dest/'robustness_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for protocol,ax in zip(('A','B'),axes):
        for method,color in list(COLORS.items())[:2]:
            name='C2C common' if method=='C2C common' else method
            base=supp[(supp.condition=='clean')&(supp.method==name)].iloc[0]
            group=supp[(supp.protocol==protocol)&(supp.method==name)].sort_values('severity')
            ax.plot([0]+group.severity.tolist(),[base.combined_macro_f1]+group.combined_macro_f1.tolist(),
                     'o-',color=color,label=NAMES[method])
        ax.set(title=f'Occlusion {protocol}: '+('all requested Scan 0 cores' if protocol=='A' else 'exclude known blind cylinder'),
               xlabel='Deleted cylinder radius (m)',ylabel='Pooled 3-class macro F1',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Separate exploratory all-scene geometric supplement; 2,000 requested cores/epoch')
    fig.tight_layout();fig.savefig(dest/'occlusion_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.5))
    group=common[common.method=='M3C2-style'].drop_duplicates(['condition','protocol'])
    ax.bar(np.arange(len(group)),group.coverage,color='#2563eb',label='valid')
    ax.bar(np.arange(len(group)),group.unknown_rate,bottom=group.coverage,color='#9ca3af',label='unknown')
    ax.set_xticks(np.arange(len(group)),group.condition+' '+group.protocol,rotation=45,ha='right')
    ax.set(ylabel='Fraction of retained requested cores',ylim=(0,1),title='M3C2 coverage by simulated condition');ax.legend()
    fig.tight_layout();fig.savefig(dest/'coverage.png',dpi=180);plt.close(fig)
    s=dict(np.load(out/'splits.npz'))
    fig,ax=plt.subplots(figsize=(10,5))
    for split,color in [('train','#2563eb'),('calibration','#eab308'),('test','#0f766e')]:
        p=np.concatenate([s[f'{split}_xyz{i}'] for i in (0,1)])
        ax.scatter(p[:,0],p[:,1],s=1,alpha=.4,color=color,label=split)
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Geometry-defined split with 2.22 m empty gaps between core regions');ax.legend(markerscale=5)
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(dest/'spatial_split.png',dpi=180);plt.close(fig)
    history=pd.read_csv(out/'learned/training_history.csv')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(history.epoch,history.train_loss,label='training')
    ax.plot(history.epoch,history.calibration_loss,label='calibration')
    frozen=json.loads((out/'learned/model_frozen.json').read_text())
    ax.axvline(frozen['best_epoch'],ls='--',color='gray',label=f'selected epoch {frozen["best_epoch"]}')
    ax.set(xlabel='Epoch',ylabel='Weighted BCE',title='Fixed 30-epoch run; calibration loss selects the checkpoint');ax.legend()
    fig.tight_layout();fig.savefig(dest/'training_curve.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,4.5),sharex=True,sharey=True)
    p=clean['xyz1'];valid=clean['decision1_valid']
    fields=[clean['gt1'],clean['c2c_prediction1'],clean['decision1_prediction'],learn['prediction1']]
    for ax,title,values in zip(axes,['Ground truth additions','C2C common support','M3C2-style','Siamese PointNet + NN'],fields):
        vals=values if title=='Ground truth additions' else np.where(valid,values,-1)
        c=np.where(vals<0,'#9ca3af',np.where(vals==1,'#dc2626','#2563eb'))
        ax.scatter(p[:,0],p[:,1],c=c,s=2)
        ax.set(title=title,xlabel='X (m)');ax.set_aspect('equal')
    axes[0].set_ylabel('Y (m)');fig.suptitle('Scan 1 spatial-test cores: red added candidate, blue unchanged, gray unknown/excluded')
    fig.tight_layout();fig.savefig(dest/'change_maps.png',dpi=190);plt.close(fig)
    # Convenient native-chart values for the editable presentation.
    compact=common[['condition','family','severity','protocol','method','combined_macro_f1','combined_macro_mIoU','coverage']]
    compact.to_json(dest/'presentation_metrics.json',orient='records',indent=2)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(DEFAULT))
    analyze(parser.parse_args().out)
