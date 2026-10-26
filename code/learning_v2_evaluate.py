"""Identical-core evaluation and independent audits for the learning upgrade."""
import sys
# Keep the compatible CLI while routing CUDA work before loading CPU geometry
# and plotting libraries; these imports exhaust this Windows host's commit limit.
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='inference':
    from learning_v2_infer import main
    main()
    raise SystemExit(0)
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from learning_v2_data import (ROOT,BASE,DEFAULT,PAIRS,VARIANTS,SUPPORT_BOUND,base_fold,fold_dir,
    load_pair,features,contexts_for)
from multipair_evaluate import case_cloud,metric_rows
from multipair_metrics import pair_scores,directional_three_class_scores
from m3c2_baseline import digest,write_json,MemoryTimer


def cache_evaluation(out=DEFAULT,fold=0,pair='01',stress=False):
    out=Path(out)
    assert all((fold_dir(out,i)/'learning/selection.json').exists() for i in range(5)), 'Freeze all fold choices before testing'
    root=fold_dir(out,fold);src=base_fold(fold)
    folders=[p for p in sorted((src/f'evaluation/primary/pair_{pair}').iterdir()) if p.is_dir() and
        ((p.name!='clean_seed42') if stress else p.name=='clean_seed42')]
    pending=[p for p in folders if not (root/f'evaluation/pair_{pair}'/p.name/'feature_complete.json').exists()]
    if not pending:return
    data=load_pair(pair);tree0=cKDTree(data[0]['xyz']);context0=None
    protocol=json.loads((root/'protocol.json').read_text())
    train_displacement=max(json.loads(p.read_text())['maximum_observed_displacement_m']
        for p in (root/'cache').glob('pair_*_train_*.json'))
    with np.load(src/f'splits/pair_{pair}.npz') as z:wanted=z['test_indices1']
    if stress:wanted=np.sort(wanted[np.random.default_rng(7025+2*int(pair)+1).choice(len(wanted),512,replace=False)])
    for old in pending:
        dst=root/f'evaluation/pair_{pair}'/old.name;dst.mkdir(parents=True,exist_ok=True)
        cond=json.loads((old/'condition.json').read_text())
        receipt=json.loads((old/'geometry_complete.json').read_text());raw_path=old/'geometry_measurements.npz'
        assert digest(raw_path)==receipt['measurement_sha256']
        with MemoryTimer() as timer:
            xyz1,retained,_=case_cloud(data,cond['kind'],cond['severity'],cond['seed'],
                np.array(protocol['stress_rotation_center_m']),data[1]['xyz'][wanted])
            # Independent noise can move both a target core and its target
            # support point; rigid transforms preserve within-scan distances.
            displacement=cond.get('maximum_observed_perturbation_m',0.)
            test_support_displacement=displacement*(2 if cond['kind']=='noise' else 1)
            bound=2*SUPPORT_BOUND+train_displacement+test_support_displacement
            assert bound<protocol['minimum_cross_split_core_distance_m']
            tree1=cKDTree(xyz1);trees=[tree0,tree1]
            if context0 is None:context0=[c[0] for c in contexts_for([tree0,cKDTree(np.empty((0,3)))])]
            context1=[c[0] for c in contexts_for([tree1,cKDTree(np.empty((0,3)))])]
            context=[[a,b] for a,b in zip(context0,context1)];saved={}
            with np.load(raw_path) as z:
                for scan in (0,1):
                    ids=z[f'indices{scan}'];centers=z[f'xyz{scan}']
                    np.testing.assert_array_equal(z[f'labels{scan}'],data[scan]['labels'][ids])
                    if scan==0:np.testing.assert_array_equal(centers,data[0]['xyz'][ids])
                    else:
                        pos=np.searchsorted(retained,ids);np.testing.assert_array_equal(retained[pos],ids)
                        np.testing.assert_array_equal(centers,xyz1[pos])
                    f=features([trees[scan],trees[1-scan]],[[a[scan],a[1-scan]] for a in context],centers)
                    for key,value in f.items():saved[f'{key}{scan}']=value
                    for key in ('labels','indices','xyz'):saved[f'{key}{scan}']=z[f'{key}{scan}']
            np.savez_compressed(dst/'features.npz',**saved)
            write_json(dst/'condition.json',dict(cond,v2_total_support_guard_m=float(bound),v2_guard_passed=True,
                v2_test_support_displacement_bound_m=test_support_displacement))
        write_json(dst/'feature_complete.json',dict(sha256=digest(dst/'features.npz'),
            baseline_geometry_sha256=receipt['measurement_sha256'],source_sha256=digest(ROOT/'code/learning_v2_data.py'),
            design_sha256=digest(out/'design.json'),identical_original_cores_and_coordinates_verified=True,timing=timer.stats))
        print('V2 test features',fold,pair,old.name,round(timer.stats['seconds'],1),'seconds',flush=True)
        del tree1,trees,context1,context,xyz1,retained,saved,f;gc.collect()


def load_models(out,fold):
    import torch,joblib
    from learning_v2_model import MetricFusion
    folder=fold_dir(out,fold)/'learning';selection=json.loads((folder/'selection.json').read_text())
    names=[v+'_seed42' for v in VARIANTS]
    if fold==0:names.extend(selection['primary_variant']+f'_seed{s}' for s in (43,44))
    models={}
    for name in names:
        frozen=json.loads((folder/name/'frozen.json').read_text());path=folder/name/'best.pt'
        assert digest(path)==frozen['checkpoint_sha256']
        model=MetricFusion().cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);model.eval();del state
        models[name]=(model,frozen['threshold'],frozen['checkpoint_sha256'])
    folder=folder/'bounded_geometry';meta=json.loads((folder/'frozen.json').read_text())
    assert digest(folder/'model.joblib')==meta['checkpoint_sha256']
    models['bounded_geometry']=(joblib.load(folder/'model.joblib'),meta['threshold'],meta['checkpoint_sha256'])
    return models


def inference(out=DEFAULT,fold=0,pair='01',stress=False):
    out=Path(out);root=fold_dir(out,fold);folders=[p for p in sorted((root/f'evaluation/pair_{pair}').iterdir())
        if p.is_dir() and ((p.name!='clean_seed42') if stress else p.name=='clean_seed42') and not (p/'complete.json').exists()]
    if not folders:return
    from learning_v2_model import probabilities
    from siamese_pointnet import setup
    setup(42);models=load_models(out,fold)
    for dst in folders:
        receipt=json.loads((dst/'feature_complete.json').read_text());assert digest(dst/'features.npz')==receipt['sha256']
        with np.load(dst/'features.npz') as z:saved={k:z[k] for k in z.files}
        labels=[saved[f'labels{i}'] for i in (0,1)];prediction={};probability={}
        timing={};condition=json.loads((dst/'condition.json').read_text())
        for name,(model,threshold,_) in models.items():
            started=time.perf_counter();p=[]
            for scan in (0,1):
                f={k:saved[f'{k}{scan}'] for k in ('x','mask','geo','labels')}
                p.append(model.predict_proba(f['geo'])[:,1] if name=='bounded_geometry' else probabilities(model,f))
            probability[name]=p;prediction[name]=[(a>=threshold).astype(np.int8) for a in p]
            timing[name]=time.perf_counter()-started
        old=base_fold(fold)/f'evaluation/primary/pair_{pair}'/dst.name
        with np.load(old/'predictions.npz') as z:common=[z[f'm3c2_calibrated_prediction{i}']>=0 for i in (0,1)]
        rows=[];name,seed=dst.name.rsplit('_seed',1)
        for visibility in ('A','B') if condition['kind']=='occlusion' else ('standard',):
            masks=[np.ones(len(y),bool) for y in labels]
            if visibility=='B':masks[0]=np.linalg.norm(saved['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
            metadata=dict(experiment='learning_v2',spatial_fold=fold,pair=pair,condition=name,family=condition['kind'],
                severity=condition['severity'],perturbation_seed=int(seed),visibility=visibility)
            rows.extend(metric_rows(labels,prediction,common,metadata,masks,condition['target_requested_cores'],condition.get('requested_label_counts')))
        result={k:v for k,v in saved.items() if not k.startswith(('x0','x1','mask','geo'))}
        for i in (0,1):
            result[f'common_valid{i}']=common[i]
            for method in prediction:
                result[f'{method}_prediction{i}']=prediction[method][i];result[f'{method}_probability{i}']=probability[method][i]
        np.savez_compressed(dst/'predictions.npz',**result);pd.DataFrame(rows).to_csv(dst/'metrics.csv',index=False)
        write_json(dst/'complete.json',dict(passed=True,predictions_sha256=digest(dst/'predictions.npz'),
            model_sha256={k:m[2] for k,m in models.items()},feature_sha256=receipt['sha256'],
            selection_sha256=digest(root/'learning/selection.json'),no_test_retuning=True,inference_seconds=timing))
        print('V2 inference',fold,pair,dst.name,flush=True)
        del saved,result,prediction,probability;gc.collect()


def analyse(out=DEFAULT):
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True)
    new=[];old=[]
    for fold in range(5):
        selection=json.loads((fold_dir(out,fold)/'learning/selection.json').read_text())
        old_selection=json.loads((base_fold(fold)/'learning/primary/selection_frozen.json').read_text())['selected_variant']+'_seed42'
        for path in (fold_dir(out,fold)/'evaluation').glob('*/*/metrics.csv'):
            if not (path.parent/'complete.json').exists():continue
            frame=pd.read_csv(path,dtype={'pair':str});new.append(frame)
            if path.parent.name=='clean_seed42':
                primary=frame[frame.method==selection['primary_variant']+'_seed42'].copy();primary['method']='v2_validation_selected';new.append(primary)
                secondary=frame[frame.method==selection['secondary_robust_variant']+'_seed42'].copy();secondary['method']='v2_robust_validation_selected';new.append(secondary)
            baseline=base_fold(fold)/'evaluation/primary'/path.parent.parent.name/path.parent.name/'metrics.csv'
            b=pd.read_csv(baseline,dtype={'pair':str});b['spatial_fold']=fold;b['method']='previous_'+b.method
            old.append(b)
            if path.parent.name=='clean_seed42':
                selected=b[b.method=='previous_'+old_selection].copy();selected['method']='previous_validation_selected';old.append(selected)
    frame=pd.concat([*new,*old],ignore_index=True);frame.to_csv(folder/'comparison_all_metrics.csv',index=False)
    clean=frame[(frame.condition=='clean')&(frame.support=='full_requested')]
    clean.to_csv(folder/'clean_per_pair_fold.csv',index=False)
    # An absent class receives F1=0 under the predeclared convention, even for
    # perfect predictions. Keep this diagnostic distinct from a model score.
    reference=clean[clean.method=='v2_validation_selected'];binary_caps=[];changed_caps=[];pooled_caps=[]
    for row in reference.to_dict('records'):
        positives=[int(row[f'scan{i}_positives']) for i in (0,1)];cores=[int(row[f'scan{i}_cores']) for i in (0,1)]
        binary_caps.extend(.5*(int(p>0)+int(n>p)) for p,n in zip(positives,cores))
        changed_caps.extend(int(p>0) for p in positives)
        pooled_caps.append((int(sum(cores)>sum(positives))+sum(int(p>0) for p in positives))/3)
    write_json(folder/'metric_ceiling_diagnostic.json',dict(diagnostic_only_not_a_predictive_method=True,
        scan_directions=len(changed_caps),directions_with_changed_gt=sum(changed_caps),
        perfect_prediction_mean_binary_macro_f1=float(np.mean(binary_caps)),
        perfect_prediction_mean_directional_changed_f1=float(np.mean(changed_caps)),
        perfect_prediction_mean_pair_three_class_macro_f1=float(np.mean(pooled_caps)),
        explanation='F1=0 when a class is absent. All-region false alarms remain counted; supplementary global confusion pooling has different weights and is saved separately.'))
    summary=clean.groupby('method').agg(binary_macro_f1=('selection_binary_macro_f1','mean'),changed_f1=('directional_changed_f1','mean'),
        detector_unknown_rate=('detector_unknown_rate','mean'),pair_fold_count=('pair','size'))
    summary.to_csv(folder/'clean_comparison.csv');print(summary.to_string(),flush=True)
    per_fold=clean.groupby(['spatial_fold','method']).agg(binary_macro_f1=('selection_binary_macro_f1','mean'),
        changed_f1=('directional_changed_f1','mean'))
    per_fold.to_csv(folder/'clean_per_fold.csv');per_fold.groupby('method').agg(['mean','std']).to_csv(folder/'between_fold_variation.csv')
    selected_clean=clean[clean.method=='v2_validation_selected'].set_index(['spatial_fold','pair'])
    previous_clean=clean[clean.method=='previous_validation_selected'].set_index(['spatial_fold','pair'])
    paired=selected_clean[['selection_binary_macro_f1','directional_changed_f1']].join(
        previous_clean[['selection_binary_macro_f1','directional_changed_f1']],lsuffix='_new',rsuffix='_previous',validate='one_to_one')
    for metric in ('selection_binary_macro_f1','directional_changed_f1'):paired[metric+'_delta']=paired[metric+'_new']-paired[metric+'_previous']
    paired.to_csv(folder/'paired_clean_deltas.csv')
    seed_part=clean[(clean.spatial_fold==0)&clean.method.isin([
        json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['primary_variant']+f'_seed{s}' for s in (42,43,44)])]
    seed_part.groupby('method').agg(binary_macro_f1=('selection_binary_macro_f1','mean'),
        changed_f1=('directional_changed_f1','mean'),pair_count=('pair','size')).to_csv(folder/'training_seed_variation.csv')
    from multipair_analysis import spatial_support_summary,stress_condition_summary
    spatial_support_summary(clean).to_csv(folder/'class_support_and_pooled_confusions.csv',index=False)
    robust=frame[frame.spatial_fold==0].copy();robust['experiment']='primary'
    stress_condition_summary(robust).to_csv(folder/'stress_condition_summary.csv',index=False)
    # Plot baseline and all predeclared neural ablations. Clean endpoints use
    # the exact original 512 stress-core IDs, not the larger 1500-core sample.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['primary_variant']+'_seed42'
    old_selected=json.loads((BASE/'learning/primary/selection_frozen.json').read_text())['selected_variant']+'_seed42'
    names=['previous_'+old_selected,*[v+'_seed42' for v in VARIANTS]];base_rows=[]
    for pair in PAIRS:
        target=fold_dir(out,0)/f'evaluation/pair_{pair}'
        with np.load(target/'translation_0.05_seed42/predictions.npz') as z:ids=[z[f'indices{i}'] for i in (0,1)]
        for location,prefix in [(target,''),(BASE/f'evaluation/primary/pair_{pair}','previous_')]:
            with np.load(location/'clean_seed42/predictions.npz') as z:
                positions=[np.searchsorted(z[f'indices{i}'],ids[i]) for i in (0,1)]
                for i in (0,1):np.testing.assert_array_equal(z[f'indices{i}'][positions[i]],ids[i])
                labels=[z[f'labels{i}'][positions[i]] for i in (0,1)]
                for key in z.files:
                    if key.endswith('_prediction0'):
                        method=key[:-len('_prediction0')]
                        base_rows.append(dict(pair=pair,method=prefix+method,**pair_scores(labels,[z[f'{method}_prediction{i}'][positions[i]] for i in (0,1)])))
    base_frame=pd.DataFrame(base_rows);base_frame.to_csv(folder/'matched_clean_stress_cores.csv',index=False)
    families=[('translation','Translation (m)'),('rotation','Rotation (degrees)'),('retention','Retained fraction'),
        ('noise','Noise sigma (m)'),('occlusion','Requested test-core deletion fraction')]
    for metric,label,filename in [('selection_binary_macro_f1','Binary macro F1','robustness_binary_macro.png'),
        ('directional_changed_f1','Mean removal/addition F1','robustness_changed_f1.png')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8));baseline=base_frame.groupby('method')[metric].mean().to_dict()
        for ax,(family,xlabel) in zip(axes.ravel(),families):
            part=robust[(robust.family==family)&(robust.support=='full_requested')&(robust.visibility!='B')]
            for method in names:
                values=part[part.method==method].groupby(['severity','perturbation_seed'])[metric].mean().groupby('severity').agg(['mean','std'])
                x=np.r_[1. if family=='retention' else 0.,values.index];y=np.r_[baseline[method],values['mean']];sd=np.r_[0.,values['std'].fillna(0)]
                order=np.argsort(x);x,y,sd=x[order],y[order],sd[order]
                line=ax.plot(x,y,'o-',label=method)[0];ax.fill_between(x,y-sd,y+sd,color=line.get_color(),alpha=.12)
            ax.set_xlabel(xlabel);ax.set_ylabel(label);ax.set_ylim(0,1);ax.grid(alpha=.2)
        axes.ravel()[-1].axis('off');axes.ravel()[-1].text(0,1,
            'Fold 0; mean across six same-site pairs\nAll displayed models use training seed42.\nShading: perturbation-seed SD, not a CI.\nClean endpoints: identical 512 cores/scan.\n\nOcclusion uses strict A scoring;\ndeleted target cores have no prediction.\nInitial-core coverage is saved separately.\n\nPrimary validation-selected model:\n'+selected,va='top',fontsize=10)
        handles,labels=axes.ravel()[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
        fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(folder/filename,dpi=180);plt.close(fig)
    write_json(folder/'interpretation_scope.json',dict(exposure=json.loads((out/'design.json').read_text())['exposure'],
        selected_variants={str(f):json.loads((fold_dir(out,f)/'learning/selection.json').read_text())['primary_variant'] for f in range(5)},
        primary_metric='Same pair/direction binary macro F1 and changed F1 as previous study; zero convention retained for absent classes',
        source_of_selection='Validation only; comparison_all_metrics contains tests for descriptive assessment',
        stress='Fold0 models only; 16 conditions x 3 seeds x 6 pairs',
        independent_site_available=(out/'external_open_space/verification.json').exists(),
        independent_site_results='Separate untuned evaluation in external_open_space; not included in the same-site means',
        full_scan_augmentation='Two fixed full-scan perturbed training views, distinct validation seeds; finite-bank augmentation, not freshly generated each epoch'))
    plot_learning_and_errors(out)
    plot_crossvalidated_overviews(out)


def plot_crossvalidated_overviews(out):
    """Show all five held-out regions; retain the separate fold0 3D maps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=Path(out)/'analysis/crossvalidated_maps';folder.mkdir(exist_ok=True)
    palette={'TN':'#cbd5e1','TP':'#16a34a','FP':'#dc2626','FN':'#f59e0b'}
    for pair in PAIRS:
        packed={k:[[],[]] for k in ('xyz','labels','indices','previous','selected','geometry')}
        for fold in range(5):
            selected=json.loads((fold_dir(out,fold)/'learning/selection.json').read_text())['primary_variant']+'_seed42'
            previous=json.loads((base_fold(fold)/'learning/primary/selection_frozen.json').read_text())['selected_variant']+'_seed42'
            with (np.load(fold_dir(out,fold)/f'evaluation/pair_{pair}/clean_seed42/predictions.npz') as z,
                np.load(base_fold(fold)/f'evaluation/primary/pair_{pair}/clean_seed42/predictions.npz') as old):
                for scan in (0,1):
                    np.testing.assert_array_equal(z[f'indices{scan}'],old[f'indices{scan}'])
                    for key in ('xyz','labels','indices'):packed[key][scan].append(z[f'{key}{scan}'])
                    packed['previous'][scan].append(old[f'{previous}_prediction{scan}'])
                    packed['selected'][scan].append(z[f'{selected}_prediction{scan}'])
                    packed['geometry'][scan].append(z[f'bounded_geometry_prediction{scan}'])
        packed={k:[np.concatenate(v[i]) for i in (0,1)] for k,v in packed.items()}
        fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
        for scan in (0,1):
            assert len(np.unique(packed['indices'][scan]))==7500
            xyz=packed['xyz'][scan];y=packed['labels'][scan]
            for col,(key,title) in enumerate([('GT','Ground truth'),('previous','Previous PointNet workflow'),
                ('selected','New PointNet workflow'),('geometry','Bounded geometry classifier')]):
                ax=axes[scan,col]
                if key=='GT':ax.scatter(xyz[:,0],xyz[:,1],c=np.where(y==1,'#dc2626','#94a3b8'),s=2)
                else:
                    p=packed[key][scan]
                    for name,mask in [('TN',(y==0)&(p==0)),('TP',(y==1)&(p==1)),('FP',(y==0)&(p==1)),('FN',(y==1)&(p==0))]:
                        ax.scatter(xyz[mask,0],xyz[mask,1],s=2,c=palette[name])
                ax.set_title(title+f' | scan {scan}',fontsize=10);ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
        fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=k) for k,c in palette.items()],loc='lower center',ncol=4)
        fig.suptitle(f'pair_{pair}: XY projection of 7500 held-out original cores per scan across five folds\nModels selected separately in each fold. Buffered interiors only; not a full-cloud map. GT red = changed.',fontsize=11)
        # Explicit row spacing keeps X labels clear of the next row's titles
        # after equal-aspect axes shrink to match the scene footprint.
        fig.subplots_adjust(left=.045,right=.99,bottom=.12,top=.84,wspace=.25,hspace=.48)
        fig.savefig(folder/f'pair_{pair}.png',dpi=170);plt.close(fig)


def plot_learning_and_errors(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(out);folder=out/'analysis';fig,axes=plt.subplots(2,3,figsize=(14,8))
    for fold,ax in enumerate(axes.ravel()[:5]):
        for variant in VARIANTS:
            model=fold_dir(out,fold)/f'learning/{variant}_seed42'
            history=pd.read_csv(model/'history.csv');frozen=json.loads((model/'frozen.json').read_text())
            line=ax.plot(history.epoch,history.clean_validation_bce,label=variant)[0]
            ax.plot(frozen['best_epoch'],frozen['best_clean_validation_bce'],'o',color=line.get_color())
        ax.set_title(f'Fold {fold}');ax.set_xlabel('Epoch');ax.set_ylabel('Clean validation BCE');ax.grid(alpha=.2)
    axes.ravel()[-1].axis('off');axes.ravel()[-1].text(0,1,
        'All candidates: 30 fixed training epochs.\nDots: selected minimum-validation-loss\ncheckpoints. Thresholds are calibrated\nseparately on clean validation data.\n\nThese are validation curves,\nnot held-out test curves.',va='top',fontsize=11)
    handles,labels=axes.ravel()[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(folder/'validation_learning_curves.png',dpi=180);plt.close(fig)
    maps=folder/'error_maps';maps.mkdir(exist_ok=True)
    selected=json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['primary_variant']+'_seed42'
    previous=json.loads((BASE/'learning/primary/selection_frozen.json').read_text())['selected_variant']+'_seed42'
    palette={'TP':'#16a34a','FP':'#dc2626','FN':'#f59e0b','TN':'#cbd5e1'}
    for pair in PAIRS:
        with (np.load(fold_dir(out,0)/f'evaluation/pair_{pair}/clean_seed42/predictions.npz') as z,
            np.load(BASE/f'evaluation/primary/pair_{pair}/clean_seed42/predictions.npz') as old):
            xyz=[z[f'xyz{i}'] for i in (0,1)];gt=[z[f'labels{i}'] for i in (0,1)]
            both=np.concatenate(xyz);lo=both.min(0);hi=both.max(0)
            fig=plt.figure(figsize=(16,8))
            for scan in (0,1):
                np.testing.assert_array_equal(z[f'indices{scan}'],old[f'indices{scan}'])
                panels=[('Ground truth',None),('Previous PointNet',old[f'{previous}_prediction{scan}']),
                    ('New validation-selected model',z[f'{selected}_prediction{scan}']),('Bounded geometry classifier',z[f'bounded_geometry_prediction{scan}'])]
                for col,(title,pred) in enumerate(panels):
                    ax=fig.add_subplot(2,4,scan*4+col+1,projection='3d');y=gt[scan]
                    if pred is None:colors=np.where(y==1,'#dc2626','#94a3b8')
                    else:
                        colors=np.full(len(y),palette['TN']);colors[(y==1)&(pred==1)]=palette['TP']
                        colors[(y==0)&(pred==1)]=palette['FP'];colors[(y==1)&(pred==0)]=palette['FN']
                    ax.scatter(*xyz[scan].T,c=colors,s=2,alpha=.8,depthshade=False)
                    ax.set_xlim(lo[0]-.1,hi[0]+.1);ax.set_ylim(lo[1]-.1,hi[1]+.1);ax.set_zlim(lo[2]-.1,hi[2]+.1)
                    ax.set_box_aspect(np.maximum(hi-lo,.1));ax.view_init(27,-57)
                    for dim,setter in enumerate((ax.set_xticks,ax.set_yticks,ax.set_zticks)):setter(np.round([lo[dim],hi[dim]],1))
                    ax.set_xlabel('X (m)',fontsize=7);ax.set_ylabel('Y (m)',fontsize=7);ax.set_zlabel('Z (m)',fontsize=7)
                    ax.tick_params(labelsize=6);ax.set_title(title+f' | scan {scan}',fontsize=9)
            fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=k) for k,c in palette.items()],loc='lower center',ncol=4)
            fig.suptitle(f'pair_{pair}: identical fold0 held-out original cores; not a full-cloud map\nGround truth red = changed; other panels color prediction errors.',fontsize=12)
            fig.subplots_adjust(left=.03,right=.98,top=.88,bottom=.11,wspace=.12,hspace=.08)
            fig.savefig(maps/f'pair_{pair}.png',dpi=160);plt.close(fig)


def verify(out=DEFAULT):
    out=Path(out);preserved=json.loads((out/'baseline_preservation.json').read_text())
    coverage=json.loads((out/'measured_patch_coverage.json').read_text());assert coverage['passed']
    assert coverage['scales'][-1]['median_selected_outer_radius_m']>.6
    for name,sha in preserved['files'].items():assert digest(BASE/name)==sha,('Previous results changed',name)
    audit=json.loads((out/'data_audit.json').read_text())
    from plyfile import PlyData
    raw={}
    for row in audit['clouds']:
        assert digest(row['path'])==row['sha256'];raw[(row['pair'],row['scan'])]=PlyData.read(row['path'],mmap='r')['params'].data
    from learning_v2_data import arrays
    from multipair_metrics import choose_threshold
    from learning_v2_runtime import THRESHOLDS
    cases=0;rows_checked=0;models_checked=0;feature_views=0;assembled_probes=0
    for fold in range(5):
        root=fold_dir(out,fold);selection=json.loads((root/'learning/selection.json').read_text())
        frozen_models={}
        candidates=[json.loads((root/f'learning/{v}_seed42/frozen.json').read_text()) for v in VARIANTS]
        chosen=max(candidates,key=lambda c:(c['validation_objective'],-VARIANTS.index(c['variant'])))['variant']
        assert chosen==selection['primary_variant'] and not selection['test_accessed']
        for receipt in (root/'cache').glob('*.json'):
            metadata=json.loads(receipt.read_text());assert digest(receipt.with_suffix('.npz'))==metadata['sha256']
            assert metadata['bounded_guard_passed'] and metadata['train_test_total_support_bound_m']<5.
            with np.load(receipt.with_suffix('.npz')) as cache:
                for scan in (0,1):
                    ids=cache[f'indices{scan}'];record=raw[(metadata['pair'],scan)]
                    np.testing.assert_array_equal(cache[f'labels{scan}'],record['cd_type'][ids])
                    assert cache[f'available{scan}'].dtype==bool
                    if metadata['view']=='clean':
                        np.testing.assert_array_equal(cache[f'xyz{scan}'],np.column_stack([record[k][ids] for k in ('x','y','z')]))
                    assembled=arrays(out,fold,metadata['split'],metadata['view'])
                    count=len(ids);offset=(PAIRS.index(metadata['pair'])*2+scan)*count
                    probe=np.random.default_rng(5200+fold+scan).choice(count,8,replace=False)
                    for key in ('x','mask','geo','indices','labels','available'):
                        np.testing.assert_array_equal(assembled[key][offset+probe],cache[f'{key}{scan}'][probe])
                    assert (assembled['pair'][offset+probe]==int(metadata['pair'])).all()
                    assert (assembled['scan'][offset+probe]==scan).all();assembled_probes+=len(probe)
                    del assembled
            feature_views+=1
        for meta_path in (root/'learning').glob('*/frozen.json'):
            meta=json.loads(meta_path.read_text());folder=meta_path.parent
            model_path=folder/('model.joblib' if folder.name=='bounded_geometry' else 'best.pt')
            assert digest(model_path)==meta['checkpoint_sha256'] and digest(folder/'config.json')==meta['config_sha256']
            config=json.loads((folder/'config.json').read_text())
            for name,sha in config['source_sha256'].items():assert digest(folder/'source'/name)==sha
            assert not meta['test_accessed'];models_checked+=1
            frozen_models[folder.name]=meta
            if folder.name!='bounded_geometry':
                with np.load(folder/'validation_clean.npz') as z:
                    threshold,_=choose_threshold(z['labels'],z['probability'],z['pair']*2+z['scan'],THRESHOLDS)
                assert threshold['threshold']==meta['threshold']
                assert np.isclose(threshold['objective'],meta['validation_objective'],rtol=0,atol=1e-12)
                assert np.isclose(threshold['changed_f1'],meta['validation_changed_f1'],rtol=0,atol=1e-12)
        for receipt in (root/'evaluation').glob('*/*/complete.json'):
            dst=receipt.parent;meta=json.loads(receipt.read_text());assert digest(dst/'predictions.npz')==meta['predictions_sha256']
            assert meta['no_test_retuning'] and meta['selection_sha256']==digest(root/'learning/selection.json')
            assert set(meta['model_sha256'])==set(frozen_models)
            for name,sha in meta['model_sha256'].items():assert sha==frozen_models[name]['checkpoint_sha256']
            pair=dst.parent.name.split('_')[1];old=base_fold(fold)/f'evaluation/primary/pair_{pair}'/dst.name
            condition=json.loads((dst/'condition.json').read_text());assert condition['v2_guard_passed'] and condition['v2_total_support_guard_m']<5.
            with np.load(dst/'predictions.npz') as z,np.load(old/'predictions.npz') as previous:
                for scan in (0,1):
                    for key in ('labels','indices','xyz'):np.testing.assert_array_equal(z[f'{key}{scan}'],previous[f'{key}{scan}'])
                    np.testing.assert_array_equal(z[f'labels{scan}'],raw[(pair,scan)]['cd_type'][z[f'indices{scan}']])
                    np.testing.assert_array_equal(z[f'common_valid{scan}'],previous[f'm3c2_calibrated_prediction{scan}']>=0)
                    for name,model in frozen_models.items():
                        probability=z[f'{name}_probability{scan}']
                        assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                        np.testing.assert_array_equal(z[f'{name}_prediction{scan}'],(probability>=model['threshold']).astype(np.int8))
                labels=[z[f'labels{i}'] for i in (0,1)]
                metrics=pd.read_csv(dst/'metrics.csv')
                for row in metrics.to_dict('records'):
                    masks=[np.ones(len(y),bool) for y in labels]
                    if row['visibility']=='B':masks[0]=np.linalg.norm(z['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
                    if row['support']=='common_valid':masks=[m&z[f'common_valid{i}'] for i,m in enumerate(masks)]
                    prediction=[z[f"{row['method']}_prediction{i}"] for i in (0,1)]
                    calculated=pair_scores([y[m] for y,m in zip(labels,masks)],[p[m] for p,m in zip(prediction,masks)])
                    for key,value in calculated.items():assert np.isclose(row[key],value,rtol=0,atol=1e-12)
                    rows_checked+=1
            cases+=1
        assert len(list((root/'evaluation').glob('*/clean_seed42/complete.json')))==6
    assert cases==318 and feature_views==180 and models_checked==22,(cases,feature_views,models_checked)
    previous_manifest=json.loads((ROOT/'final_submission/submission_manifest.json').read_text())
    for name,sha in previous_manifest.items():assert digest(ROOT/'final_submission'/name)==sha
    result=dict(passed=True,completed_cases=cases,metric_rows_independently_recomputed=rows_checked,model_artifacts_verified=models_checked,
        training_validation_full_scan_feature_views_checked=feature_views,previous_result_files_preserved=len(preserved['files']),
        assembled_feature_and_original_id_alignment_probes=assembled_probes,
        original_clouds_preserved=len(audit['clouds']),same_original_test_ids_xyz_gt_as_previous=True,
        old_submission_files_preserved=len(previous_manifest),reports_and_presentations_untouched=True,
        validation_only_selection_recomputed=True,independent_final_site_test=False,
        intended_large_radius_coverage_independently_checked=True,
        limitations=['One site and four epochs','Prior test findings informed this follow-up; these tests are comparative',
            'Finite bank of two full-scan training perturbations','Custom multiscale network, not a published architecture reproduction'])
    write_json(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','inference','analyse','verify'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);p.add_argument('--pair',default='01')
    p.add_argument('--stress',action='store_true');a=p.parse_args()
    if a.stage=='features':cache_evaluation(a.out,a.fold,a.pair,a.stress)
    elif a.stage=='inference':inference(a.out,a.fold,a.pair,a.stress)
    elif a.stage=='analyse':analyse(a.out)
    else:verify(a.out)
