"""Frozen construction-model inference on identical v2 original cores and contexts."""
import sys
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='inference' and '--worker' not in sys.argv:
    from learning_v3_control import inference_cli
    inference_cli()
    raise SystemExit(0)
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from learning_v3_runtime import BASE,DEFAULT,ROOT,PAIRS,CANDIDATES,fold_dir,digest,write_json,write_csv,setup


def worker(out,fold,pair,stress=False):
    import torch
    from learning_v3_model import make_model,execution_device,setup_device
    from learning_v2_model import probabilities
    from learning_v2_scores import metric_rows
    root=fold_dir(out,fold);selection=json.loads((root/'learning/selection.json').read_text())
    assert json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['pilot_gate_passed']
    assert all((fold_dir(out,i)/'learning/selection.json').exists() for i in range(5))
    device=execution_device(out);setup_device(42,device);names=[(c,42) for c in CANDIDATES]
    if fold==0:names.extend((selection['candidate'],s) for s in (43,44))
    models={}
    for candidate,seed in names:
        name=f'{candidate}_seed{seed}';folder=root/'learning'/name;meta=json.loads((folder/'frozen.json').read_text())
        assert digest(folder/'best.pt')==meta['checkpoint_sha256']
        model=make_model(candidate).to(device);state=torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);model.eval();del state;models[name]=(model,meta)
    folders=[p for p in sorted((BASE/f'fold_{fold}/evaluation/pair_{pair}').iterdir()) if p.is_dir() and
        ((p.name!='clean_seed42') if stress else p.name=='clean_seed42') and not (root/f'evaluation/pair_{pair}'/p.name/'complete.json').exists()]
    maximum=max(json.loads(p.read_text())['maximum_observed_training_displacement_m'] for p in (root/'epochs').glob('*/pair_*.json'))
    support=json.loads((out/'design.json').read_text())['support_bound_m']
    for src in folders[:8]:
        dst=root/f'evaluation/pair_{pair}'/src.name;dst.mkdir(parents=True,exist_ok=True)
        receipt=json.loads((src/'feature_complete.json').read_text());assert digest(src/'features.npz')==receipt['sha256']
        cond=json.loads((src/'condition.json').read_text());guard=2*support+2*maximum+cond['v2_test_support_displacement_bound_m'];assert guard<5
        with np.load(src/'features.npz') as z:saved={k:z[k] for k in z.files}
        labels=[saved[f'labels{i}'] for i in (0,1)];prob={};prediction={}
        for name,(model,meta) in models.items():
            prob[name]=[probabilities(model,{k:saved[f'{k}{i}'] for k in ('x','mask','geo','labels')},device=device,batch_size=32) for i in (0,1)]
            prediction[name]=[(p>=meta['threshold']).astype(np.int8) for p in prob[name]]
        with np.load(src/'predictions.npz') as old:common=[old[f'common_valid{i}'] for i in (0,1)]
        condition,seed=src.name.rsplit('_seed',1);rows=[]
        for visibility in ('A','B') if cond['kind']=='occlusion' else ('standard',):
            masks=[np.ones(len(y),bool) for y in labels]
            if visibility=='B':masks[0]=np.linalg.norm(saved['xyz0'][:,:2]-np.array(cond['blind_center_m'])[:2],axis=1)>cond['radius_m']
            metadata=dict(experiment='learning_v3',spatial_fold=fold,pair=pair,condition=condition,family=cond['kind'],severity=cond['severity'],perturbation_seed=int(seed),visibility=visibility)
            rows.extend(metric_rows(labels,prediction,common,metadata,masks,cond['target_requested_cores'],cond.get('requested_label_counts')))
        result={k:saved[k] for k in saved if k.startswith(('labels','indices','xyz'))}
        for i in (0,1):
            result[f'common_valid{i}']=common[i]
            for name in models:result[f'{name}_prediction{i}']=prediction[name][i];result[f'{name}_probability{i}']=prob[name][i]
        np.savez_compressed(dst/'predictions.npz',**result);write_csv(dst/'metrics.csv',rows)
        write_json(dst/'condition.json',dict(cond,v3_total_support_guard_m=guard))
        write_json(dst/'complete.json',dict(passed=True,predictions_sha256=digest(dst/'predictions.npz'),feature_sha256=receipt['sha256'],
            selection_sha256=digest(root/'learning/selection.json'),model_sha256={k:m['checkpoint_sha256'] for k,(_,m) in models.items()},
            no_test_retuning=True,source_sha256=digest(__file__)))
        print('V3 inference',fold,pair,src.name,flush=True)


def inference(out,fold,pair,stress=False):
    src=BASE/f'fold_{fold}/evaluation/pair_{pair}';dst=fold_dir(out,fold)/f'evaluation/pair_{pair}'
    def pending():return [p for p in src.iterdir() if p.is_dir() and ((p.name!='clean_seed42') if stress else p.name=='clean_seed42') and not (dst/p.name/'complete.json').exists()]
    while pending():
        command=[sys.executable,__file__,'inference','--out',str(out),'--fold',str(fold),'--pair',pair,'--worker']
        if stress:command.append('--stress')
        before=len(pending());subprocess.run(command,check=True);assert len(pending())<before


def analyse(out):
    import pandas as pd
    from multipair_analysis import spatial_support_summary,stress_condition_summary
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True);frames=[]
    for fold in range(5):
        selection=json.loads((fold_dir(out,fold)/'learning/selection.json').read_text())
        for p in (fold_dir(out,fold)/'evaluation').glob('*/*/metrics.csv'):
            frame=pd.read_csv(p,dtype={'pair':str});frames.append(frame)
            selected=frame[frame.method==selection['candidate']+'_seed42'].copy();selected['method']='v3_validation_selected';frames.append(selected)
    new=pd.concat(frames,ignore_index=True)
    previous=pd.read_csv(BASE/'analysis/comparison_all_metrics.csv',dtype={'pair':str})
    old=previous[previous.method.isin(['v2_validation_selected','v2_robust_validation_selected','previous_validation_selected','bounded_geometry','previous_c2c_calibrated','previous_geometry_boosted','previous_m3c2_calibrated','previous_m3c2_legacy'])]
    v2_choice=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    extra=[]
    for variant,alias in [(v2_choice['primary_variant'],'v2_validation_selected'),(v2_choice['secondary_robust_variant'],'v2_robust_validation_selected')]:
        part=previous[(previous.spatial_fold==0)&(previous.condition!='clean')&(previous.method==variant+'_seed42')].copy()
        part['method']=alias;extra.append(part)
    old=pd.concat([old,*extra],ignore_index=True)
    frame=pd.concat([new,old],ignore_index=True)
    from learning_v3_metrics import supplementary
    extra=pd.DataFrame([supplementary(row) for row in frame.to_dict('records')])
    for column in extra:frame[column]=extra[column]
    frame.to_csv(folder/'comparison_all_metrics.csv',index=False)
    clean=frame[(frame.condition=='clean')&(frame.support=='full_requested')];clean.to_csv(folder/'clean_per_pair_fold.csv',index=False)
    summary=clean.groupby('method').agg(binary_macro_f1=('selection_binary_macro_f1','mean'),changed_f1=('directional_changed_f1','mean'),
        mean_pair_three_class_f1=('pooled_three_class_macro_f1','mean'),mean_pair_three_class_miou=('pooled_three_class_miou','mean'),
        detector_unknown_rate=('detector_unknown_rate','mean'),pair_fold_count=('pair','size'))
    summary.to_csv(folder/'clean_comparison.csv');spatial_support_summary(clean).to_csv(folder/'class_support_and_pooled_confusions.csv',index=False)
    pooled=[]
    for method,part in clean.groupby('method'):
        counts={f'scan{i}_{key}':int(part[f'scan{i}_{key}'].sum()) for i in (0,1) for key in ('tp','fp','fn','tn','unknown_positive','unknown_negative')}
        pooled.append(dict(method=method,cases=len(part),**counts,**supplementary(counts)))
    write_csv(folder/'global_pooled_clean_metrics.csv',pooled)
    write_json(folder/'metric_interpretation.json',dict(pair_direction_means='Same zero-absent-class convention as v2; compare unchanged cohorts',
        global_pooling='Separate weighting from pair/direction mean; pool confusion counts before deriving three-class F1 and mIoU',
        unknowns='Retained as errors in full requested support; common-valid support saved separately',
        baselines='Frozen C2C and M3C2-style predictions reused on identical original cores; no CloudCompare equivalence claim',
        construction_progress='Geometric additions/removals only; no completion percentages or semantic work-item claims'))
    robust=frame[frame.spatial_fold==0].copy();robust['experiment']='primary';stress_condition_summary(robust).to_csv(folder/'stress_condition_summary.csv',index=False)
    clean.groupby(['spatial_fold','method']).agg(binary_macro_f1=('selection_binary_macro_f1','mean'),changed_f1=('directional_changed_f1','mean')).to_csv(folder/'clean_per_fold.csv')
    print(summary.to_string(),flush=True)
    plot_results(out,robust)


def plot_results(out,frame):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from multipair_metrics import pair_scores
    folder=out/'analysis';root=fold_dir(out,0)
    selected=json.loads((root/'learning/selection.json').read_text())['candidate']+'_seed42'
    old=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    policies=[('v3_validation_selected',root,selected),('v2_validation_selected',BASE/'fold_0',old['primary_variant']+'_seed42'),
        ('v2_robust_validation_selected',BASE/'fold_0',old['secondary_robust_variant']+'_seed42')]
    rows=[]
    for pair in PAIRS:
        with np.load(root/f'evaluation/pair_{pair}/translation_0.05_seed42/predictions.npz') as z:ids=[z[f'indices{i}'] for i in (0,1)]
        for method,location,model in policies:
            with np.load(location/f'evaluation/pair_{pair}/clean_seed42/predictions.npz') as z:
                positions=[np.searchsorted(z[f'indices{i}'],ids[i]) for i in (0,1)]
                for i in (0,1):np.testing.assert_array_equal(z[f'indices{i}'][positions[i]],ids[i])
                rows.append(dict(pair=pair,method=method,**pair_scores([z[f'labels{i}'][positions[i]] for i in (0,1)],
                    [z[f'{model}_prediction{i}'][positions[i]] for i in (0,1)])))
    matched=pd.DataFrame(rows);matched.to_csv(folder/'matched_clean_stress_cores.csv',index=False)
    families=[('translation','Translation (m)'),('rotation','Rotation (degrees)'),('retention','Retained fraction'),('noise','Noise sigma (m)'),('occlusion','Target-core deletion fraction')]
    for metric,label,filename in [('selection_binary_macro_f1','Binary macro F1','robustness_binary_macro.png'),('directional_changed_f1','Mean removal/addition F1','robustness_changed_f1.png')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8));baseline=matched.groupby('method')[metric].mean().to_dict()
        for ax,(family,xlabel) in zip(axes.flat,families):
            part=frame[(frame.family==family)&(frame.support=='full_requested')&(frame.visibility!='B')]
            for method,_,_ in policies:
                values=part[part.method==method].groupby(['severity','perturbation_seed'])[metric].mean().groupby('severity').agg(['mean','std'])
                x=np.r_[1. if family=='retention' else 0.,values.index];y=np.r_[baseline[method],values['mean']];sd=np.r_[0.,values['std'].fillna(0)];order=np.argsort(x)
                line=ax.plot(x[order],y[order],'o-',label=method)[0];ax.fill_between(x[order],(y-sd)[order],(y+sd)[order],color=line.get_color(),alpha=.12)
            ax.set_xlabel(xlabel);ax.set_ylabel(label);ax.set_ylim(0,1);ax.grid(alpha=.2)
        axes.flat[-1].axis('off');axes.flat[-1].text(0,1,'Fold0; six pairs from one construction site.\nClean endpoints use the same 512 original cores.\nShading: perturbation-seed SD, not a CI.\nOcclusion: strict A scoring; coverage saved separately.\n\nV3 model selected on validation only.\nSame-site follow-up comparisons; not an\nindependent construction-site benchmark.',va='top',fontsize=10)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(folder/filename,dpi=180);plt.close(fig)
    maps=folder/'crossvalidated_maps';maps.mkdir(exist_ok=True)
    for pair in PAIRS:
        packed={k:[[],[]] for k in ('xyz','labels','indices','v2','v3')}
        for fold in range(5):
            v3=json.loads((fold_dir(out,fold)/'learning/selection.json').read_text())['candidate']+'_seed42'
            v2=json.loads((BASE/f'fold_{fold}/learning/selection.json').read_text())['primary_variant']+'_seed42'
            with np.load(fold_dir(out,fold)/f'evaluation/pair_{pair}/clean_seed42/predictions.npz') as z,np.load(BASE/f'fold_{fold}/evaluation/pair_{pair}/clean_seed42/predictions.npz') as old:
                for i in (0,1):
                    for key in ('xyz','labels','indices'):packed[key][i].append(z[f'{key}{i}'])
                    packed['v2'][i].append(old[f'{v2}_prediction{i}']);packed['v3'][i].append(z[f'{v3}_prediction{i}'])
        packed={k:[np.concatenate(v[i]) for i in (0,1)] for k,v in packed.items()}
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True);palette={'TN':'#cbd5e1','TP':'#16a34a','FP':'#dc2626','FN':'#f59e0b'}
        for i in (0,1):
            assert len(np.unique(packed['indices'][i]))==7500;xyz=packed['xyz'][i];y=packed['labels'][i]
            for j,(key,title) in enumerate([('GT','Ground truth'),('v2','V2 validation-selected'),('v3','V3 validation-selected')]):
                ax=axes[i,j]
                if key=='GT':ax.scatter(xyz[:,0],xyz[:,1],c=np.where(y==1,'#dc2626','#94a3b8'),s=2)
                else:
                    p=packed[key][i]
                    for name,mask in [('TN',(y==0)&(p==0)),('TP',(y==1)&(p==1)),('FP',(y==0)&(p==1)),('FN',(y==1)&(p==0))]:ax.scatter(xyz[mask,0],xyz[mask,1],s=2,c=palette[name])
                ax.set_title(title+f' | scan {i}',fontsize=10);ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
        fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=k) for k,c in palette.items()],loc='lower center',ncol=4)
        fig.suptitle(f'Construction pair_{pair}: 7500 held-out original cores per scan across five folds\nBuffered interiors only; not a full-cloud map. GT red = changed.',fontsize=12)
        fig.subplots_adjust(left=.045,right=.99,bottom=.12,top=.84,wspace=.25,hspace=.48);fig.savefig(maps/f'pair_{pair}.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inference','analyse'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);p.add_argument('--pair',default='01');p.add_argument('--stress',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.stage=='analyse':analyse(a.out)
    elif a.worker:worker(a.out,a.fold,a.pair,a.stress)
    else:inference(a.out,a.fold,a.pair,a.stress)
