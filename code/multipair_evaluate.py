"""Frozen-model six-pair clean/stress evaluation with explicit support protocols."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from multipair_data import DEFAULT,PAIRS,load_pair
from multipair_features import voxel_centroids,extract_features
from multipair_metrics import pair_scores
from multipair_geometry import NORMALS
from m3c2_baseline import Config,MemoryTimer,estimate_normals,project_direction,classify,digest,write_json
from c2c_robustness import apply_rigid_transformation,apply_point_sparsity,apply_positional_noise,apply_spatial_occlusion

STRESS=[('translation_0.05','translation',.05),('translation_0.20','translation',.2),('translation_0.50','translation',.5),
    ('rotation_0.5','rotation',.5),('rotation_2','rotation',2.),('rotation_5','rotation',5.),
    ('retention_0.50','retention',.5),('retention_0.25','retention',.25),('retention_0.10','retention',.1),
    ('noise_0.01','noise',.01),('noise_0.03','noise',.03),('noise_0.075','noise',.075),
    ('occlusion_25','occlusion',.25),('occlusion_50','occlusion',.5),
    ('combined_mild','combined',.1),('combined_large','combined',.2)]


def temporal_geometry(out):
    path=out/'temporal_geometry_frozen.json'
    if path.exists():return json.loads(path.read_text())
    frame=pd.read_csv(out/'geometry_calibration/pair_01/candidate_metrics.csv')
    selected={}
    for method in ('C2C','M3C2-style'):
        row=frame[frame.method==method].sort_values(['selection_binary_macro_f1','coverage','candidate'],ascending=[False,False,True]).iloc[0]
        if method=='C2C':selected['c2c_threshold_m']=float(row.threshold_m)
        else:selected['m3c2_config']=asdict(Config(normal_radii_m=NORMALS[row.normal_set],
            projection_radii_m=(float(row.projection_radius_m),),half_length_m=float(row.half_length_m),minimum_effect_m=float(row.effect_floor_m)))
    result=dict(**selected,selection_pairs=['01'],test_pair='23',test_accessed=False,
        source_calibration_sha256=digest(out/'geometry_calibration/pair_01/candidate_metrics.csv'))
    write_json(path,result);return result


def load_models(out,protocol,ablation):
    import torch
    from siamese_pointnet import SiamesePointNet
    folder=out/f'learning/{protocol}'
    selected=json.loads((folder/'selection_frozen.json').read_text())
    models={};names=[f"{selected['selected_variant']}_seed{s}" for s in selected['seeds']]
    if ablation:names+=['nearest_seed42','voxel_seed42','voxel_aug_seed42']
    for name in dict.fromkeys(names):
        meta=json.loads((folder/name/'frozen.json').read_text())
        assert digest(folder/name/'best.pt')==meta['checkpoint_sha256']
        state=torch.load(folder/name/'best.pt',map_location='cpu',weights_only=False)
        model=SiamesePointNet().to('cuda');model.load_state_dict(state['model']);model.eval()
        models[name]=dict(model=model,threshold=meta['threshold'],variant='nearest' if name.startswith('nearest') else 'voxel',meta=meta)
        del state
    path=folder/'geometry_boosted/model.joblib';meta=json.loads((path.parent/'frozen.json').read_text())
    assert digest(path)==meta['checkpoint_sha256']
    models['geometry_boosted']=dict(model=joblib.load(path),threshold=meta['threshold'],variant='voxel',meta=meta)
    return models,selected


def case_cloud(data,kind,value,seed,center,reference_cores):
    xyz=data[1]['xyz'];ids=np.arange(len(xyz),dtype=np.int64);detail={}
    if kind=='clean':return xyz,ids,detail
    if kind=='translation':points=apply_rigid_transformation(xyz,value,0,center,seed)
    elif kind=='rotation':points=apply_rigid_transformation(xyz,0,value,center,seed)
    elif kind=='combined':points=apply_rigid_transformation(xyz,value,1. if value==.1 else 2.,center,seed)
    elif kind=='noise':points=apply_positional_noise(xyz,value,seed)
    elif kind=='retention':points,ids=apply_point_sparsity(xyz,ids,value,seed)
    elif kind=='occlusion':
        # A seeded test-core location, with radius set from geometry only.
        blind_center=reference_cores[np.random.default_rng(seed).integers(len(reference_cores))].astype(np.float64)
        radial=np.linalg.norm(reference_cores[:,:2]-blind_center[:2],axis=1)
        radius=float(np.quantile(radial,value))
        points,ids,removed,percent=apply_spatial_occlusion(xyz,ids,radius,blind_center)
        detail=dict(blind_center_m=blind_center.tolist(),radius_m=radius,full_cloud_deleted_fraction=percent/100,
            desired_test_core_deleted_fraction=value,full_cloud_deleted_count=removed)
    else:raise ValueError(kind)
    if kind in ('translation','rotation','combined','noise'):
        maximum=0.
        for start in range(0,len(points),100000):
            delta=points[start:start+100000].astype(np.float64)-xyz[ids[start:start+100000]]
            maximum=max(maximum,float(np.sqrt(np.einsum('ij,ij->i',delta,delta).max())))
        detail['maximum_observed_perturbation_m']=maximum
    return points,ids,detail


def metric_rows(labels,predictions,common,base,protocol_mask,requested_target,requested_label_counts=None):
    rows=[]
    eligible_labels=[y[m] for y,m in zip(labels,protocol_mask)]
    common_mask=[valid&mask for valid,mask in zip(common,protocol_mask)]
    eligible_count=sum(int(m.sum()) for m in protocol_mask)
    for method,pred in predictions.items():
        unknown=sum(int(((p<0)&mask).sum()) for p,mask in zip(pred,protocol_mask))/max(1,eligible_count)
        for support,masks in [('full_requested',protocol_mask),('common_valid',common_mask)]:
            y=[a[m] for a,m in zip(labels,masks)];p=[a[m] for a,m in zip(pred,masks)]
            row=dict(**base,method=method,support=support,**pair_scores(y,p))
            row['evaluation_coverage']=sum(int(m.sum()) for m in masks)/max(1,eligible_count)
            row['detector_unknown_rate']=unknown
            row['source0_protocol_excluded']=int((~protocol_mask[0]).sum())
            row['target_requested_cores']=requested_target;row['target_retained_cores']=len(labels[1])
            initial_total=len(labels[0])+requested_target
            correct=sum(int(((p==y)&m).sum()) for p,y,m in zip(pred,labels,masks))
            row['correct_over_initial_requested_cores']=correct/max(1,initial_total)
            row['evaluation_coverage_initial_requested_cores']=sum(int(m.sum()) for m in masks)/max(1,initial_total)
            for i in (0,1):
                for label in (0,1):
                    denom=int(((labels[i]==label)&protocol_mask[i]).sum())
                    row[f'scan{i}_gt{label}_evaluation_coverage']=float(((labels[i]==label)&masks[i]).sum()/denom) if denom else None
                    requested=(requested_label_counts[i][label] if requested_label_counts else int((labels[i]==label).sum()))
                    row[f'scan{i}_requested_gt{label}']=requested
                    row[f'scan{i}_retained_gt{label}']=int((labels[i]==label).sum())
                    row[f'scan{i}_protocol_eligible_gt{label}']=denom
                    row[f'scan{i}_evaluated_gt{label}']=int(((labels[i]==label)&masks[i]).sum())
            rows.append(row)
    return rows


def geometry_pass(out=DEFAULT,protocol='primary',stress=False,seeds=(42,43,44),only=None,pairs=None):
    out=Path(out)
    # The requested model-selection track must finish before exposing test scores.
    assert (out/f'learning/{protocol}/selection_frozen.json').exists()
    cfgmeta=json.loads((out/'geometry_frozen.json').read_text()) if protocol=='primary' else temporal_geometry(out)
    cfg=Config(**cfgmeta['m3c2_config']);design=json.loads((out/'protocol.json').read_text())
    folder=out/f'evaluation/{protocol}';folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'experiment_design.json').exists():
        write_json(folder/'experiment_design.json',dict(protocol=protocol,seeds=list(seeds),conditions=[['clean','clean',0.]]+STRESS,
            source_sha256=digest(__file__),geometry=cfgmeta,protocol_sha256=digest(out/'protocol.json'),
            frozen_learning_sha256=digest(out/f'learning/{protocol}/selection_frozen.json'),
            support='Full clouds as support; uniform original test cores. Common-valid and full-requested views.',
            occlusion='Synthetic XY deletion; A strict source0 scoring, B mask source0 within known blind cylinder. No unobserved target predictions.',
            absent_changed_class='Changed F1 returns zero when its denominator is zero; positive counts and false-positive counts are retained.'))
    selected=json.loads((out/f'learning/{protocol}/selection_frozen.json').read_text())
    variants={'voxel','nearest'}  # include the clean-only/augmentation ablation under stress too
    if stress and not (folder/'stress_sampling_design.json').exists():
        write_json(folder/'stress_sampling_design.json',dict(cores_per_scan=512,
            rule='Fixed uniform subset of clean test original IDs; same subset for all perturbations and methods',
            seed_rule='7025 + 2*int(pair) + scan; independent of perturbation seed',
            model_policy='Selected architecture at seeds42/43/44 plus nearest, voxel-clean and voxel-augmentation controls at seed42',
            purpose='Bounded repeated stress study; clean/current deltas intersect these original IDs'))
    for pair in (pairs or (PAIRS if protocol=='primary' else ('23',))):
        planned=[('clean',42)] if not stress else [(n,s) for n,_,_ in STRESS for s in seeds]
        pending=[(n,s) for n,s in planned if (not only or n in only) and
            not (folder/f'pair_{pair}/{n}_seed{s}/geometry_complete.json').exists()]
        if not pending:continue
        data=load_pair(pair);normal0_cache={}
        with np.load(out/f'splits/pair_{pair}.npz' if protocol=='primary' else out/'splits/temporal_test_pair23.npz') as z:
            wanted=[z[f'test_indices{i}' if protocol=='primary' else f'indices{i}'] for i in (0,1)]
        if stress:
            wanted=[np.sort(ids[np.random.default_rng(7025+2*int(pair)+i).choice(len(ids),min(512,len(ids)),replace=False)]) for i,ids in enumerate(wanted)]
        reference_cores=data[1]['xyz'][wanted[1]]
        tree0=cKDTree(data[0]['xyz']);context0=None
        cases=[('clean','clean',0.,42)] if not stress else [(n,k,v,s) for n,k,v in STRESS for s in seeds]
        for name,kind,value,seed in cases:
            if only and name not in only:continue
            dst=folder/f'pair_{pair}'/f'{name}_seed{seed}';dst.mkdir(parents=True,exist_ok=True)
            if (dst/'geometry_complete.json').exists():continue
            print('TEST',protocol,pair,name,'seed',seed,flush=True)
            with MemoryTimer() as timer:
                xyz1,retained,detail=case_cloud(data,kind,value,seed,np.array(design['stress_rotation_center_m']),reference_cores)
                support_sum=design['learning_support_bound_m']+design['geometry_support_halo_bound_m']+detail.get('maximum_observed_perturbation_m',0.)
                assert support_sum<design['minimum_cross_split_core_distance_m'],'Perturbed receptive fields can cross split guards'
                detail['measured_support_guard_passed']=True
                detail['context_plus_observed_displacement_m']=support_sum
                position=np.searchsorted(retained,wanted[1]);keep=position<len(retained)
                keep[keep]&=retained[position[keep]]==wanted[1][keep]
                chosen=[wanted[0],wanted[1][keep]]
                cores=[data[0]['xyz'][chosen[0]].astype(np.float64),xyz1[position[keep]].astype(np.float64)]
                labels=[data[i]['labels'][chosen[i]] for i in (0,1)]
                trees=[tree0,cKDTree(xyz1)]
                distance=[trees[1-i].query(cores[i],workers=1)[0] for i in (0,1)]
                predictions={'c2c_calibrated':[(d>cfgmeta['c2c_threshold_m']).astype(np.int8) for d in distance]}
                predictions['c2c_legacy_075']=[(d>.075).astype(np.int8) for d in distance]
                saved={};timing={}
                configs={'m3c2_calibrated':cfg}
                if not stress:configs['m3c2_legacy']=Config(projection_radii_m=(.1,))
                reused_normal0_before_case=bool(normal0_cache)
                for method,config in configs.items():
                    pred=[]
                    for i in (0,1):
                        t0=time.perf_counter()
                        if i==0 and method in normal0_cache:norm=normal0_cache[method]
                        else:norm=estimate_normals(trees[i],cores[i],config)
                        if i==0:normal0_cache[method]=norm
                        raw=project_direction(trees[i],trees[1-i],cores[i],norm,config)
                        decision=classify(raw,config.registration_error_m,config.minimum_effect_m)
                        pred.append(decision['prediction'][:,0])
                        timing[f'{method}_scan{i}']=time.perf_counter()-t0
                        for key,val in raw.items():saved[f'{method}_scan{i}_{key}']=val
                        for key,val in norm.items():saved[f'{method}_scan{i}_normal_{key}']=val
                        for key,val in decision.items():saved[f'{method}_scan{i}_decision_{key}']=val[:,0]
                    predictions[method]=pred
                common=[p>=0 for p in predictions['m3c2_calibrated']]
                if context0 is None:context0=cKDTree(voxel_centroids(data[0]['xyz']))
                context=[context0,cKDTree(voxel_centroids(xyz1))]
                for variant in variants:
                    feature_save={}
                    for i in (0,1):
                        t0=time.perf_counter()
                        f=extract_features([trees[i],trees[1-i]],[context[i],context[1-i]],cores[i],variant)
                        for key,val in f.items():feature_save[f'{key}_{i}']=val
                        timing[f'features_{variant}_scan{i}']=time.perf_counter()-t0
                    np.savez_compressed(dst/f'features_{variant}.npz',**feature_save)
                    del feature_save,f;gc.collect()
                for i in (0,1):
                    saved[f'indices{i}']=chosen[i];saved[f'xyz{i}']=cores[i];saved[f'labels{i}']=labels[i]
                    saved[f'c2c_distance{i}']=distance[i]
                    for method,pred in predictions.items():saved[f'{method}_prediction{i}']=pred[i]
                np.savez_compressed(dst/'geometry_measurements.npz',**saved)
                write_json(dst/'condition.json',dict(kind=kind,severity=value,seed=seed,**detail,
                    original_source_points=len(data[0]['xyz']),retained_target_points=len(xyz1),
                    rotation_center_m=design['stress_rotation_center_m'],target_requested_cores=len(wanted[1]),
                    requested_label_counts=[[int((data[i]['labels'][wanted[i]]==label).sum()) for label in (0,1)] for i in (0,1)]))
                write_json(dst/'method_seconds.json',timing)
            write_json(dst/'geometry_complete.json',dict(passed=True,measurement_sha256=digest(dst/'geometry_measurements.npz'),
                feature_sha256={v:digest(dst/f'features_{v}.npz') for v in variants},
                experiment_design_sha256=digest(folder/'experiment_design.json'),process_timing=timer.stats,
                evaluator_sha256=digest(__file__),normal0_cache_reused=reused_normal0_before_case,normal1_reestimated=True,
                no_stress_retuning=True,test_indices_original=True))
            print('Completed',pair,name,seed,round(timer.stats['seconds'],1),'seconds',flush=True)
            del xyz1,retained,cores,trees,context,saved,raw,norm,predictions,decision;gc.collect()
        del data,tree0,context0,normal0_cache;gc.collect()
    collect(out)


def inference_pass(out=DEFAULT,protocol='primary',stress=False,seeds=(42,43,44),only=None,pairs=None):
    import torch
    from siamese_pointnet import predict,setup
    setup(42);out=Path(out)
    folder=out/f'evaluation/{protocol}';models,selected=load_models(out,protocol,ablation=True)
    for pair_folder in sorted(folder.glob('pair_*')):
        pair=pair_folder.name.split('_')[1]
        if pairs and pair not in pairs:continue
        for dst in sorted(p for p in pair_folder.iterdir() if p.is_dir()):
            name,seed_text=dst.name.rsplit('_seed',1);seed=int(seed_text)
            if stress==(name=='clean') or (stress and seed not in seeds) or (only and name not in only):continue
            if (dst/'complete.json').exists() or not (dst/'geometry_complete.json').exists():continue
            receipt=json.loads((dst/'geometry_complete.json').read_text())
            assert digest(dst/'geometry_measurements.npz')==receipt['measurement_sha256']
            with MemoryTimer() as timer:
                with np.load(dst/'geometry_measurements.npz') as z:saved={k:z[k] for k in z.files}
                predictions={k[:-len('_prediction0')]:[saved[k],saved[k[:-1]+'1']] for k in saved if k.endswith('_prediction0')}
                labels=[saved[f'labels{i}'] for i in (0,1)];cores=[saved[f'xyz{i}'] for i in (0,1)]
                timing=json.loads((dst/'method_seconds.json').read_text())
                for variant in set(m['variant'] for m in models.values()):
                    path=dst/f'features_{variant}.npz';assert digest(path)==receipt['feature_sha256'][variant]
                    with np.load(path) as z:features=[{k:z[f'{k}_{i}'] for k in ('p0','m0','p1','m1','aux','geo')} for i in (0,1)]
                    for method,m in models.items():
                        if m['variant']!=variant:continue
                        start=time.perf_counter()
                        probability=[m['model'].predict_proba(f['geo'])[:,1] if method=='geometry_boosted' else predict(m['model'],f,'cuda') for f in features]
                        predictions[method]=[(p>=m['threshold']).astype(np.int8) for p in probability]
                        for i in (0,1):saved[f'{method}_probability{i}']=probability[i]
                        timing[method+'_inference']=time.perf_counter()-start
                    del features;gc.collect()
                for method,prediction in predictions.items():
                    for i in (0,1):saved[f'{method}_prediction{i}']=prediction[i]
                condition=json.loads((dst/'condition.json').read_text());kind=condition['kind'];rows=[]
                common=[p>=0 for p in predictions['m3c2_calibrated']]
                for visibility in ('A','B') if kind=='occlusion' else ('standard',):
                    mask=[np.ones(len(y),bool) for y in labels]
                    if visibility=='B':mask[0]=np.linalg.norm(cores[0][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
                    base=dict(experiment=protocol,pair=pair,condition=name,family=kind,severity=condition['severity'],perturbation_seed=seed,visibility=visibility)
                    rows.extend(metric_rows(labels,predictions,common,base,mask,condition['target_requested_cores'],condition.get('requested_label_counts')))
                np.savez_compressed(dst/'predictions.npz',**saved)
                pd.DataFrame(rows).to_csv(dst/'metrics.csv',index=False);write_json(dst/'method_seconds.json',timing)
            write_json(dst/'complete.json',dict(passed=True,measurement_sha256=digest(dst/'predictions.npz'),
                experiment_design_sha256=digest(folder/'experiment_design.json'),process_timing=timer.stats,
                geometry_process_timing=receipt['process_timing'],evaluator_sha256=digest(__file__),
                no_stress_retuning=True,test_indices_original=True))
            print('Inference complete',protocol,pair,name,seed,flush=True)
    collect(out)


def evaluate(out=DEFAULT,protocol='primary',stress=False,seeds=(42,43,44),only=None,pairs=None):
    import subprocess,sys
    args=['--out',str(out),'--protocol',protocol,'--seeds',*map(str,seeds)]
    if stress:args+=['--stress']
    if only:args+=['--only',*only]
    if pairs:args+=['--pairs',*pairs]
    # Release Windows/SciPy allocator reservations between full-cloud pairs.
    # Every case keeps the same coordinates, RNG seeds and frozen parameters.
    # A long-lived CPU worker exhausted commit on entering its fourth pair.
    for pair in (pairs or (PAIRS if protocol=='primary' else ('23',))):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),*args,'--worker','geometry','--pairs',pair],check=True)
    # The inference worker also encountered a Windows allocation failure.
    # Bound its process lifetime to one pair, matching CPU worker isolation.
    for pair in (pairs or (PAIRS if protocol=='primary' else ('23',))):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),*args,'--worker','inference','--pairs',pair],check=True)


def collect(out=DEFAULT):
    out=Path(out);frames=[]
    for path in (out/'evaluation').glob('*/*/*/metrics.csv'):
        if (path.parent/'complete.json').exists():frames.append(pd.read_csv(path,dtype={'pair':str}))
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'evaluated_metrics.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--protocol',choices=['primary','temporal'],default='primary');p.add_argument('--stress',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--only',nargs='+');p.add_argument('--pairs',nargs='+')
    p.add_argument('--worker',choices=['geometry','inference'])
    a=p.parse_args();function=geometry_pass if a.worker=='geometry' else inference_pass if a.worker=='inference' else evaluate
    function(a.out,a.protocol,a.stress,a.seeds,a.only,a.pairs)
