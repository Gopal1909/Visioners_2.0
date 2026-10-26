"""Postflight verification and scientific plots for the completed follow-ups."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from learning_v3_control import ROOT,BASE,digest,write_json

V3=ROOT/'results/learning_v3_construction_2026-10-25_v1'
CAL=ROOT/'results/operating_point_followup_2026-10-26_v1'
REF=ROOT/'results/construction_curriculum_refinement_2026-10-26_v1'
REG=ROOT/'results/construction_change_regions_2026-10-26_v1'
VIEWS=('clean','mild','medium')


def direct_score(z,threshold):
    keep=z['available'];group=z['pair']*2+z['scan'];macro=[];changed=[];fp=tn=0
    for key in np.unique(group):
        m=keep&(group==key);y=z['labels'][m];p=z['probability'][m]>=threshold
        a=int(((y==1)&p).sum());b=int(((y==0)&p).sum());c=int(((y==1)&~p).sum());d=int(((y==0)&~p).sum())
        positive=2*a/max(1,2*a+b+c);negative=2*d/max(1,2*d+b+c)
        macro.append((positive+negative)/2);changed.append(positive);fp+=b;tn+=d
    return dict(binary_macro_f1=float(np.mean(macro)),changed_f1=float(np.mean(changed)),false_positive_rate=fp/max(1,fp+tn))


def verify_refinement():
    import torch
    from learning_v3_model import make_model,setup_device
    from learning_v2_model import probabilities
    from learning_v3_runtime import validation
    from learning_v3_audit import preserved
    plan=json.loads((REF/'design.json').read_text());frozen=json.loads((REF/'frozen.json').read_text());setup_device(plan['seed'],'cpu')
    assert not frozen['promotion_gate_passed'];assert frozen['new_test_cases_scored']==0
    assert digest(plan['source_checkpoint'])==plan['source_sha256'] and digest(frozen['checkpoint_path'])==frozen['checkpoint_sha256']
    for name,sha in json.loads((REF/'sources/manifest.json').read_text()).items():assert digest(REF/'sources'/name)==sha
    model=make_model(plan['source_candidate']);count=probes=0;max_delta=0.;valid=[]
    for epoch in range(5):
        folder=REF/f'epoch_{epoch:02d}';path=Path(plan['source_checkpoint']) if epoch==0 else folder/'checkpoint.pt'
        state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.eval();del state
        receipt=json.loads((folder/'validation_complete.json').read_text());data={}
        for view_index,view in enumerate(VIEWS):
            p=folder/f'validation_{view}.npz';assert digest(p)==receipt['probability_sha256'][view]
            with np.load(p) as z:data[view]={k:z[k] for k in z.files}
            source=validation(0,view)
            for key in ('labels','pair','scan','indices','available'):np.testing.assert_array_equal(data[view][key],source[key])
            ids=np.random.default_rng(4096+100*epoch+view_index).choice(np.flatnonzero(source['available']),32,replace=False)
            mini={k:np.asarray(source[k][ids]) for k in ('x','mask','geo','labels')}
            prediction=probabilities(model,mini,device='cpu',batch_size=32)
            delta=float(np.max(np.abs(prediction-data[view]['probability'][ids])));max_delta=max(max_delta,delta)
            np.testing.assert_allclose(prediction,data[view]['probability'][ids],atol=1e-6,rtol=0);probes+=len(ids);del source,mini
        rows=list(csv.DictReader((folder/'thresholds.csv').open()))
        eligible=[]
        for row in rows:
            threshold=float(row['threshold']);scores={v:direct_score(z,threshold) for v,z in data.items()}
            assert np.isclose(np.mean([x['binary_macro_f1'] for x in scores.values()]),float(row['mean_macro']),atol=1e-12,rtol=0)
            for view,value in scores.items():
                for key,number in value.items():assert np.isclose(number,float(row[f'{view}_{key}']),atol=1e-12,rtol=0)
            if scores['clean']['binary_macro_f1']>=plan['clean_floor']-1e-12 and scores['clean']['false_positive_rate']<=plan['fpr_ceiling']+1e-12:eligible.append(row)
            count+=1
        assert (receipt['selected'] is not None)==bool(eligible)
        valid.extend(eligible)
    assert valid and frozen['selection']['epoch']==0
    assert np.isclose(max(float(r['mean_macro']) for r in valid),frozen['selection']['mean_macro'],atol=1e-12,rtol=0)
    write_json(REF/'verification.json',dict(passed=True,validation_threshold_rows_recomputed=count,prediction_checkpoint_probes=probes,
        maximum_probability_probe_delta=max_delta,original_validation_ids_labels_and_availability_preserved=True,
        promotion_gate_not_relaxed=True,original_focal_checkpoint_retained=True,new_test_cases_scored=0,**preserved(V3)))
    print('Refinement verified',count,'threshold rows',probes,'prediction probes',flush=True)


def verify_calibration():
    plan=json.loads((CAL/'design.json').read_text());frozen=json.loads((CAL/'frozen_operating_points.json').read_text());checked=0
    for key,lock in plan['source_models'].items():
        folder=Path(lock['folder'])
        for filename,name in [('best.pt','best_sha256'),('last.pt','last_sha256'),('frozen.json','frozen_meta_sha256')]:assert digest(folder/filename)==lock[name]
        receipt=json.loads((CAL/'final_checkpoint_validation'/key/'complete.json').read_text());assert receipt['checkpoint_sha256']==lock['last_sha256']
        for view,sha in receipt['probability_sha256'].items():assert digest(CAL/'final_checkpoint_validation'/key/f'{view}.npz')==sha
        checked+=2
    for row in [*frozen['v2_selected'].values(),frozen['v3_fold0_selected']]:
        if row is None:continue
        key=f"{row['family']}_fold{row['fold']}_{row['candidate']}";folder=Path(plan['source_models'][key]['folder'])
        for view in VIEWS:
            path=folder/f'validation_{view}.npz' if row['checkpoint']=='best' else CAL/'final_checkpoint_validation'/key/f'{view}.npz'
            with np.load(path) as z:metrics=direct_score(z,row['threshold'])
            for name,value in metrics.items():assert np.isclose(value,row[f'{view}_{name}'],atol=1e-12,rtol=0)
    assert not frozen['v2_promotion_gate_passed'] and not frozen['v3_promotion_gate_passed']
    write_json(CAL/'postflight_verification_v2.json',dict(passed=True,source_checkpoint_files_verified=checked,source_fits_verified=len(plan['source_models']),
        selected_policy_metrics_independently_recomputed=True,no_promotion_after_failed_gate=True,new_test_cases_scored=0))


def verify_regions():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    plan=json.loads((REG/'design.json').read_text());groups=cores=positive=0
    model_manifest=json.loads((REG/'model_manifest.json').read_text())
    for path,sha in json.loads((REG/'source_predictions_manifest.json').read_text()).items():assert digest(path)==sha
    rows=list(csv.DictReader((REG/'candidate_regions.csv').open()))
    for pair in ('01','02','03','12','13','23'):
        for kind,scan in [('removed',0),('added',1)]:
            with np.load(REG/f'pair_{pair}_{kind}_cores.npz') as z:
                assert len(z['original_indices'])==7500 and len(np.unique(z['original_indices']))==7500
                source_fields={k:[] for k in ('xyz','original_indices','prediction','model_score','evaluation_labels','spatial_fold')}
                for fold in range(5):
                    model=model_manifest[str(fold)]['model']
                    with np.load(BASE/f'fold_{fold}/evaluation/pair_{pair}/clean_seed42/predictions.npz') as original:
                        for key,source_key in [('xyz',f'xyz{scan}'),('original_indices',f'indices{scan}'),('evaluation_labels',f'labels{scan}'),
                            ('prediction',f'{model}_prediction{scan}'),('model_score',f'{model}_probability{scan}')]:source_fields[key].append(original[source_key])
                        source_fields['spatial_fold'].append(np.full(len(original[f'indices{scan}']),fold,np.int8))
                for key,values in source_fields.items():np.testing.assert_array_equal(z[key],np.concatenate(values))
                indices=np.flatnonzero(z['prediction']==1);points=z['xyz'][indices];n=len(points)
                edges=cKDTree(points).query_pairs(plan['grouping_radius_m'],output_type='ndarray')
                adjacency=coo_matrix((np.ones(len(edges)*2,bool),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
                _,labels=connected_components(adjacency,directed=False);counts=np.bincount(labels)
                for label,size in enumerate(counts):
                    loc=indices[labels==label];membership=z['component_index'][loc]
                    if size<plan['minimum_sampled_positive_cores']:assert (membership==-1).all()
                    else:
                        assert (membership>=0).all() and len(np.unique(membership))==1;groups+=1
                        r=next(r for r in rows if r['pair']==pair and r['kind']=='candidate_'+kind and r['region_id']==f'{kind}_{int(membership[0])+1:03d}')
                        assert int(r['sampled_positive_cores'])==len(loc)
                        assert int(r['evaluation_changed_cores'])==int(z['evaluation_labels'][loc].sum())
                assert (z['component_index'][z['prediction']==0]==-2).all()
                assert int((z['component_index']>=0).sum())+int((z['component_index']==-1).sum())==n
                cores+=len(z['prediction']);positive+=n
    assert groups==len(rows)
    write_json(REG/'postflight_verification.json',dict(passed=True,independent_sparse_graph_components_verified=groups,
        original_point_observations=cores,predicted_positive_points_conserved=positive,point_scores_and_evaluation_masks_unchanged=True,
        component_point_purity_is_not_object_level_precision=True))
    print('Regions independently verified',groups,cores,positive,flush=True)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((REF/'design.json').read_text());rows=[]
    for epoch in range(5):
        row=next(r for r in csv.DictReader((REF/f'epoch_{epoch:02d}/thresholds.csv').open()) if float(r['threshold'])==.8)
        rows.append({k:float(v) for k,v in row.items()})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for view in VIEWS:axes[0].plot(range(5),[r[f'{view}_binary_macro_f1'] for r in rows],'o-',label=view)
    axes[0].plot(range(5),[r['mean_macro'] for r in rows],'o--',c='black',label='three-view mean')
    axes[0].axhline(plan['reference']['mean_macro'],color='#64748b',ls=':',label='v2 robust reference mean')
    axes[0].set_ylabel('Validation binary macro F1');axes[0].legend(fontsize=8)
    axes[1].plot(range(5),[100*r['clean_fpr'] for r in rows],'o-',c='#dc2626',label='clean false-positive rate')
    axes[1].axhline(100*plan['fpr_ceiling'],ls=':',color='#64748b',label='selection ceiling')
    axes[1].set_ylabel('Unchanged validation cores predicted changed (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Fine-tuning epoch (0 = original checkpoint)');ax.set_xticks(range(5));ax.grid(alpha=.2)
    fig.suptitle('Fixed threshold 0.8 diagnostic; model selection also searched the declared grid under clean-quality constraints',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(REF/'refinement_validation.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['refinement','calibration','regions','plots']);a=p.parse_args()
    {'refinement':verify_refinement,'calibration':verify_calibration,'regions':verify_regions,'plots':plots}[a.stage]()
