"""GT-free grouping of frozen sampled construction change predictions.

Components describe observed positive cores, not complete objects, quantities,
semantic construction activities, or completion percentages. Point scores/masks
are never modified. Labels are read only for a separate descriptive purity audit.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from learning_v3_control import ROOT,BASE,digest,write_json
from learning_v2_runtime import write_csv

DEFAULT=ROOT/'results/construction_change_regions_2026-10-26_v1'
PAIRS=('01','02','03','12','13','23')


def group_positives(xyz,prediction,original_ids,radius=.4,minimum=3):
    xyz=np.asarray(xyz);prediction=np.asarray(prediction);original_ids=np.asarray(original_ids)
    assert len(xyz)==len(prediction)==len(original_ids) and len(np.unique(original_ids))==len(original_ids)
    assert np.isin(prediction,[-1,0,1]).all() and np.isfinite(xyz).all()
    membership=np.full(len(xyz),-2,np.int32);membership[prediction<0]=-3;membership[prediction==1]=-1
    positive=np.flatnonzero(prediction==1);positive=positive[np.argsort(original_ids[positive],kind='stable')]
    if not len(positive):return membership,[]
    parent=np.arange(len(positive));rank=np.zeros(len(positive),np.int8)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in cKDTree(xyz[positive]).query_pairs(radius,output_type='ndarray'):
        a=find(a);b=find(b)
        if a==b:continue
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    groups={}
    for i in range(len(positive)):groups.setdefault(find(i),[]).append(positive[i])
    components=[np.asarray(v,np.int64) for v in groups.values() if len(v)>=minimum]
    components.sort(key=lambda ids:(-len(ids),*xyz[ids].mean(0),int(original_ids[ids].min())))
    for component,ids in enumerate(components):membership[ids]=component
    assert int((membership>=0).sum())+int((membership==-1).sum())==len(positive)
    return membership,components


def run(out=DEFAULT,source=BASE):
    out=Path(out);source=Path(source);out.mkdir(exist_ok=True,parents=True)
    if (out/'verification.json').exists():return
    plan=out/'design.json'
    if not plan.exists():
        write_json(plan,dict(source_study=str(source),point_model='Original v2 clean-validation-selected seed42 model separately in each spatial fold',
            grouping_radius_m=.4,minimum_sampled_positive_cores=3,parameter_source='Fixed medium feature radius and a small display-support minimum; no label-based tuning',
            prediction_changes=False,point_evaluation_masks_changed=False,ground_truth_used_for_grouping=False,
            membership_codes={'unknown_prediction':-3,'predicted_unchanged':-2,'ungrouped_positive':-1,'component_ids':'0 and above'},
            scope='Connected components of sampled original held-out predicted-positive cores. Buffered interiors have gaps; not full-cloud object segmentation.',
            scores='Mean model output scores are not calibrated probabilities of construction completion',
            temporal_scope='Per-pair earlier/later acquisition indices; no verified component identity association across pairs',
            construction_claim='Candidate added/removed geometry only. No installed-component labels, material quantities, or project completion percentages.',
            permanent_vs_temporary_work_verified=False,object_level_recall_available=False))
    plan_data=json.loads(plan.read_text());assert plan_data['source_study']==str(source)
    inventory=json.loads((source/'data_audit.json').read_text())['clouds'];summary=[];rows=[];preserved={};models={}
    for fold in range(5):
        root=source/f'fold_{fold}';choice=json.loads((root/'learning/selection.json').read_text())['primary_variant']+'_seed42'
        meta=json.loads((root/f'learning/{choice}/frozen.json').read_text());assert digest(root/f'learning/{choice}/best.pt')==meta['checkpoint_sha256']
        models[str(fold)]=dict(model=choice,seed=42,threshold=meta['threshold'],checkpoint_sha256=meta['checkpoint_sha256'])
    write_json(out/'model_manifest.json',models)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for pair in PAIRS:
        packed={k:[[],[]] for k in ('xyz','indices','labels','probability','prediction','fold')}
        for fold in range(5):
            root=source/f'fold_{fold}';selected=json.loads((root/'learning/selection.json').read_text())['primary_variant']+'_seed42'
            path=root/f'evaluation/pair_{pair}/clean_seed42/predictions.npz';receipt=json.loads((path.parent/'complete.json').read_text())
            assert digest(path)==receipt['predictions_sha256'];preserved[str(path)]=digest(path)
            with np.load(path) as z:
                for scan in (0,1):
                    for key in ('xyz','indices','labels'):packed[key][scan].append(z[f'{key}{scan}'])
                    for key in ('probability','prediction'):packed[key][scan].append(z[f'{selected}_{key}{scan}'])
                    np.testing.assert_array_equal(z[f'{selected}_prediction{scan}'],(z[f'{selected}_probability{scan}']>=models[str(fold)]['threshold']).astype(np.int8))
                    packed['fold'][scan].append(np.full(len(z[f'indices{scan}']),fold,np.int8))
        packed={k:[np.concatenate(v[scan]) for scan in (0,1)] for k,v in packed.items()}
        fig,axes=plt.subplots(1,2,figsize=(13,6),sharex=True,sharey=True)
        for scan,kind in ((0,'removed'),(1,'added')):
            xyz=packed['xyz'][scan];ids=packed['indices'][scan];prediction=packed['prediction'][scan]
            assert len(ids)==7500
            membership,components=group_positives(xyz,prediction,ids,plan_data['grouping_radius_m'],plan_data['minimum_sampled_positive_cores'])
            # Evaluation labels enter only after GT-free grouping is complete.
            labels=packed['labels'][scan];probability=packed['probability'][scan]
            np.savez_compressed(out/f'pair_{pair}_{kind}_cores.npz',xyz=xyz,original_indices=ids,prediction=prediction,
                model_score=probability,component_index=membership,spatial_fold=packed['fold'][scan],evaluation_labels=labels)
            ax=axes[scan];ax.scatter(xyz[:,0],xyz[:,1],s=2,c='#cbd5e1',alpha=.6)
            for component,indices in enumerate(components):
                points=xyz[indices];lo=points.min(0);hi=points.max(0);centroid=points.mean(0);name=f'{kind}_{component+1:03d}'
                rows.append(dict(pair=pair,earlier_epoch=int(pair[0]),later_epoch=int(pair[1]),cloud_epoch=int(pair[scan]),kind='candidate_'+kind,
                    region_id=name,sampled_positive_cores=len(indices),centroid_x=float(centroid[0]),centroid_y=float(centroid[1]),centroid_z=float(centroid[2]),
                    bbox_min_x=float(lo[0]),bbox_min_y=float(lo[1]),bbox_min_z=float(lo[2]),bbox_max_x=float(hi[0]),bbox_max_y=float(hi[1]),bbox_max_z=float(hi[2]),
                    mean_model_score=float(probability[indices].mean()),minimum_model_score=float(probability[indices].min()),
                    evaluation_changed_cores=int(labels[indices].sum()),evaluation_unchanged_cores=int((labels[indices]==0).sum()),
                    evaluation_point_purity=float(labels[indices].mean()),contributing_spatial_folds=','.join(map(str,np.unique(packed['fold'][scan][indices]))),
                    model_decision_thresholds=','.join(str(models[str(f)]['threshold']) for f in np.unique(packed['fold'][scan][indices]))))
                color=plt.get_cmap('turbo')(component/max(1,len(components)-1));ax.scatter(points[:,0],points[:,1],s=6,color=color)
                if component<8:
                    ax.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],fill=False,color=color,lw=1))
                    ax.annotate(name,(centroid[0],centroid[1]),fontsize=6,xytext=(3,3),textcoords='offset points')
            residual=membership==-1
            ax.scatter(xyz[residual,0],xyz[residual,1],s=7,c='#dc2626',marker='x',label='Ungrouped positives')
            ax.set_title(f'Candidate {kind} geometry | {len(components)} regions');ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.legend(fontsize=8,loc='lower left')
            cloud=next(c for c in inventory if c['pair']==pair and c['scan']==scan)
            summary.append(dict(pair=pair,kind='candidate_'+kind,evaluated_original_cores=len(ids),full_cloud_points=cloud['points'],sampling_coverage=len(ids)/cloud['points'],
                predicted_positive_cores=int((prediction==1).sum()),grouped_positive_cores=int((membership>=0).sum()),ungrouped_positive_cores=int(residual.sum()),
                detector_unknown_cores=int((prediction<0).sum()),candidate_regions=len(components),
                grouped_evaluation_changed_cores=int(labels[membership>=0].sum()),grouped_evaluation_unchanged_cores=int((labels[membership>=0]==0).sum())))
        fig.suptitle(f'Construction pair_{pair}: epoch {pair[0]} to {pair[1]} | sampled candidate change regions\n7500 original cores per scan across five folds; gaps are unscored. Colors identify groups, not work-item classes.',fontsize=11)
        fig.subplots_adjust(left=.055,right=.98,bottom=.12,top=.82,wspace=.18);fig.savefig(out/f'pair_{pair}_regions.png',dpi=170);plt.close(fig)
    write_csv(out/'candidate_regions.csv',rows);write_csv(out/'pair_summary.csv',summary)
    for path,sha in preserved.items():assert digest(path)==sha
    write_json(out/'source_predictions_manifest.json',preserved)
    write_json(out/'verification.json',dict(passed=True,pairs=6,sampled_point_observations=90000,source_prediction_files_unchanged=len(preserved),
        predicted_positive_membership_conserved=True,ground_truth_not_used_for_grouping=True,point_metrics_and_masks_unchanged=True,
        candidate_regions=len(rows),construction_completion_or_semantic_component_claims=False))
    print('Candidate construction regions complete',len(rows),'source files unchanged',len(preserved),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--source',type=Path,default=BASE);a=p.parse_args();run(a.out,a.source)
