"""Bounded multiscale cross-scan features and real full-scan training views.

CPU-only: full point-cloud trees are never resident alongside CUDA libraries.
Original core indices and GT remain separate from voxel support points.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from multipair_data import ROOT, PAIRS, load_pair
from c2c_robustness import apply_rigid_transformation, apply_point_sparsity
from m3c2_baseline import digest, write_json, MemoryTimer

BASE = ROOT/'results/multipair_nested_2026-10-25_v1'
DEFAULT = ROOT/'results/learning_v2_coverage_2026-10-25_v1'
RADII = (.15, .4, .8)
VOXELS = (.025, .05, .075)
K = 64
MATCH_RADIUS = .25
SUPPORT_BOUND = max(r+np.sqrt(3)*v+MATCH_RADIUS for r,v in zip(RADII,VOXELS))
VIEWS = {'clean':(0.,0.,1.,0.), 'mild':(.1,1.,.7,.005), 'medium':(.2,2.,.4,.01)}
VARIANTS = ('multiscale_clean','multiscale_robust','multiscale_robust_hard')


def base_fold(fold):return BASE if fold==0 else BASE/f'spatial_cv/fold_{fold}'
def fold_dir(out,fold):return Path(out)/f'fold_{fold}'


def prepare(out=DEFAULT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'design.json'
    if path.exists():return json.loads(path.read_text())
    audit=json.loads((BASE/'data_audit.json').read_text())
    for row in audit['clouds']:assert digest(row['path'])==row['sha256']
    protected={}
    for p in BASE.rglob('*'):
        if p.is_file() and 'features' not in p.relative_to(BASE).parts and 'execution_logs' not in p.relative_to(BASE).parts:
            protected[str(p.relative_to(BASE)).replace('\\','/')]=digest(p)
    write_json(out/'baseline_preservation.json',dict(root=str(BASE),files=protected))
    maximum=max(json.loads(p.read_text()).get('maximum_observed_perturbation_m',0.)
        for p in (BASE/'evaluation/primary').glob('*/*/condition.json'))
    design=dict(baseline=str(BASE),scene='BikeParkingConstruction',pairs=list(PAIRS),folds=list(range(5)),
        variants=list(VARIANTS),seed=42,repeat_primary_fold0_seeds=[43,44],epochs=30,batch_size=128,
        radii_m=list(RADII),voxel_sizes_m=list(VOXELS),points_per_patch=K,
        patch_sampling='Complete radius query; retain nearest 16 and 48 evenly spaced distance ranks from remaining voxel points, including the farthest',
        development_revision='Prior nearest-64 development run preserved at results/learning_v2_2026-10-25_v1; measured widest patch median radius was 0.245 m. No test predictions were generated there.',
        cross_point_match_cap_m=MATCH_RADIUS,bounded_feature_support_m=float(SUPPORT_BOUND),
        maximum_previous_test_displacement_m=maximum,
        views={k:dict(translation_m=v[0],rotation_deg=v[1],retention=v[2],clipped_noise_sigma_m=v[3]) for k,v in VIEWS.items()},
        training_seed_rule='610000 + 100*int(pair) + 10*fold + view_id; validation base 810000',
        checkpoint='Minimum clean validation BCE for every neural candidate',
        threshold='Clean validation mean pair/direction binary macro F1; fixed previous threshold grid',
        primary_selection='Maximum clean validation binary macro F1; fixed variant order breaks ties',
        secondary_selection='Mean clean/mild/medium validation binary macro F1 at the same clean-calibrated threshold',
        hard_sampling='70% uniform + 30% running loss emphasis within each pair/class; original group size preserved; existing class/pair loss weights retained',
        robust_sampling='Each original core chooses clean/mild/medium with equal probability; an unavailable deleted target core falls back to its clean view',
        evaluation='Same previous original test IDs; 1500/scan clean in all five folds; all 288 previous fold0 stress cases',
        exposure='Previous held-out results informed this architectural follow-up. These are comparative same-site results, not an untouched final benchmark.',
        independent_site_available=False,temporal_reference_not_retuned=True,
        reports_and_presentations_untouched=True,
        architecture='Custom shared multiscale PointNet with bounded per-point cross-scan match channels and a geometry branch; not a PointNet++ or KPConv reproduction')
    write_json(path,design)
    shutil.copy2(BASE/'data_audit.json',out/'data_audit.json')
    for fold in range(5):
        dst=fold_dir(out,fold);dst.mkdir(exist_ok=True)
        shutil.copy2(base_fold(fold)/'protocol.json',dst/'protocol.json')
    print('Frozen learning-v2 design; preserved baseline files:',len(protected),flush=True)
    return design


def raw_moments(relative,mask):
    n=mask.sum(1).clip(1);p=relative*mask[...,None]
    mean=p.sum(1)/n[:,None];q=(relative-mean[:,None])*mask[...,None]
    cov=np.einsum('nki,nkj->nij',q,q)/n[:,None,None]
    eig=np.linalg.eigvalsh(cov).clip(0);rad=np.linalg.norm(p,axis=2)
    return np.column_stack((mean,np.sqrt(np.diagonal(cov,axis1=1,axis2=2).clip(0)),eig,
        rad.max(1),rad.sum(1)/n,mask.sum(1)/K)).astype(np.float32)


def bounded_nearest(tree,points):
    distance,ids=tree.query(points,distance_upper_bound=MATCH_RADIUS,workers=1)
    valid=np.isfinite(distance)
    delta=np.zeros((*points.shape[:-1],3),np.float32)
    if valid.any():delta[valid]=(tree.data[ids[valid]]-points[valid])/MATCH_RADIUS
    return np.minimum(distance,MATCH_RADIUS).astype(np.float32)/MATCH_RADIUS,delta,valid


def coverage_patches(tree,centers,radius):
    """Keep core detail and actual full-radius context without a nearest-K cap."""
    points=np.zeros((len(centers),K,3),np.float64);valid=np.zeros((len(centers),K),bool)
    neighborhoods=tree.query_ball_point(centers,radius,workers=1,return_sorted=False)
    for row,ids in enumerate(neighborhoods):
        if not len(ids):continue
        p=tree.data[np.asarray(ids,np.int64)];delta=p-centers[row]
        d2=np.einsum('ij,ij->i',delta,delta)
        order=np.lexsort((p[:,2],p[:,1],p[:,0],d2))
        if len(order)>K:
            central=16
            ranks=np.r_[np.arange(central),central+np.linspace(0,len(order)-central-1,K-central).round().astype(int)]
            order=order[ranks]
        points[row,:len(order)]=p[order];valid[row,:len(order)]=True
    return points,valid


def features(trees,contexts,centers,chunk=256):
    """GT-free, translation-equivariant, bounded geometry for every channel."""
    n=len(centers);x=np.zeros((n,3,2,K,8),np.float32);mask=np.zeros((n,3,2,K),bool)
    geometry=[]
    core_distance,core_delta,core_valid=bounded_nearest(trees[1],centers)
    for scale,radius in enumerate(RADII):
        stats=[]
        for side in (0,1):
            tree=contexts[scale][side]
            for start in range(0,n,chunk):
                end=min(n,start+chunk);c=centers[start:end]
                points,valid=coverage_patches(tree,c,radius)
                relative=(points-c[:,None,:])/radius
                d,delta,matched=bounded_nearest(trees[1-side],points.reshape(-1,3))
                channels=np.concatenate((relative,delta.reshape(-1,K,3),d.reshape(-1,K,1),matched.reshape(-1,K,1)),axis=-1)
                channels[~valid]=0.;x[start:end,scale,side]=channels;mask[start:end,scale,side]=valid
            stats.append(raw_moments(x[:,scale,side,:,:3],mask[:,scale,side]))
        counts=[np.log1p(t.query_ball_point(centers,radius,return_length=True,workers=1))/np.log1p(10000) for t in trees]
        a,b=stats
        geometry.append(np.column_stack((a,b,b-a,np.abs(b-a),*counts,core_distance,core_delta,core_valid)))
    geo=np.concatenate(geometry,axis=1).astype(np.float32)
    assert geo.shape==(n,165) and np.isfinite(x).all() and np.isfinite(geo).all()
    return dict(x=x,mask=mask,geo=geo)


def fixed_voxel_centroids(xyz,size):
    """A zero-anchored grid: remote geometry cannot move local voxel boundaries."""
    if not len(xyz):return np.empty((0,3),np.float32)
    q=np.floor(np.asarray(xyz,dtype=np.float64)/size).astype(np.int64)
    q-=q.min(0);dims=q.max(0)+1
    code=(q[:,0]*dims[1]+q[:,1])*dims[2]+q[:,2]
    _,inverse=np.unique(code,return_inverse=True);count=np.bincount(inverse)
    return np.column_stack([np.bincount(inverse,weights=xyz[:,j])/count for j in range(3)]).astype(np.float32)


def contexts_for(trees):
    return [[cKDTree(fixed_voxel_centroids(t.data,size)) for t in trees] for size in VOXELS]


def transform_training(xyz,view,seed,center):
    translation,rotation,retention,sigma=VIEWS[view]
    points=apply_rigid_transformation(xyz,translation,rotation,np.asarray(center),seed)
    ids=np.arange(len(points),dtype=np.int64)
    if retention<1:points,ids=apply_point_sparsity(points,ids,retention,seed+10000)
    if sigma:
        rng=np.random.default_rng(seed+20000)
        for start in range(0,len(points),100000):
            end=min(len(points),start+100000)
            points[start:end]+=rng.normal(size=(end-start,3)).clip(-3,3).astype(points.dtype)*sigma
    maximum=0.
    for start in range(0,len(points),100000):
        difference=points[start:start+100000].astype(np.float64)-xyz[ids[start:start+100000]]
        maximum=max(maximum,float(np.linalg.norm(difference,axis=1).max(initial=0.)))
    return points,ids,maximum


def cache_training(out=DEFAULT,fold=0,pair='01',split='train',view='clean'):
    out=Path(out);design=json.loads((out/'design.json').read_text());dst=fold_dir(out,fold)/'cache';dst.mkdir(exist_ok=True)
    name=f'pair_{pair}_{split}_{view}';path=dst/(name+'.npz');receipt=dst/(name+'.json')
    if receipt.exists():assert digest(path)==json.loads(receipt.read_text())['sha256'];return
    protocol=json.loads((fold_dir(out,fold)/'protocol.json').read_text())
    data=load_pair(pair)
    with np.load(base_fold(fold)/f'splits/pair_{pair}.npz') as z:
        wanted=[z[f'{split}_indices{i}'] for i in (0,1)]
    seed=(610000 if split=='train' else 810000)+100*int(pair)+10*fold+list(VIEWS).index(view)
    with MemoryTimer() as timer:
        xyz1,retained,maximum=transform_training(data[1]['xyz'],view,seed,protocol['stress_rotation_center_m'])
        guard=2*SUPPORT_BOUND+maximum+design['maximum_previous_test_displacement_m']
        assert guard<protocol['minimum_cross_split_core_distance_m'],('Feature/augmentation guard',guard)
        trees=[cKDTree(data[0]['xyz']),cKDTree(xyz1)];context=contexts_for(trees)
        result={}
        for scan in (0,1):
            ids=wanted[scan];available=np.ones(len(ids),bool);centers=data[scan]['xyz'][ids].astype(np.float64)
            if scan==1:
                positions=np.searchsorted(retained,ids);available=positions<len(retained)
                available[available]&=retained[positions[available]]==ids[available]
                centers[available]=xyz1[positions[available]]
                # Unavailable queries are cached only for row alignment and are
                # excluded from training/validation; their coordinates carry no claim.
            f=features([trees[scan],trees[1-scan]],[[a[scan],a[1-scan]] for a in context],centers[available])
            for key,value in f.items():
                aligned=np.zeros((len(ids),*value.shape[1:]),value.dtype);aligned[available]=value
                result[f'{key}{scan}']=aligned
            result[f'labels{scan}']=data[scan]['labels'][ids];result[f'indices{scan}']=ids
            result[f'available{scan}']=available;result[f'xyz{scan}']=centers
        np.savez_compressed(path,**result)
    write_json(receipt,dict(sha256=digest(path),pair=pair,fold=fold,split=split,view=view,seed=seed,
        maximum_observed_displacement_m=maximum,train_test_total_support_bound_m=float(guard),
        bounded_guard_passed=True,full_scan_transformed_before_rebuilding_trees=True,
        split_sha256=digest(base_fold(fold)/f'splits/pair_{pair}.npz'),design_sha256=digest(out/'design.json'),
        feature_source_sha256=digest(__file__),timing=timer.stats,test_predictions_accessed=False))
    print('V2 features',fold,pair,split,view,round(timer.stats['seconds'],1),'seconds',flush=True)


def assemble(out=DEFAULT,fold=0,split='train',view='clean'):
    folder=fold_dir(out,fold)/f'assembled/{split}_{view}';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():return
    sources={pair:digest(fold_dir(out,fold)/f'cache/pair_{pair}_{split}_{view}.npz') for pair in PAIRS}
    count=3000 if split=='train' else 1200
    for key in ('x','mask','geo','labels','indices','available','pair','scan'):
        mapping=None;offset=0
        for pair in PAIRS:
            with np.load(fold_dir(out,fold)/f'cache/pair_{pair}_{split}_{view}.npz') as z:
                for scan in (0,1):
                    value=np.full(count,int(pair) if key=='pair' else scan,np.int16) if key in ('pair','scan') else z[f'{key}{scan}']
                    assert len(value)==count
                    if mapping is None:mapping=np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',dtype=value.dtype,shape=(12*count,*value.shape[1:]))
                    mapping[offset:offset+count]=value;offset+=count
                    del value
        mapping.flush();del mapping;gc.collect()
    write_json(marker,dict(rows=12*count,source_cache_sha256=sources,design_sha256=digest(Path(out)/'design.json')))
    print('Assembled v2',fold,split,view,flush=True)


def arrays(out,fold,split,view):
    folder=fold_dir(out,fold)/f'assembled/{split}_{view}'
    assert (folder/'complete.json').exists()
    return {key:np.load(folder/f'{key}.npy',mmap_mode='r') for key in ('x','mask','geo','labels','indices','available','pair','scan')}


def check_coverage(out=DEFAULT):
    path=fold_dir(out,0)/'cache/pair_01_train_clean.npz';rows=[]
    with np.load(path) as z:
        for scale,radius in enumerate(RADII):
            measured=np.concatenate([z[f'geo{i}'][:,55*scale+9]*radius for i in (0,1)])
            rows.append(dict(nominal_radius_m=radius,median_selected_outer_radius_m=float(np.median(measured)),
                q90_selected_outer_radius_m=float(np.quantile(measured,.9))))
    assert rows[-1]['median_selected_outer_radius_m']>.6,'Largest patch still fails to cover the intended neighborhood'
    write_json(Path(out)/'measured_patch_coverage.json',dict(passed=True,source=str(path),sha256=digest(path),scales=rows))
    print('Verified physical patch coverage:',json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','cache','assemble','check-coverage'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--pair',default='01');p.add_argument('--split',default='train');p.add_argument('--view',default='clean')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='cache':cache_training(a.out,a.fold,a.pair,a.split,a.view)
    elif a.stage=='assemble':assemble(a.out,a.fold,a.split,a.view)
    else:check_coverage(a.out)
