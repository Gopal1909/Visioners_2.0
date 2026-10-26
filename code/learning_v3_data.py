"""Fresh original construction cores every epoch; CPU-only full-scan augmentation."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from learning_v3_runtime import BASE,DEFAULT,ROOT,PAIRS,CANDIDATES,EPOCHS,CORES,fold_dir,digest,write_json,MemoryTimer
from learning_v3_kernels import features,SUPPORT_BOUND,split_mask,apply_rigid_transformation,apply_point_sparsity
from learning_v3_support import load_mapped_pair,contexts as mapped_contexts,mapped_transform,support64


def stream_npz(path,arrays):
    """Atomic NPZ with bounded 256 KiB compression buffers under host commit limits."""
    path=Path(path);temporary=path.with_name(path.stem+'_writing.npz')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for key,value in arrays.items():
            value=np.asarray(value);assert value.flags.c_contiguous and not value.dtype.hasobject
            with archive.open(key+'.npy','w',force_zip64=True) as entry:
                np.lib.format.write_array_header_1_0(entry,dict(descr=np.lib.format.dtype_to_descr(value.dtype),fortran_order=False,shape=value.shape))
                flat=value.reshape(-1);step=max(1,262144//value.dtype.itemsize)
                for begin in range(0,len(flat),step):entry.write(flat[begin:begin+step].tobytes())
    temporary.replace(path)


def balanced_ids(labels,eligible,count,rng):
    """Training-only class balancing; replacement only for undersized classes."""
    present=[np.flatnonzero(eligible&(labels==label)) for label in (0,1)]
    present=[ids for ids in present if len(ids)]
    if not present:raise ValueError('No eligible training points')
    sizes=[count//len(present)+(i<count%len(present)) for i in range(len(present))]
    return np.concatenate([rng.choice(ids,n,replace=len(ids)<n) for ids,n in zip(present,sizes)])


def augmentation_parameters(seed,attempt=0):
    rng=np.random.default_rng(seed);factor=.7**attempt
    return dict(translation_m=float(rng.choice([0.,.03,.08,.12]))*factor,
        rotation_deg=float(rng.choice([0.,.4,.8,1.2]))*factor,
        retention=float(rng.choice([.4,.7,1.])),
        noise_sigma_m=float(rng.choice([0.,.005,.015,.03]))*factor)


def transformed(xyz,parameters,seed,center):
    p=apply_rigid_transformation(xyz,parameters['translation_m'],parameters['rotation_deg'],center,seed)
    ids=np.arange(len(xyz),dtype=np.int64)
    p,ids=apply_point_sparsity(p,ids,parameters['retention'],seed+10000)
    rng=np.random.default_rng(seed+20000);maximum=0.
    for begin in range(0,len(p),100000):
        end=min(len(p),begin+100000)
        if parameters['noise_sigma_m']:p[begin:end]+=rng.normal(size=(end-begin,3)).clip(-3,3).astype(p.dtype)*parameters['noise_sigma_m']
        maximum=max(maximum,float(np.linalg.norm(p[begin:end].astype(np.float64)-xyz[ids[begin:end]],axis=1).max(initial=0)))
    return p,ids,maximum


def prepare(out=DEFAULT):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'design.json').exists():return
    assert json.loads((out/'sanity_gate.json').read_text())['passed']
    maximum=json.loads((BASE/'design.json').read_text())['maximum_previous_test_displacement_m']
    design=dict(scene='BikeParkingConstruction',pairs=list(PAIRS),folds=list(range(5)),epochs=EPOCHS,
        candidates=list(CANDIDATES),cores_per_pair_scan_epoch=CORES,seed=42,effective_batch_size=128,microbatch_size=64,
        sampling='Fresh class-balanced original cores from fixed training-cell interiors per epoch; replacement only for undersized present classes. Equal pair contribution.',
        augmentation='One independently sampled full-scan target perturbation per epoch/pair; one third of cores clean, two thirds augmented. Rebuild target voxel contexts and bounded NN channels.',
        magnitudes=dict(translation_m=[0,.03,.08,.12],rotation_deg=[0,.4,.8,1.2],retention=[.4,.7,1.],noise_sigma_m=[0,.005,.015,.03]),
        augmentation_guard='If full-scan measured bound fails, scale rigid/noise magnitudes by 0.7 per attempt, at most 12 attempts; retention stays fixed. Noise is clipped at 3 sigma.',
        support_bound_m=float(SUPPORT_BOUND),maximum_test_displacement_m=maximum,
        guard_formula='2*support_bound + 2*observed_training_displacement + observed_test_displacement < 5m',
        architecture='Custom 64-point to 16-anchor masked local aggregation with learned cross-scan attention at three scales, plus the existing bounded geometry branch. Not KPConv/PointNet++ reproduction.',
        compatibility='EF-Siamese KPConv compiled dependencies are absent; existing PyTorch implementation used without environment changes.',
        checkpoint='Minimum clean validation unweighted BCE; cosine learning rate 0.0006 to 0.00006, AdamW wd 0.01, dropout 0.25, clip norm 1',
        threshold='Same frozen v2 threshold grid and clean validation binary macro F1 calibration',
        primary_selection='Highest mean clean/mild/medium validation binary macro F1, candidate-order tie break',
        pilot_gate='Fold0 winner clean validation macro >= v2 robust-selected clean minus 0.005, and three-view mean macro >= v2 robust-selected mean plus 0.01. No test scoring if gate fails.',
        expanded_gate='If pilot passes, same three candidates in all five folds; no tests until all choices frozen. Repeat fold0 selected candidate seeds43/44; identical clean and stress features.',
        exposure='All previous same-site test findings informed this follow-up; comparative results only. No new independent construction site is available.',
        scope='Construction change detection first; semantic completion/progress claims require separate event annotations and planned scope.',
        office_and_open_space='Not used in this experiment; no car-parking downloads or training.',
        synthetic_or_pseudo_labels=False,reports_and_presentations_untouched=True)
    write_json(out/'design.json',design)
    preservation={}
    for p in BASE.rglob('*'):
        rel=p.relative_to(BASE)
        if p.is_file() and not any(x in rel.parts for x in ('assembled','cache','execution_logs')) and p.name!='features.npz':
            preservation[str(rel).replace('\\','/')]=digest(p)
    write_json(out/'baseline_preservation.json',dict(root=str(BASE),files=preservation))
    shutil.copy2(BASE/'data_audit.json',out/'data_audit.json')
    for fold in range(5):
        root=fold_dir(out,fold);root.mkdir(exist_ok=True)
        shutil.copy2(BASE/f'fold_{fold}/protocol.json',root/'protocol.json')
    write_json(out/'compatibility_gate.json',dict(passed=True,implementation='Existing native PyTorch; no environment installs',
        absent_libraries=['torch_geometric','torch_cluster','torch_scatter','torch_points3d','torch_points_kernels'],
        published_reference='https://github.com/IdeGelis/torch-points3d-SiamKPConvVariants',published_model_reproduced=False))
    print('V3 frozen design; original construction split regions retained',flush=True)


def cache_pair(out=DEFAULT,fold=0,pair='01',epoch_chunk=2):
    out=Path(out);protocol=json.loads((fold_dir(out,fold)/'protocol.json').read_text())
    design=json.loads((out/'design.json').read_text());root=fold_dir(out,fold)/'epochs'
    pending=[e for e in range(EPOCHS) if not (root/f'epoch_{e:02d}/pair_{pair}.json').exists()]
    if not pending:return
    data=load_mapped_pair(out,pair);eligible=[split_mask(d['xyz'],protocol,'train') for d in data]
    trees=[cKDTree(d['support'],copy_data=False) for d in data];contexts=mapped_contexts(trees)
    for epoch in pending[:epoch_chunk]:
        folder=root/f'epoch_{epoch:02d}';folder.mkdir(exist_ok=True,parents=True)
        seed=735000+fold*10000+int(pair)*100+epoch
        with MemoryTimer() as timer:
            for attempt in range(12):
                params=augmentation_parameters(seed,attempt)
                points,retained,maximum=mapped_transform(data[1]['xyz'],params,seed,np.array(protocol['stress_rotation_center_m']),out/'disk_support'/f'worker_f{fold}_p{pair}')
                guard=2*SUPPORT_BOUND+2*maximum+design['maximum_test_displacement_m']
                if guard<5:break
                del points,retained;gc.collect()
            else:raise RuntimeError('Unable to satisfy original spatial guard')
            aug_support=support64(points,out/'disk_support'/f'worker_f{fold}_p{pair}')
            aug_tree=cKDTree(aug_support,copy_data=False)
            aug_context=[row[0] for row in mapped_contexts([aug_tree,cKDTree(np.empty((0,3)))])]
            augmented=[trees[0],aug_tree];aug_contexts=[[row[0],a] for row,a in zip(contexts,aug_context)]
            retained_mask=np.zeros(len(data[1]['xyz']),bool);retained_mask[retained]=True
            result={};sampling=[]
            for scan in (0,1):
                chunks={};ids_all=[];xyz_all=[];view=[]
                for augmented_view,count in ((False,CORES//3),(True,CORES-CORES//3)):
                    mask=eligible[scan] if not augmented_view or scan==0 else eligible[scan]&retained_mask
                    ids=balanced_ids(data[scan]['labels'],mask,count,np.random.default_rng(seed+100000*scan+1000*augmented_view))
                    centers=data[scan]['xyz'][ids].astype(np.float64)
                    if augmented_view and scan==1:
                        positions=np.searchsorted(retained,ids);np.testing.assert_array_equal(retained[positions],ids)
                        centers=points[positions]
                    use=augmented if augmented_view else trees;ct=aug_contexts if augmented_view else contexts
                    f=features([use[scan],use[1-scan]],[[r[scan],r[1-scan]] for r in ct],centers)
                    for key,value in f.items():chunks.setdefault(key,[]).append(value)
                    ids_all.append(ids);xyz_all.append(centers);view.extend([augmented_view]*count)
                ids=np.concatenate(ids_all);assert eligible[scan][ids].all()
                for key,value in chunks.items():result[f'{key}{scan}']=np.concatenate(value)
                result[f'indices{scan}']=ids;result[f'labels{scan}']=data[scan]['labels'][ids]
                result[f'xyz{scan}']=np.concatenate(xyz_all);result[f'augmented{scan}']=np.array(view,bool)
                result[f'available{scan}']=np.ones(len(ids),bool)
                sampling.append(dict(scan=scan,cores=len(ids),unique_original_ids=len(np.unique(ids)),changed=int(result[f'labels{scan}'].sum())))
            target=folder/f'pair_{pair}.npz';stream_npz(target,result)
        write_json(folder/f'pair_{pair}.json',dict(passed=True,fold=fold,pair=pair,epoch=epoch,seed=seed,parameters=params,
            attempts=attempt+1,maximum_observed_training_displacement_m=maximum,total_support_guard_m=float(guard),
            sha256=digest(target),sampling=sampling,training_cells_only=True,full_scan_transformed_before_rebuilding=True,
            design_sha256=digest(out/'design.json'),feature_source_sha256=digest(__file__),timing=timer.stats))
        print('V3 fresh features',fold,pair,'epoch',epoch+1,round(timer.stats['seconds'],1),'seconds',flush=True)
        del points,retained,aug_support,aug_tree,aug_context,augmented,aug_contexts,retained_mask,result,chunks,f,use,ct;gc.collect()


def assemble(out=DEFAULT,fold=0):
    root=fold_dir(out,fold)/'epochs'
    for epoch in range(EPOCHS):
        folder=root/f'epoch_{epoch:02d}';dst=folder/'assembled';dst.mkdir(exist_ok=True)
        if (dst/'complete.json').exists():continue
        sources={pair:json.loads((folder/f'pair_{pair}.json').read_text())['sha256'] for pair in PAIRS}
        for pair in PAIRS:assert digest(folder/f'pair_{pair}.npz')==sources[pair]
        for key in ('x','mask','geo','labels','indices','available','pair','scan'):
            mapping=None;offset=0
            for pair in PAIRS:
                with np.load(folder/f'pair_{pair}.npz') as z:
                    for scan in (0,1):
                        value=np.full(CORES,int(pair) if key=='pair' else scan,np.int16) if key in ('pair','scan') else z[f'{key}{scan}']
                        if mapping is None:mapping=np.lib.format.open_memmap(dst/f'{key}.npy',mode='w+',dtype=value.dtype,shape=(12*CORES,*value.shape[1:]))
                        mapping[offset:offset+CORES]=value;offset+=CORES
            mapping.flush();del mapping
        write_json(dst/'complete.json',dict(passed=True,epoch=epoch,rows=12*CORES,source_sha256=sources))
    print('V3 assembled fresh epoch banks',fold,EPOCHS,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','cache','assemble'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);p.add_argument('--pair',default='01');p.add_argument('--epoch-chunk',type=int,default=2);a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='cache':cache_pair(a.out,a.fold,a.pair,a.epoch_chunk)
    else:assemble(a.out,a.fold)
