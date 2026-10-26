"""Paired geometry features; uniform voxel context is separate from GT/core IDs."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from multipair_data import DEFAULT,PAIRS,load_pair
from m3c2_baseline import MemoryTimer,digest,write_json

RADIUS=.6
VOXEL=.075
K=128


def paired_features(source_tree,target_tree,centers,k=128,radius=.6):
    """NumPy-only copy of the legacy pairing kernel, verified equal by regression.

    Keeping this worker free of PyTorch avoids loading CUDA libraries while
    complete-cloud search trees occupy memory. No labels enter this function.
    """
    patches=[];masks=[];aux=[]
    for tree in (source_tree,target_tree):
        distance,ids=tree.query(centers,k=k,distance_upper_bound=radius,workers=1)
        mask=np.isfinite(distance);safe=np.minimum(ids,max(0,len(tree.data)-1))
        coords=tree.data[safe] if len(tree.data) else np.zeros((*safe.shape,3))
        relative=(coords-centers[:,None,:])/radius;relative[~mask]=0.
        patches.append(relative.astype(np.float32));masks.append(mask)
        aux.append(mask.sum(1).astype(np.float32)/k)
        if len(patches)==2:aux.append(np.minimum(distance[:,0],radius).astype(np.float32)/radius)
    return dict(p0=patches[0],m0=masks[0],p1=patches[1],m1=masks[1],aux=np.column_stack(aux).astype(np.float32))


def voxel_centroids(xyz,size=VOXEL):
    if not len(xyz):return np.empty((0,3),np.float32)
    origin=xyz.min(0).astype(np.float64)
    q=np.floor((xyz.astype(np.float64)-origin)/size).astype(np.int64)
    dims=q.max(0)+1
    code=(q[:,0]*dims[1]+q[:,1])*dims[2]+q[:,2]
    _,inverse=np.unique(code,return_inverse=True)
    count=np.bincount(inverse)
    return np.column_stack([np.bincount(inverse,weights=xyz[:,j])/count for j in range(3)]).astype(np.float32)


def moments(p,mask):
    count=mask.sum(1).clip(1)
    mean=(p*mask[:,:,None]).sum(1)/count[:,None]
    centered=(p-mean[:,None,:])*mask[:,:,None]
    cov=np.einsum('nki,nkj->nij',centered,centered)/count[:,None,None]
    eigen=np.linalg.eigvalsh(cov).clip(0)
    radius=np.linalg.norm(p,axis=2)
    return np.column_stack([mean,np.sqrt(np.diagonal(cov,axis1=1,axis2=2).clip(0)),eigen,
        radius.max(1),(radius*mask).sum(1)/count,mask.sum(1)/K]).astype(np.float32)


def extract_features(trees,context_trees,centers,variant):
    """Features cannot access labels or absolute center coordinates."""
    use=trees if variant=='nearest' else context_trees
    f=paired_features(use[0],use[1],centers,k=K,radius=RADIUS)
    distance,nearest=trees[1].query(centers,workers=1)
    counts=[t.query_ball_point(centers,RADIUS,return_length=True,workers=1) for t in trees]
    f['aux']=np.column_stack([np.log1p(n)/np.log1p(10000) for n in counts]+[np.minimum(distance,RADIUS)/RADIUS]).astype(np.float32)
    a=moments(f['p0'],f['m0']);b=moments(f['p1'],f['m1'])
    delta=((trees[1].data[nearest]-centers).clip(-RADIUS,RADIUS)/RADIUS
        if len(trees[1].data) else np.zeros_like(centers))
    f['geo']=np.column_stack([a,b,b-a,np.abs(b-a),f['aux'],delta]).astype(np.float32)
    return f


def cache_pair(out,pair):
    out=Path(out);folder=out/'features';folder.mkdir(exist_ok=True)
    variants=['nearest','voxel']
    wanted=[v for v in variants if not (folder/f'pair_{pair}_{v}.json').exists()]
    if not wanted:return
    data=load_pair(pair)
    with MemoryTimer() as timer:
        trees=[cKDTree(d['xyz']) for d in data]
        context=[cKDTree(voxel_centroids(d['xyz'])) for d in data]
        with np.load(out/f'splits/pair_{pair}.npz') as z:
            diagnostics=[]
            for variant in wanted:
                arrays={}
                for split in ('train','validation'):
                    for scan in (0,1):
                        ids=z[f'{split}_indices{scan}'];centers=data[scan]['xyz'][ids]
                        f=extract_features([trees[scan],trees[1-scan]],[context[scan],context[1-scan]],centers,variant)
                        for key,value in f.items():arrays[f'{split}_{key}{scan}']=value
                        arrays[f'{split}_labels{scan}']=data[scan]['labels'][ids]
                        arrays[f'{split}_indices{scan}']=ids
                        radii=np.max(np.linalg.norm(f['p0'],axis=2),axis=1)*RADIUS
                        diagnostics.append(dict(variant=variant,split=split,scan=scan,median_outer_radius_m=float(np.median(radii)),
                            q90_outer_radius_m=float(np.quantile(radii,.9)),empty_target_rate=float((~f['m1'].any(1)).mean())))
                path=folder/f'pair_{pair}_{variant}.npz'
                np.savez_compressed(path,**arrays)
                write_json(folder/f'pair_{pair}_{variant}.json',dict(sha256=digest(path),protocol_sha256=digest(out/'protocol.json'),
                    source_sha256=digest(__file__),variant=variant,context_points=[len(t.data) for t in context],
                    full_points=[len(t.data) for t in trees],patch_radius_m=RADIUS,voxel_size_m=VOXEL if variant=='voxel' else None,
                    k=K,original_labels_never_voxelized=True,test_accessed=False,diagnostics=diagnostics))
                del arrays,f;gc.collect()
    write_json(folder/f'pair_{pair}_timing.json',timer.stats)
    print('Training/validation features ready pair',pair,round(timer.stats['seconds'],1),'seconds',flush=True)


def assemble(out,variant,split,pairs=PAIRS):
    # Stream one compressed array at a time into .npy memory maps. Concatenating
    # all pair caches temporarily doubles RAM and can exceed laptop commit limits.
    folder=Path(out)/'features/assembled'/('_'.join(pairs))/variant/split
    folder.mkdir(parents=True,exist_ok=True)
    keys=('p0','m0','p1','m1','aux','geo','labels','pair','scan','indices')
    metadata=folder/'complete.json'
    if not metadata.exists():
        lengths=[]
        for pair in pairs:
            with np.load(Path(out)/f'features/pair_{pair}_{variant}.npz') as z:
                lengths.extend(len(z[f'{split}_labels{i}']) for i in (0,1))
        for key in keys:
            mapping=None;offset=0
            for pair in pairs:
                with np.load(Path(out)/f'features/pair_{pair}_{variant}.npz') as z:
                    for scan in (0,1):
                        n=len(z[f'{split}_labels{scan}'])
                        value=(np.full(n,int(pair)) if key=='pair' else np.full(n,scan) if key=='scan' else z[f'{split}_{key}{scan}'])
                        if mapping is None:
                            mapping=np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',dtype=value.dtype,shape=(sum(lengths),*value.shape[1:]))
                        mapping[offset:offset+n]=value;offset+=n
                        del value
            mapping.flush();del mapping;gc.collect()
        write_json(metadata,dict(rows=sum(lengths),pairs=list(pairs),variant=variant,split=split,
            source_cache_sha256={pair:digest(Path(out)/f'features/pair_{pair}_{variant}.npz') for pair in pairs}))
    receipt=json.loads(metadata.read_text())
    for pair,expected in receipt['source_cache_sha256'].items():
        assert digest(Path(out)/f'features/pair_{pair}_{variant}.npz')==expected,'Feature source cache changed'
    return {key:np.load(folder/f'{key}.npy',mmap_mode='r') for key in keys}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--pairs',nargs='+',default=list(PAIRS));a=p.parse_args()
    for pair in a.pairs:cache_pair(a.out,pair)
