"""File-backed raw support and bounded temporaries; unchanged metric geometry."""
import gc
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from learning_v3_kernels import VOXELS,discover_pair,apply_rigid_transformation
from learning_v3_runtime import ROOT,digest,write_json


def load_mapped_pair(out,pair):
    folder=Path(out)/'disk_support'/f'pair_{pair}';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'dataset/MultiChange3D/BikeParkingConstruction'
    result=[]
    for scan,path in enumerate(discover_pair(source,pair)):
        receipt=folder/f'cloud{scan}.json'
        if not receipt.exists():
            record=PlyData.read(path,mmap='r')['params'].data;n=len(record)
            mappings={k:np.lib.format.open_memmap(folder/f'cloud{scan}_{k}.npy',mode='w+',dtype=dtype,shape=shape) for k,dtype,shape in
                [('xyz',np.float32,(n,3)),('support',np.float64,(n,3)),('labels',np.int8,(n,))]}
            for start in range(0,n,100000):
                end=min(n,start+100000);xyz=np.column_stack([record[k][start:end] for k in ('x','y','z')]).astype(np.float32)
                labels=record['cd_type'][start:end];assert np.isin(labels,[0,1]).all() and np.isfinite(xyz).all()
                mappings['xyz'][start:end]=xyz;mappings['support'][start:end]=xyz;mappings['labels'][start:end]=labels
            for value in mappings.values():value.flush()
            del mappings,value;gc.collect()
            write_json(receipt,dict(source=str(path),source_sha256=digest(path),rows=n,
                arrays={k:digest(folder/f'cloud{scan}_{k}.npy') for k in ('xyz','support','labels')}))
        meta=json.loads(receipt.read_text());assert digest(path)==meta['source_sha256']
        original=next(r for r in json.loads((Path(out)/'data_audit.json').read_text())['clouds'] if r['pair']==pair and r['scan']==scan)
        assert meta['source_sha256']==original['sha256']
        for key,sha in meta['arrays'].items():assert digest(folder/f'cloud{scan}_{key}.npy')==sha
        result.append({**{k:np.load(folder/f'cloud{scan}_{k}.npy',mmap_mode='r') for k in ('xyz','support','labels')},'path':path})
    return result


def bounded_voxels(xyz,size,max_points=100000):
    """Every voxel belongs to one partition; sums retain original point order.

    Sorting decoded integer voxel keys recovers exactly the legacy lexicographic
    output order. No voxel is averaged twice and no downsampling scale changes.
    """
    if not len(xyz):return np.empty((0,3),np.float32)
    keys=[];centers=[]
    def leaf(points):
        q=np.floor(np.asarray(points,dtype=np.float64)/size).astype(np.int64);minimum=q.min(0);q-=minimum;dims=q.max(0)+1
        code=(q[:,0]*dims[1]+q[:,1])*dims[2]+q[:,2]
        unique,inverse=np.unique(code,return_inverse=True);count=np.bincount(inverse)
        centers.append(np.column_stack([np.bincount(inverse,weights=points[:,j])/count for j in range(3)]).astype(np.float32))
        keys.append(np.column_stack((unique//(dims[1]*dims[2]),unique//dims[2]%dims[1],unique%dims[2]))+minimum)
    def partition(points,axis=0,width=32):
        if len(points)<=max_points:leaf(points);return
        bins=np.empty(len(points),np.int32)
        for start in range(0,len(points),100000):bins[start:start+100000]=(np.floor(points[start:start+100000,axis]/size).astype(np.int64)//width).astype(np.int32)
        values=np.unique(bins)
        if len(values)==1:
            if axis<2:partition(points,axis+1,width)
            elif width>1:partition(points,0,max(1,width//2))
            else:
                # A single massively repeated voxel: sequential accumulation
                # matches bincount addition without N-sized inverse buffers.
                total=np.zeros(3,np.float64)
                for start in range(0,len(points),100000):total=np.cumsum(np.vstack((total,points[start:start+100000])),axis=0)[-1]
                centers.append((total/len(points)).astype(np.float32)[None]);keys.append(np.floor(points[:1]/size).astype(np.int64))
            return
        for value in values:
            subset=points[bins==value]
            partition(subset,(axis+1)%3,width if axis<2 else max(1,width//2))
    partition(xyz)
    q=np.concatenate(keys);p=np.concatenate(centers);order=np.lexsort((q[:,2],q[:,1],q[:,0]))
    return p[order]


def contexts(trees):
    return [[cKDTree(bounded_voxels(tree.data,size)) for tree in trees] for size in VOXELS]


def support64(points,folder):
    if points.dtype==np.float64:return points
    path=Path(folder)/'points_support64.npy'
    support=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=points.shape)
    for start in range(0,len(points),100000):support[start:start+100000]=points[start:start+100000]
    support.flush();return support


def mapped_transform(xyz,parameters,seed,center,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);n=len(xyz)
    if parameters['retention']<1:
        ids=np.random.default_rng(seed+10000).choice(n,int(np.round(n*parameters['retention'])),replace=False);ids.sort()
    else:ids=np.arange(n,dtype=np.int64)
    np.save(folder/'retained.npy',ids);del ids
    ids=np.load(folder/'retained.npy',mmap_mode='r')
    dtype=np.float64 if parameters['rotation_deg']!=0 else np.float32
    points=np.lib.format.open_memmap(folder/'points.npy',mode='w+',dtype=dtype,shape=(len(ids),3))
    rng=np.random.default_rng(seed+20000);maximum=0.
    for begin in range(0,len(ids),100000):
        end=min(len(ids),begin+100000);original=np.asarray(xyz[ids[begin:end]])
        transformed=apply_rigid_transformation(original,parameters['translation_m'],parameters['rotation_deg'],center,seed)
        if parameters['noise_sigma_m']:transformed+=rng.normal(size=(len(original),3)).clip(-3,3).astype(transformed.dtype)*parameters['noise_sigma_m']
        points[begin:end]=transformed
        maximum=max(maximum,float(np.linalg.norm(transformed.astype(np.float64)-original,axis=1).max(initial=0.)))
    points.flush();return points,ids,maximum
