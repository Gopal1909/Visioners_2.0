"""Resumable buffered spatial study using the existing geometric implementations.

prepare -> geometry (calibration, freeze, clean + representative stresses).
Original data and historical outputs are read-only. All indices are original PLY rows.
"""
from dataclasses import asdict
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree
from m3c2_baseline import (Config, MemoryTimer, estimate_normals, project_direction,
                           classify, spatial_core_indices, score_pair, write_json, digest)
from c2c_robustness import (apply_rigid_transformation, apply_point_sparsity,
                            apply_positional_noise, apply_spatial_occlusion)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/study_2026-10-25_v1'
DATA_ROOT=Path(os.environ.get('CV_PROJECT_DATA_ROOT',str(ROOT/'dataset')))
GEOMETRY=Config(projection_radii_m=(.10,))
CONDITIONS=[('clean','clean',0.),('translation_0.10','translation',.10),
            ('translation_0.20','translation',.20),('rotation_0.5','rotation',.5),
            ('rotation_1.0','rotation',1.),('sparsity_0.25','sparsity',.25),
            ('sparsity_0.10','sparsity',.10),('noise_0.020','noise',.02),
            ('noise_0.075','noise',.075),('occlusion_4.6','occlusion',4.6),
            ('occlusion_7.6','occlusion',7.6)]


def load_pair():
    data=[]
    for i in (0,1):
        path=DATA_ROOT/f'MultiChange3D/BikeParkingConstruction/pair_01/pointCloud{i}.ply'
        ply=PlyData.read(str(path),mmap='r')
        records=ply['params'].data  # preserves every original field without rewriting
        xyz=np.column_stack([records[k] for k in ('x','y','z')]).astype('float32')
        labels=np.asarray(records['cd_type'],dtype='int8')
        assert np.isfinite(xyz).all() and np.isin(labels,[0,1]).all()
        data.append(dict(xyz=xyz,cd_type=labels,records=records,path=path))
    return data


def condition_cloud(data,kind,value):
    """Exact Stage 3 helpers/seed/mean center, with explicit original index mapping."""
    xyz=data[1]['xyz']; ids=np.arange(len(xyz),dtype=np.int64)
    center=xyz.mean(axis=0)
    if kind=='clean': return xyz,ids
    if kind=='translation': return apply_rigid_transformation(xyz,value,0,center,42),ids
    if kind=='rotation': return apply_rigid_transformation(xyz,0,value,center,42),ids
    if kind=='noise': return apply_positional_noise(xyz,value,42),ids
    if kind=='sparsity': return apply_point_sparsity(xyz,ids,value,42)
    if kind=='occlusion':
        points,retained,_,_=apply_spatial_occlusion(xyz,ids,value,center)
        return points,retained
    raise ValueError(kind)


def prepare(out,train=8000,calibration=1000,test=4000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'splits.npz').exists():
        print('Existing split retained',flush=True);return
    data=load_pair()
    lo=min(float(d['xyz'][:,0].min()) for d in data)
    hi=max(float(d['xyz'][:,0].max()) for d in data)
    span=hi-lo
    bounds={'train':[lo,lo+.45*span], 'calibration':[lo+.53*span,lo+.63*span],
            'test':[lo+.71*span,hi]}
    min_gap=.08*span; halo=np.hypot(.10,1.)
    assert min_gap>2*halo, 'Geometry alone cannot guarantee disjoint support halos'
    arrays={};meta=[]
    for split,n in [('train',train),('calibration',calibration),('test',test)]:
        for i in (0,1):
            a,b=bounds[split]
            eligible=np.flatnonzero((data[i]['xyz'][:,0]>=a)&(data[i]['xyz'][:,0]<=b))
            seed=20261025+100*list(bounds).index(split)+i
            sub,sm=spatial_core_indices(data[i]['xyz'][eligible],min(n,len(eligible)),seed,2.)
            ids=eligible[sub]
            arrays[f'{split}_indices{i}']=ids
            arrays[f'{split}_xyz{i}']=data[i]['xyz'][ids]
            arrays[f'{split}_gt{i}']=data[i]['cd_type'][ids]
            meta.append(dict(split=split,scan=i,cores=len(ids),eligible=len(eligible),
                             changed=int(arrays[f'{split}_gt{i}'].sum()),**sm))
    np.savez_compressed(out/'splits.npz',**arrays)
    write_json(out/'split_config.json',dict(axis='X',bounds_m=bounds,gap_m=min_gap,
        maximum_geometric_support_halo_m=halo,learned_patch_radius_m=.6,
        support_halos_disjoint=True,seeds='20261025 + 100*split_order + scan',
        epochs={'train':[0,1],'calibration':[0,1],'test':[0,1]},scene='BikeParkingConstruction',pair='01',
        selection='geometry-only fixed X fractions .45/.53/.63/.71; no label-based spatial selection',
        caveat='Same scene and epochs; model spatial holdout only. Historic C2C threshold and pilot used this pair, so not an unbiased benchmark test.',samples=meta))
    manifest={str(d['path']):dict(bytes=d['path'].stat().st_size,sha256=digest(d['path']),
        fields=list(d['records'].dtype.names),points=len(d['xyz'])) for d in data}
    write_json(out/'dataset_manifest.json',manifest)
    print(json.dumps(meta,indent=2),flush=True)


def evaluate_geometry(points,trees,cores,cfg,normal0=None):
    raw=[];normal=[];dec=[];timeinfo={}
    for i in (0,1):
        with MemoryTimer() as timer:
            n=normal0 if i==0 and normal0 is not None else estimate_normals(trees[i],cores[i],cfg)
        timeinfo[f'normals{i}']=timer.stats
        with MemoryTimer() as timer:
            r=project_direction(trees[i],trees[1-i],cores[i],n,cfg)
            d=classify(r,cfg.registration_error_m,cfg.minimum_effect_m)
        timeinfo[f'cylinders{i}']=timer.stats
        raw.append(r);normal.append(n);dec.append(d)
    with MemoryTimer() as timer:
        distances=[trees[1-i].query(cores[i],k=1,workers=1)[0] for i in (0,1)]
    timeinfo['c2c_queries']=timer.stats
    return raw,normal,dec,distances,timeinfo


def geometry(out,only=None):
    out=Path(out)
    if not (out/'splits.npz').exists(): prepare(out)
    splits=dict(np.load(out/'splits.npz'))
    data=load_pair();cfg=GEOMETRY
    with MemoryTimer() as timer: tree0=cKDTree(data[0]['xyz'])
    tree0time=timer.stats
    # Calibration diagnostics use only the predefined calibration region. The
    # configuration is retained, not optimized on the already inspected pilot scores.
    if not (out/'geometry_frozen.json').exists():
        trees=[tree0,cKDTree(data[1]['xyz'])]
        cores=[splits[f'calibration_xyz{i}'].astype('float64') for i in (0,1)]
        smoke=evaluate_geometry(None,trees,[c[:32] for c in cores],cfg)
        write_json(out/'geometry_smoke.json',dict(cores_per_scan=32,
            valid_counts=[int(d['valid'].sum()) for d in smoke[2]],passed=True))
        raw,norm,dec,dist,times=evaluate_geometry(None,trees,cores,cfg)
        calibration={}
        for i in (0,1):
            for key in raw[i]: calibration[f'scan{i}_{key}']=raw[i][key]
            calibration[f'scan{i}_prediction']=dec[i]['prediction'][:,0]
        np.savez_compressed(out/'calibration_geometry.npz',**calibration)
        write_json(out/'geometry_frozen.json',dict(config=asdict(cfg),c2c_threshold_m=.075,
            calibration_valid_counts=[int(d['valid'].sum()) for d in dec],
            calibration_cores=[len(c) for c in cores],calibration_runtime=times,
            decision='Retain predeclared pilot geometry/effect/registration parameters after independent geometry audit and buffered calibration support inspection; no metric maximization.',
            scope='Frozen before new spatial-test or stress scoring. Earlier full-pair exploratory pilot is disclosed.',
            split_hash=digest(out/'splits.npz'),implementation_hash=digest(ROOT/'code/m3c2_baseline.py')))
        del trees,smoke,raw,norm,dec,dist;gc.collect()
    normal0=None
    for name,kind,value in CONDITIONS:
        if only and name not in only:continue
        folder=out/'geometry'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            print(f'{name}: checkpoint retained',flush=True);continue
        print(f'RUN {name}',flush=True)
        try:
            xyz1,ids1=condition_cloud(data,kind,value)
            chosen0=splits['test_indices0'];wanted=splits['test_indices1']
            # ids1 sorted by Stage 3 helpers: preserving the exact original row map.
            positions=np.searchsorted(ids1,wanted)
            keep=positions<len(ids1)
            keep[keep] &= ids1[positions[keep]]==wanted[keep]
            chosen1=wanted[keep];positions=positions[keep]
            cores=[data[0]['xyz'][chosen0].astype('float64'),xyz1[positions].astype('float64')]
            gt=[data[0]['cd_type'][chosen0],data[1]['cd_type'][chosen1]]
            with MemoryTimer() as timer: trees=[tree0,cKDTree(xyz1)]
            trees1time=timer.stats
            used_normal0_cache=normal0 is not None
            result=evaluate_geometry(None,trees,cores,cfg,normal0)
            raw,norm,dec,dist,times=result
            if normal0 is None:normal0=norm[0]
            times['tree0_shared']=tree0time;times['tree1']=trees1time
            rows=[]
            masks=[d['valid'][:,0] for d in dec]
            pred=[d['prediction'][:,0] for d in dec]
            c2c=[(d>.075).astype('int8') for d in dist]
            save={}
            for i in (0,1):
                save[f'indices{i}']=(chosen0,chosen1)[i]
                save[f'gt{i}']=gt[i];save[f'xyz{i}']=cores[i]
                save[f'c2c_distance{i}_m']=dist[i];save[f'c2c_prediction{i}']=c2c[i]
                for k,v in raw[i].items():save[f'raw{i}_{k}']=v
                for k,v in norm[i].items():save[f'normal{i}_{k}']=v
                for k,v in dec[i].items():save[f'decision{i}_{k}']=v[:,0]
            np.savez_compressed(folder/'measurements.npz',**save)
            for protocol in ('A','B') if kind=='occlusion' else ('standard',):
                eligible=[np.ones(len(g),bool) for g in gt]
                if protocol=='B':
                    center=data[1]['xyz'].mean(axis=0)
                    eligible[0]=np.linalg.norm(cores[0][:,:2]-center[:2],axis=1)>value
                gt_eval=[g[e] for g,e in zip(gt,eligible)]
                for method,preds,common in [('M3C2-style',pred,True),('C2C common',c2c,True),('C2C all',c2c,False)]:
                    pp=[q[e] for q,e in zip(preds,eligible)]
                    mm=[m[e] for m,e in zip(masks,eligible)] if common else [np.ones(len(g),bool) for g in gt_eval]
                    row=dict(condition=name,family=kind,severity=value,protocol=protocol,method=method,
                        scope=json.loads((out/'geometry_frozen.json').read_text()).get('evaluation_scope',
                            'buffered same-scene spatial-test cores; historically selected C2C threshold'),
                        **score_pair(gt_eval,pp,mm))
                    row['protocol_excluded_scan0']=int((~eligible[0]).sum())
                    row['requested_scan1_cores']=len(wanted)
                    row['retained_scan1_cores']=len(chosen1)
                    row['geometry_seconds']=(sum(v['seconds'] for k,v in times.items() if k!='c2c_queries')
                        if method=='M3C2-style' else times['c2c_queries']['seconds']+tree0time['seconds']+trees1time['seconds'])
                    row['sampled_process_peak_rss_mib']=max(v['sampled_peak_rss_mib'] for v in times.values())
                    rows.append(row)
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            write_json(folder/'timing.json',times)
            write_json(folder/'complete.json',dict(condition=name,seed=42,
                full_support_points0=len(data[0]['xyz']),full_support_points1=len(xyz1),
                normal0_cached=used_normal0_cache,normal1_reestimated=True,
                configuration_hash=digest(out/'geometry_frozen.json'),
                measurement_hash=digest(folder/'measurements.npz')))
            print(pd.DataFrame(rows)[['method','protocol','combined_macro_f1','coverage','unknown_rate']].to_string(index=False),flush=True)
            del xyz1,ids1,trees,raw,norm,dec,dist,result;gc.collect()
        except Exception:
            (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            raise
    collect_geometry(out)


def collect_geometry(out):
    out=Path(out)
    frames=[pd.read_csv(f) for f in (out/'geometry').glob('*/metrics.csv') if (f.parent/'complete.json').exists()]
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'geometry_metrics.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','geometry','collect'])
    p.add_argument('--out',default=str(DEFAULT));p.add_argument('--only',nargs='+')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='geometry':geometry(a.out,a.only)
    else:collect_geometry(a.out)
