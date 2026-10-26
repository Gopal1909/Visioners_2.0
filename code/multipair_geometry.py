"""Bounded validation-only geometric calibration across the six shared pairs."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from multipair_data import DEFAULT,PAIRS,load_pair
from multipair_metrics import pair_scores
from m3c2_baseline import Config,MemoryTimer,estimate_normals,project_direction,classify,digest,write_json

NORMALS={'legacy':(.1,.2,.4),'broad':(.2,.4,.8)}
RADII=(.05,.1,.2)
LENGTHS=(.5,1.)
FLOORS=(.02,.05,.1,.15,.2,.3,.5)
C2C_THRESHOLDS=(.02,.03,.05,.075,.1,.15,.2,.3,.5,.75,1.)


def calibrate(out=DEFAULT):
    out=Path(out);folder=out/'geometry_calibration';folder.mkdir(exist_ok=True)
    if (out/'geometry_frozen.json').exists():
        print('Geometric configuration already frozen',flush=True);return
    candidate_file=folder/'candidate_design.json'
    if not candidate_file.exists():
        write_json(candidate_file,dict(normal_sets=NORMALS,projection_radii_m=RADII,half_lengths_m=LENGTHS,
            minimum_effect_floors_m=FLOORS,c2c_thresholds_m=C2C_THRESHOLDS,registration_error_m=0.,
            calibration_cores_per_scan=512,selection='Mean over pairs/directions of binary macro F1; unknowns are errors for their ground-truth class',
            primary_result='Changed F1/IoU and precision/recall reported separately from the selection objective',
            units='metres',no_test_access=True,protocol_hash=digest(out/'protocol.json')))
    for pair in PAIRS:
        dst=folder/f'pair_{pair}';dst.mkdir(exist_ok=True)
        if (dst/'complete.json').exists():continue
        t0=time.perf_counter();data=load_pair(pair)
        with np.load(out/f'splits/pair_{pair}.npz') as z:
            ids=[];cores=[];labels=[]
            for i in (0,1):
                allids=z[f'validation_indices{i}']
                pick=np.sort(np.random.default_rng(713+int(pair)*2+i).choice(len(allids),512,replace=False))
                ids.append(allids[pick]);cores.append(data[i]['xyz'][ids[i]].astype(np.float64));labels.append(data[i]['labels'][ids[i]])
        rows=[];raw_save={f'indices{i}':ids[i] for i in (0,1)}
        raw_save.update({f'labels{i}':labels[i] for i in (0,1)})
        with MemoryTimer() as timer:
            trees=[cKDTree(d['xyz']) for d in data]
            dist=[trees[1-i].query(cores[i],workers=1)[0] for i in (0,1)]
            for i in (0,1):raw_save[f'c2c_distance{i}']=dist[i]
            for threshold in C2C_THRESHOLDS:
                rows.append(dict(pair=pair,method='C2C',candidate=f'c2c_{threshold}',threshold_m=threshold,
                    **pair_scores(labels,[(d>threshold).astype(np.int8) for d in dist])))
            for name,radii in NORMALS.items():
                cfg=Config(normal_radii_m=radii,projection_radii_m=RADII)
                normals=[estimate_normals(trees[i],cores[i],cfg) for i in (0,1)]
                for length in LENGTHS:
                    cfg=Config(normal_radii_m=radii,projection_radii_m=RADII,half_length_m=length)
                    raw=[project_direction(trees[i],trees[1-i],cores[i],normals[i],cfg) for i in (0,1)]
                    for i in (0,1):
                        for k,v in raw[i].items():raw_save[f'{name}_L{length}_scan{i}_{k}']=v
                    for floor in FLOORS:
                        decision=[classify(r,0.,floor) for r in raw]
                        for j,radius in enumerate(RADII):
                            pred=[d['prediction'][:,j] for d in decision]
                            rows.append(dict(pair=pair,method='M3C2-style',candidate=f'{name}_L{length}_R{radius}_F{floor}',
                                normal_set=name,half_length_m=length,projection_radius_m=radius,effect_floor_m=floor,
                                **pair_scores(labels,pred)))
        np.savez_compressed(dst/'measurements.npz',**raw_save)
        pd.DataFrame(rows).to_csv(dst/'candidate_metrics.csv',index=False)
        write_json(dst/'complete.json',dict(passed=True,measurement_sha256=digest(dst/'measurements.npz'),
            seconds=time.perf_counter()-t0,timing=timer.stats,source_sha256=digest(__file__),
            full_support_points=[len(d['xyz']) for d in data],test_accessed=False))
        print('Validation geometry pair',pair,'seconds',round(time.perf_counter()-t0,1),flush=True)
        del data,trees,normals,raw,decision,raw_save;gc.collect()
    frame=pd.concat([pd.read_csv(folder/f'pair_{p}/candidate_metrics.csv',dtype={'pair':str}) for p in PAIRS],ignore_index=True)
    frame.to_csv(folder/'all_candidate_metrics.csv',index=False)
    summary=frame.groupby(['method','candidate'],sort=False).agg(
        objective=('selection_binary_macro_f1','mean'),changed_f1=('directional_changed_f1','mean'),coverage=('coverage','mean')).reset_index()
    summary.to_csv(folder/'candidate_summary.csv',index=False)
    chosen={}
    for method in ('C2C','M3C2-style'):
        ranked=summary[summary.method==method].sort_values(['objective','coverage','candidate'],ascending=[False,False,True])
        winner=ranked.iloc[0];original=frame[(frame.method==method)&(frame.candidate==winner.candidate)].iloc[0]
        if method=='C2C':chosen['c2c_threshold_m']=float(original.threshold_m)
        else:
            chosen['m3c2_config']=asdict(Config(normal_radii_m=NORMALS[original.normal_set],
                projection_radii_m=(float(original.projection_radius_m),),half_length_m=float(original.half_length_m),
                minimum_effect_m=float(original.effect_floor_m),registration_error_m=0.))
        chosen[method+'_validation']=winner.to_dict()
    write_json(out/'geometry_frozen.json',dict(**chosen,candidate_design_sha256=digest(candidate_file),
        protocol_sha256=digest(out/'protocol.json'),source_sha256=digest(__file__),
        note='Validation only, fixed parameters across all test pairs/stresses. Zero registration error is uncalibrated; M3C2-style, not reference software equivalence.',
        test_predictions_scored=False))
    print(json.dumps(chosen,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    calibrate(p.parse_args().out)
