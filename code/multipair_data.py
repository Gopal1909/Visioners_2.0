"""Six-pair data audit and shared, buffered spatial protocol (originals read-only)."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import zipfile
import zlib
import numpy as np
from plyfile import PlyData
from m3c2_baseline import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results/multipair_2026-10-25_v1'
DATA = Path(os.environ.get('CV_PROJECT_DATA_ROOT', str(ROOT/'dataset')))
PAIRS = ('01','02','03','12','13','23')
SPLITS = ('train','validation','test')


def discover_pair(base, pair):
    folder = Path(base)/f'pair_{pair}'
    files = []
    for scan in (0,1):
        found = sorted(folder.rglob(f'pointCloud{scan}.ply'))
        if len(found) != 1:
            raise ValueError(f'{folder}: expected one cloud {scan}, found {len(found)}')
        files.append(found[0])
    return files


def read_cloud(path):
    ply = PlyData.read(str(path), mmap='r')
    if 'params' not in [e.name for e in ply.elements]:
        raise ValueError(f'{path}: expected params element')
    record = ply['params'].data
    if not {'x','y','z','intensity','cd_type'} <= set(record.dtype.names):
        raise ValueError(f'{path}: invalid schema')
    xyz = np.column_stack([record[k] for k in ('x','y','z')]).astype(np.float32)
    raw_labels = record['cd_type']
    if not np.isfinite(xyz).all() or not np.isin(raw_labels,[0,1]).all():
        raise ValueError(f'{path}: invalid coordinates/labels')
    return dict(xyz=xyz, labels=np.asarray(raw_labels,dtype=np.int8), path=Path(path))


def load_pair(pair, base=None):
    base = Path(base) if base else DATA/'MultiChange3D/BikeParkingConstruction'
    return [read_cloud(p) for p in discover_pair(base,pair)]


def cell_coordinates(xyz, protocol):
    local = (np.asarray(xyz)[:,:2].astype(np.float64)-protocol['origin_xy_m'])/protocol['cell_size_m']
    cell = np.floor(local).astype(np.int64)
    inside = (local-cell)*protocol['cell_size_m']
    guard = protocol['core_boundary_guard_m']
    eligible = ((inside >= guard)&(inside <= protocol['cell_size_m']-guard)).all(1)
    return cell, eligible


def split_mask(xyz, protocol, split):
    cell, eligible = cell_coordinates(xyz,protocol)
    selected = np.zeros(len(xyz),bool)
    for x,y in protocol['cells'][split]:
        selected |= (cell[:,0] == x)&(cell[:,1] == y)
    return eligible&selected


def sample_indices(mask, count, seed):
    eligible = np.flatnonzero(mask)
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} eligible points for {count} requested cores')
    return np.sort(np.random.default_rng(seed).choice(eligible,count,replace=False))


def audit(out=DEFAULT, base=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'data_audit.json').exists():
        print('Existing data audit retained',flush=True);return
    base=Path(base) if base else DATA/'MultiChange3D/BikeParkingConstruction'
    rows=[];archives=[]
    for pair in PAIRS:
        files=discover_pair(base,pair)
        archive=base/f'pair_{pair}.zip'
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None,archive
            for path in files:
                member=[i for i in z.infolist() if Path(i.filename).name == path.name]
                assert len(member)==1 and member[0].file_size==path.stat().st_size
                crc=0
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(2**20),b''):crc=zlib.crc32(block,crc)
                assert crc==member[0].CRC,(path,'extraction differs from archive')
        archives.append(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),crc_passed=True,extracted_crc_match=True))
        for scan,path in enumerate(files):
            d=read_cloud(path);xyz=d['xyz'];labels=d['labels']
            rows.append(dict(pair=pair,scan=scan,epoch=int(pair[scan]),path=str(path),
                bytes=path.stat().st_size,sha256=digest(path),points=len(xyz),
                changed=int(labels.sum()),unchanged=int((labels==0).sum()),
                bbox_min=xyz.min(0).tolist(),bbox_max=xyz.max(0).tolist(),
                xyz_order_sha256=hashlib.sha256(xyz.tobytes()).hexdigest()))
            del d,xyz,labels;gc.collect()
        print('Audited pair',pair,flush=True)
    write_json(out/'data_audit.json',dict(passed=True,clouds=rows,archives=archives,
        original_files_immutable=True,labels='0 unchanged; 1 changed; earlier removed/later added',
        epoch_repetition='Six pairs from four shared acquisition epochs, not six independent scenes',
        row_identity='Pair files have different geometry/order hashes and sometimes sizes; preserve per-file indices'))


def prepare(out=DEFAULT,train=3000,validation=1200,test=1500):
    out=Path(out)
    if not (out/'data_audit.json').exists():audit(out)
    audit_data=json.loads((out/'data_audit.json').read_text())
    if not (out/'protocol.json').exists():
        lo=np.min([r['bbox_min'] for r in audit_data['clouds']],0)
        hi=np.max([r['bbox_max'] for r in audit_data['clouds']],0)
        protocol=dict(version=1,scene='BikeParkingConstruction',pairs=list(PAIRS),
            seed=20261025,cell_size_m=8.,core_boundary_guard_m=2.5,
            origin_xy_m=(np.floor(lo[:2]/8)*8).tolist(),bbox_min=lo.tolist(),bbox_max=hi.tolist(),
            counts=dict(train=train,validation=validation,test=test),
            sampling='Uniform original points without replacement within fixed split cells; no class balancing at evaluation',
            split_rule='Geometry-only occupied XY cell inventory; seed permutation, 50%/25%/25% cells; global mapping shared by all pairs/epochs',
            geometry_support_halo_bound_m=float(np.hypot(.2,1.)),learning_support_bound_m=1.2,
            stress_rotation_center_m=((lo+hi)/2).tolist(),
            max_rotation_deg=5.,max_translation_m=.5,
            exposure='Earlier pair_01 experiments were inspected. This new protocol is frozen before new method calibration/test outcomes; no cross-scene claim.',
            test_policy='No test prediction scoring until method configurations are frozen. Retrain from scratch; no old checkpoint initialization.',
            temporal_protocol=dict(training_pair='01',validation='pair_01 spatial validation cells',test_pair='23',
                train_epochs=[0,1],test_epochs=[2,3],other_pairs='Not used for temporal-model selection'))
        radius=float(np.linalg.norm((hi-lo)/2))
        displacement=2*np.sin(np.deg2rad(5)/2)*radius+.5
        protocol['rigid_stress_displacement_bound_m']=displacement
        protocol['minimum_cross_split_core_distance_m']=5.
        protocol['worst_clean_train_vs_stressed_test_support_sum_m']=1.2+np.hypot(.2,1.)+displacement
        assert 5.>protocol['worst_clean_train_vs_stressed_test_support_sum_m']
        occupied={}
        for row in audit_data['clouds']:
            d=read_cloud(row['path']);cell,eligible=cell_coordinates(d['xyz'],protocol)
            u,c=np.unique(cell[eligible],axis=0,return_counts=True)
            for key,n in zip(u,c):occupied[tuple(map(int,key))]=occupied.get(tuple(map(int,key)),0)+int(n)
            del d,cell,eligible;gc.collect()
        cells=sorted(key for key,n in occupied.items() if n>=500)
        order=np.random.default_rng(protocol['seed']).permutation(len(cells))
        nval=max(1,len(cells)//4);ntest=nval;ntrain=len(cells)-nval-ntest
        protocol['cells']={s:[list(cells[i]) for i in ids] for s,ids in zip(SPLITS,
            (order[:ntrain],order[ntrain:ntrain+nval],order[ntrain+nval:]))}
        protocol['occupied_core_cell_counts']={','.join(map(str,k)):v for k,v in occupied.items()}
        write_json(out/'protocol.json',protocol)  # no labels used in assignment
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['counts']==dict(train=train,validation=validation,test=test),'Existing protocol counts are immutable'
    folder=out/'splits';folder.mkdir(exist_ok=True)
    summary=[]
    for pair in PAIRS:
        path=folder/f'pair_{pair}.npz'
        if not path.exists():
            arrays={}
            for scan,d in enumerate(load_pair(pair)):
                for split in SPLITS:
                    seed=20261025+int(pair)*1000+SPLITS.index(split)*100+scan
                    mask=split_mask(d['xyz'],protocol,split)
                    ids=sample_indices(mask,protocol['counts'][split],seed)
                    arrays[f'{split}_indices{scan}']=ids
                    arrays[f'{split}_xyz{scan}']=d['xyz'][ids]
                    arrays[f'{split}_labels{scan}']=d['labels'][ids]
                    arrays[f'{split}_eligible{scan}']=np.array(mask.sum())
                del d;gc.collect()
            np.savez_compressed(path,**arrays)
        with np.load(path) as z:
            for split in SPLITS:
                for scan in (0,1):
                    summary.append(dict(pair=pair,scan=scan,split=split,cores=len(z[f'{split}_indices{scan}']),
                        eligible=int(z[f'{split}_eligible{scan}']),changed=int(z[f'{split}_labels{scan}'].sum())))
        print('Prepared shared splits pair',pair,flush=True)
    import pandas as pd
    pd.DataFrame(summary).to_csv(out/'split_inventory.csv',index=False)
    write_json(out/'preparation_complete.json',dict(passed=True,protocol_sha256=digest(out/'protocol.json'),
        split_sha256={pair:digest(folder/f'pair_{pair}.npz') for pair in PAIRS},
        core_counts=protocol['counts'],no_test_predictions_scored=True))
    design_path=out/'crossvalidation_design.json'
    if not design_path.exists() and 'inner_validation_design' not in protocol:
        cells=sum([protocol['cells'][s] for s in SPLITS],[]);n=len(cells);k=len(protocol['cells']['test'])
        folds=[]
        for fold in range((n+k-1)//k):
            test_ids={(n-k-fold*k+j)%n for j in range(k)}
            val_ids={(n-2*k-fold*k+j)%n for j in range(k)}
            folds.append(dict(fold=fold,cells={s:[c for i,c in enumerate(cells) if i in ids]
                for s,ids in [('train',set(range(n))-test_ids-val_ids),('validation',val_ids),('test',test_ids)]}))
        write_json(design_path,dict(protocol_sha256=digest(out/'protocol.json'),folds=folds,
            reason='Complete geometry-only cell rotation, fixed before test prediction scoring; avoids relying on one region',
            selection='Each fold performs its own training/validation only; no global tuning from out-of-fold test results'))
    temporal_path=folder/'temporal_test_pair23.npz'
    if not temporal_path.exists() and 'inner_validation_design' not in protocol:
        arrays={}
        for scan,d in enumerate(load_pair('23')):
            ids=sample_indices(np.ones(len(d['xyz']),bool),5000,99123+scan)
            arrays[f'indices{scan}']=ids;arrays[f'xyz{scan}']=d['xyz'][ids];arrays[f'labels{scan}']=d['labels'][ids]
        np.savez_compressed(temporal_path,**arrays)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','prepare'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    (audit if args.stage=='audit' else prepare)(args.out)
