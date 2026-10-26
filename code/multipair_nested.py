"""Fixed outer geographic tests, label-stratified inner splits from training only.

The first rotating prototype exposed a validation fold with 6 positives/14,400.
This revision keeps every outer test cell, records the failed attempt, and uses
only labels in each outer training pool to choose an adequate inner validation.
"""
import argparse
from itertools import combinations
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from multipair_data import ROOT,DEFAULT as PROTOTYPE,PAIRS,read_cloud,cell_coordinates,prepare
from m3c2_baseline import digest,write_json

DEFAULT=ROOT/'results/multipair_nested_2026-10-25_v1'


def choose_inner_cells(stats,all_cells,test_cells):
    test={tuple(c) for c in test_cells};pool=sorted(tuple(c) for c in all_cells if tuple(c) not in test)
    # This filter is the leakage boundary: test-cell labels never enter totals,
    # class support constraints, ranking or deterministic tie-breaking.
    allowed=stats[[tuple(c) in set(pool) for c in stats[['ix','iy']].to_numpy()]]
    groups=sorted(set(zip(allowed.pair,allowed.scan)))
    counts=np.zeros((len(pool),len(groups),2),np.float64)
    for row in allowed.itertuples():
        counts[pool.index((row.ix,row.iy)),groups.index((row.pair,row.scan))]=[row.unchanged,row.changed]
    total=counts.sum(0);present=total>0;records=[]
    for selected in combinations(range(len(pool)),3):
        validation=counts[list(selected)].sum(0);training=total-validation
        fractions=np.divide(validation,total,out=np.zeros_like(total),where=present)
        imbalance=float(np.mean((fractions[present]-3/len(pool))**2))
        vp=validation[:,1]/validation.sum(1).clip(1);tp=training[:,1]/training.sum(1).clip(1)
        expected_val=1200*vp;expected_train=3000*tp
        supported=int((expected_val>=10).sum())
        feasible=expected_val.sum()>=100 and expected_train.sum()>=300 and supported>=4
        # Prefer proportional class representation in both roles, then geographic
        # lexical order. No prediction, loss, threshold or outer test score enters.
        records.append(dict(validation_cells=[list(pool[i]) for i in selected],
            expected_validation_changed=float(expected_val.sum()),expected_training_changed=float(expected_train.sum()),
            validation_directions_expected_ge10=supported,class_fraction_imbalance=imbalance,feasible=bool(feasible)))
    feasible=[r for r in records if r['feasible']]
    if not feasible:raise ValueError('No adequate inner validation among outer training cells; preserve design and inspect support')
    chosen=min(feasible,key=lambda r:(r['class_fraction_imbalance'],r['validation_cells']))
    val={tuple(c) for c in chosen['validation_cells']}
    return dict(train=[list(c) for c in pool if c not in val],validation=chosen['validation_cells'],test=[list(c) for c in test_cells]),chosen,records


def prepare_nested(out=DEFAULT,prototype=PROTOTYPE):
    out=Path(out);prototype=Path(prototype);out.mkdir(parents=True,exist_ok=True)
    base=json.loads((prototype/'protocol.json').read_text())
    outer=json.loads((prototype/'crossvalidation_design.json').read_text())
    audit=json.loads((prototype/'data_audit.json').read_text())
    stats_path=out/'outer_training_cell_label_inventory.csv'
    if not stats_path.exists():
        rows=[]
        for cloud in audit['clouds']:
            d=read_cloud(cloud['path']);cell,eligible=cell_coordinates(d['xyz'],base)
            for ix,iy in sorted({tuple(c) for fold in outer['folds'] for c in fold['cells']['test']}):
                mask=eligible&(cell[:,0]==ix)&(cell[:,1]==iy)
                rows.append(dict(pair=cloud['pair'],scan=cloud['scan'],ix=ix,iy=iy,
                    unchanged=int((mask&(d['labels']==0)).sum()),changed=int((mask&(d['labels']==1)).sum())))
            del d,cell,eligible;gc.collect()
        pd.DataFrame(rows).to_csv(stats_path,index=False)
    stats=pd.read_csv(stats_path,dtype={'pair':str})
    cells=sorted({tuple(c) for fold in outer['folds'] for c in fold['cells']['test']})
    folds=[]
    for fold in outer['folds']:
        number=fold['fold'];folder=out if number==0 else out/f'spatial_cv/fold_{number}'
        folder.mkdir(parents=True,exist_ok=True)
        selection,chosen,candidates=choose_inner_cells(stats,cells,fold['cells']['test'])
        if not (folder/'protocol.json').exists():
            protocol=dict(base,cells=selection,spatial_fold_id=number,
                parent_outer_protocol_sha256=digest(prototype/'crossvalidation_design.json'),
                split_rule='Outer test cells fixed geometry-only; inner train/validation cells stratified from outer training labels only',
                inner_validation_design=chosen,
                exposure='Earlier prototype fold0 predictions were inspected. Revision triggered by inadequate validation positives, not a change to outer test regions. No cross-scene or untouched public benchmark claim.')
            write_json(folder/'protocol.json',protocol);shutil.copy2(prototype/'data_audit.json',folder/'data_audit.json')
            write_json(folder/'inner_split_candidates.json',dict(candidates=candidates,chosen=chosen,
                outer_test_labels_used=False,inventory_sha256=digest(stats_path)))
        else:assert json.loads((folder/'protocol.json').read_text())['cells']==selection
        prepare(folder)
        inventory=pd.read_csv(folder/'split_inventory.csv')
        validation=inventory[inventory.split=='validation']
        assert validation.changed.sum()>=100,('Insufficient actual validation positives',number,int(validation.changed.sum()))
        assert (validation.changed>=10).sum()>=4
        folds.append(dict(fold=number,cells=selection,actual_validation_changed=int(validation.changed.sum()),
            validation_directions_ge10=int((validation.changed>=10).sum()),outer_test_unchanged=True))
        print('Nested fold',number,'validation positives',int(validation.changed.sum()),flush=True)
    write_json(out/'nested_design.json',dict(folds=folds,prototype=str(prototype),
        outer_design_sha256=digest(prototype/'crossvalidation_design.json'),inner_selection_source_sha256=digest(__file__),
        test_regions_changed=False,selection_uses_outer_training_labels_only=True,
        class_blind_attempt_preserved=str(prototype),temporal_reference='Retain prototype pair_01-only models and prior pair_23 test; do not retune them',
        candidate_architectures=['nearest','voxel','voxel_aug'],extra_threshold_endpoints=[.01,.02,.98,.99],
        endpoint_reason='Validation-only boundary optima in prototype; extended grid declared before nested model fitting',
        stress_cores_per_scan=512,perturbation_seeds=[42,43,44]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();prepare_nested(a.out)
