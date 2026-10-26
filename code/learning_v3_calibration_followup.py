"""Validation-only checkpoint and robust-threshold follow-up; prior fits immutable."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from learning_v3_runtime import ROOT,BASE,CANDIDATES,THRESHOLDS,validation,digest,write_json,write_csv
from multipair_metrics import binary_scores

V3=ROOT/'results/learning_v3_construction_2026-10-25_v1'
DEFAULT=ROOT/'results/operating_point_followup_2026-10-26_v1'
V2_VARIANTS=('multiscale_clean','multiscale_robust','multiscale_robust_hard')
VIEWS=('clean','mild','medium')


def score(data,threshold):
    keep=data['available'];code=data['pair']*2+data['scan'];rows=[]
    for group in np.unique(code):
        selected=keep&(code==group)
        rows.append(binary_scores(data['labels'][selected],(data['probability'][selected]>=threshold).astype(np.int8)))
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows);tn=sum(r['tn'] for r in rows)
    return dict(binary_macro_f1=float(np.mean([r['binary_macro_f1'] for r in rows])),
        changed_f1=float(np.mean([r['f1_changed'] for r in rows])),false_positive_rate=fp/max(1,fp+tn),
        changed_precision=tp/max(1,tp+fp),changed_recall=tp/max(1,tp+fn),available_cores=int(keep.sum()))


def select_point(rows,clean_floor,fpr_ceiling):
    eligible=[r for r in rows if r['clean_macro']>=clean_floor-1e-12 and r['clean_fpr']<=fpr_ceiling+1e-12]
    if not eligible:return None
    return max(eligible,key=lambda r:(r['mean_macro'],r['mean_changed_f1'],r['clean_macro'],r['threshold'],-r['candidate_order'],-r['checkpoint_order']))


def sources():
    for family,base,folds,names in [('v2',BASE,range(5),V2_VARIANTS),('v3',V3,range(1),CANDIDATES)]:
        for fold in folds:
            for name in names:yield family,fold,name,base/f'fold_{fold}/learning/{name}_seed42'


def identifier(family,fold,name):return f'{family}_fold{fold}_{name}'


def prepare(out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'design.json').exists():return
    locked={identifier(f,fld,n):dict(family=f,fold=fld,candidate=n,folder=str(p),
        best_sha256=digest(p/'best.pt'),last_sha256=digest(p/'last.pt'),frozen_meta_sha256=digest(p/'frozen.json')) for f,fld,n,p in sources()}
    write_json(out/'design.json',dict(purpose='Test checkpoint/operating-point alignment after the completed v3 validation pilot failed its gate',
        source_v3_pilot=str(V3),source_v2_study=str(BASE),source_models=locked,
        candidates='V2: three existing seed42 fits in five folds; v3: three completed fold0 fits. Each best-BCE and final checkpoint evaluated.',
        threshold_grid=THRESHOLDS.tolist(),single_threshold_for_all_conditions=True,
        calibration='Maximize mean clean/mild/medium validation binary macro F1',
        clean_constraint='Each fold: macro F1 >= original v2 robust-selected clean minus .005; unchanged FPR <= that reference plus .005',
        tie_break='Mean changed F1, clean macro, higher threshold, original candidate order, best before final checkpoint',
        v2_promotion_gate='Mean across five validation folds improves robust macro by >=.01 and does not reduce mean robust changed F1',
        v3_promotion_gate='Fold0 new candidate exceeds recalibrated v2 mean macro by >=.01, with the same clean constraints',
        exposure='Previous same-site tests informed this follow-up; all choices here use validation only. No new test scoring before selections frozen.',
        data_policy='No downloads; no raw-cloud feature recomputation; no Office/Open_space/ConSLAM data used',
        old_frozen_thresholds_and_results_immutable=True))
    dst=out/'sources';dst.mkdir(exist_ok=True)
    for p in [Path(__file__),ROOT/'code/learning_v3_model.py',ROOT/'code/learning_v3_metrics.py']:
        shutil.copy2(p,dst/p.name)


def final_probabilities(out):
    import torch
    from learning_v3_model import make_model,setup_device
    from learning_v2_model import MetricFusion,probabilities
    out=Path(out);plan=json.loads((out/'design.json').read_text());setup_device(42,'cpu')
    for family,fold,name,src in sources():
        key=identifier(family,fold,name);dst=out/'final_checkpoint_validation'/key;dst.mkdir(parents=True,exist_ok=True)
        lock=plan['source_models'][key];assert digest(src/'last.pt')==lock['last_sha256']
        if (dst/'complete.json').exists():
            receipt=json.loads((dst/'complete.json').read_text())
            for view,sha in receipt['probability_sha256'].items():assert digest(dst/f'{view}.npz')==sha
            continue
        model=MetricFusion() if family=='v2' else make_model(name)
        state=torch.load(src/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.eval();epoch=int(state['epoch'])+1;del state
        hashes={};start=time.perf_counter()
        for view in VIEWS:
            data=validation(fold,view);p=probabilities(model,data,device='cpu',batch_size=32)
            np.savez_compressed(dst/f'{view}.npz',probability=p,**{k:data[k] for k in ('labels','pair','scan','indices','available')})
            hashes[view]=digest(dst/f'{view}.npz');del data,p
        write_json(dst/'complete.json',dict(passed=True,checkpoint_sha256=lock['last_sha256'],epoch=epoch,probability_sha256=hashes,
            prediction_device='cpu',validation_only=True,seconds=time.perf_counter()-start))
        del model;gc.collect();print('Calibration diagnostic final checkpoint',key,epoch,flush=True)


def analyse(out):
    out=Path(out);plan=json.loads((out/'design.json').read_text());rows=[];baseline={};candidates={}
    for fold in range(5):
        choice=json.loads((BASE/f'fold_{fold}/learning/selection.json').read_text());name=choice['secondary_robust_variant']
        src=BASE/f'fold_{fold}/learning/{name}_seed42';meta=json.loads((src/'frozen.json').read_text())
        values={}
        for view in VIEWS:
            with np.load(src/f'validation_{view}.npz') as z:values[view]=score(z,meta['threshold'])
            assert np.isclose(values[view]['binary_macro_f1'],meta['validation_by_view'][view]['binary_macro_f1'],atol=1e-12,rtol=0)
        baseline[fold]=dict(candidate=name,threshold=meta['threshold'],clean_macro=values['clean']['binary_macro_f1'],clean_fpr=values['clean']['false_positive_rate'],
            mean_macro=float(np.mean([x['binary_macro_f1'] for x in values.values()])),mean_changed_f1=float(np.mean([x['changed_f1'] for x in values.values()])))
    for family,fold,name,src in sources():
        key=identifier(family,fold,name);lock=plan['source_models'][key];meta=json.loads((src/'frozen.json').read_text())
        assert digest(src/'best.pt')==lock['best_sha256'] and digest(src/'frozen.json')==lock['frozen_meta_sha256']
        for checkpoint in ('best','final'):
            paths={v:(src/f'validation_{v}.npz' if checkpoint=='best' else out/'final_checkpoint_validation'/key/f'{v}.npz') for v in VIEWS}
            data={}
            for view,path in paths.items():
                with np.load(path) as z:data[view]={k:z[k] for k in z.files}
                reference=validation(fold,view)
                for k in ('labels','pair','scan','indices','available'):np.testing.assert_array_equal(data[view][k],reference[k])
                del reference
            for threshold in THRESHOLDS:
                values={v:score(z,float(threshold)) for v,z in data.items()}
                r=dict(family=family,fold=fold,candidate=name,checkpoint=checkpoint,threshold=float(threshold),
                    candidate_order=(V2_VARIANTS if family=='v2' else CANDIDATES).index(name),checkpoint_order=0 if checkpoint=='best' else 1,
                    clean_macro=values['clean']['binary_macro_f1'],clean_fpr=values['clean']['false_positive_rate'],
                    mean_macro=float(np.mean([x['binary_macro_f1'] for x in values.values()])),mean_changed_f1=float(np.mean([x['changed_f1'] for x in values.values()])),
                    checkpoint_path=str(src/('best.pt' if checkpoint=='best' else 'last.pt')),checkpoint_sha256=lock['best_sha256' if checkpoint=='best' else 'last_sha256'])
                for view,metrics in values.items():
                    for k,v in metrics.items():r[f'{view}_{k}']=v
                r['eligible']=r['clean_macro']>=baseline[fold]['clean_macro']-.005-1e-12 and r['clean_fpr']<=baseline[fold]['clean_fpr']+.005+1e-12
                rows.append(r)
                if checkpoint=='best' and threshold==meta['threshold']:
                    assert np.isclose(r['clean_macro'],meta['validation_objective'],atol=1e-12,rtol=0)
            del data
    write_csv(out/'validation_threshold_grid.csv',rows)
    for fold in range(5):
        candidates[fold]=select_point([r for r in rows if r['family']=='v2' and r['fold']==fold],baseline[fold]['clean_macro']-.005,baseline[fold]['clean_fpr']+.005)
        assert candidates[fold] is not None
    new=select_point([r for r in rows if r['family']=='v3'],baseline[0]['clean_macro']-.005,baseline[0]['clean_fpr']+.005)
    gain=float(np.mean([candidates[f]['mean_macro']-baseline[f]['mean_macro'] for f in range(5)]))
    changed_gain=float(np.mean([candidates[f]['mean_changed_f1']-baseline[f]['mean_changed_f1'] for f in range(5)]))
    v2_gate=gain>=.01 and changed_gain>=0
    v3_gate=new is not None and new['mean_macro']>=candidates[0]['mean_macro']+.01
    result=dict(passed=True,validation_only=True,v2_promotion_gate_passed=bool(v2_gate),v3_promotion_gate_passed=bool(v3_gate),
        v2_mean_robust_macro_gain=gain,v2_mean_robust_changed_f1_gain=changed_gain,
        original_v2_robust_policy=baseline,v2_selected=candidates,v3_fold0_selected=new,
        design_sha256=digest(out/'design.json'),threshold_grid_sha256=digest(out/'validation_threshold_grid.csv'),
        no_new_test_predictions_or_scores=True,original_models_and_thresholds_unchanged=True)
    write_json(out/'frozen_operating_points.json',result)
    for fold,c in candidates.items():print('V2 calibrated',fold,c['candidate'],c['checkpoint'],c['threshold'],'clean',round(c['clean_macro'],4),'robust',round(c['mean_macro'],4),flush=True)
    if new:print('V3 calibrated',new['candidate'],new['checkpoint'],new['threshold'],'clean',round(new['clean_macro'],4),'robust',round(new['mean_macro'],4),flush=True)
    print('Calibration promotion gates',v2_gate,v3_gate,'mean v2 gain',gain,flush=True)
    write_json(out/'verification.json',dict(passed=True,validation_model_checkpoint_pairs=36,threshold_rows=len(rows),
        original_validation_core_ids_and_labels_verified=True,original_frozen_scores_reproduced=True,
        all_source_checkpoints_unchanged=True,no_test_labels_used_for_selection=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','predict','analyse','run']);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    if a.stage in ('prepare','run'):prepare(a.out)
    if a.stage in ('predict','run'):final_probabilities(a.out)
    if a.stage in ('analyse','run'):analyse(a.out)
