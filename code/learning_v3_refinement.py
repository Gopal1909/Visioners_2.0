"""Bounded construction curriculum refinement; reuse all completed computations."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from learning_v3_runtime import ROOT,BASE,validation,digest,write_json,write_csv,THRESHOLDS
from learning_v3_model import make_model,setup_device
from learning_v3_optim import FunctionalAdamW
from learning_v2_model import batch,probabilities
from learning_v2_runtime import arrays,memory_snapshot
from multipair_metrics import balanced_training_weights
from learning_v3_calibration_followup import score,select_point

V3=ROOT/'results/learning_v3_construction_2026-10-25_v1'
CAL=ROOT/'results/operating_point_followup_2026-10-26_v1'
DEFAULT=ROOT/'results/construction_curriculum_refinement_2026-10-26_v1'
VIEWS=('clean','mild','medium')


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists():return json.loads((out/'design.json').read_text())
    cal=json.loads((CAL/'frozen_operating_points.json').read_text());source=cal['v3_fold0_selected'];reference=cal['v2_selected']['0']
    assert source['candidate']=='graph_fresh_focal' and source['checkpoint']=='best'
    plan=dict(fold=0,epochs=4,seed=900042,source_candidate=source['candidate'],source_checkpoint=source['checkpoint_path'],source_sha256=source['checkpoint_sha256'],
        hypothesis='Hierarchy/focal beat fresh-pool control but not the strongest old setup. Test gentle fine-tuning on the successful original uniform-core weighted-BCE regime.',
        training='Existing v2 fold0 36000 uniform original cores; equal-pair/present-class weighted BCE; clean/mild/medium views sampled equally with deleted-query clean fallback',
        effective_batch=128,microbatch=16,device='cpu',learning_rates=[.0001,.000075,.00005,.000025],weight_decay=.001,dropout=.1,gradient_clip=5.,
        optimizer='Installed native functional AdamW; reset moments at start of refinement',
        validation='Existing original v2 validation cores; evaluate epochs0..4 and the original fixed threshold grid. One shared threshold across all conditions.',
        clean_floor=cal['original_v2_robust_policy']['0']['clean_macro']-.005,
        fpr_ceiling=cal['original_v2_robust_policy']['0']['clean_fpr']+.005,
        reference=reference,gate='Mean three-view validation macro >= recalibrated v2 fold0 plus .01, and robust changed F1 >= reference; clean and FPR constraints enforced.',
        maximum_new_training_epochs=4,no_hyperparameter_expansion_after_results=True,
        test_policy='No new held-out scoring until this gate passes and checkpoint/threshold are frozen',
        scope='Construction-only same-site development follow-up, not an independent-site benchmark',original_results_and_checkpoints_immutable=True)
    assert digest(plan['source_checkpoint'])==plan['source_sha256'];write_json(out/'design.json',plan)
    src=out/'sources';src.mkdir(exist_ok=True);manifest={}
    for name in [Path(__file__).name,'learning_v3_model.py','learning_v3_optim.py','learning_v3_runtime.py','learning_v3_control.py','learning_v3_calibration_followup.py','learning_v2_model.py','learning_v2_runtime.py','multipair_metrics.py']:
        shutil.copy2(ROOT/'code'/name,src/name);manifest[name]=digest(src/name)
    write_json(src/'manifest.json',manifest)
    return plan


def operating_rows(out,epoch,model,plan):
    folder=Path(out)/f'epoch_{epoch:02d}';folder.mkdir(exist_ok=True)
    views={}
    for view in VIEWS:
        path=folder/f'validation_{view}.npz'
        if not path.exists():
            data=validation(0,view);p=probabilities(model,data,device='cpu',batch_size=32)
            np.savez_compressed(path,probability=p,**{k:data[k] for k in ('labels','pair','scan','indices','available')});del data,p
        with np.load(path) as z:views[view]={k:z[k] for k in z.files}
    rows=[]
    for threshold in THRESHOLDS:
        values={v:score(z,float(threshold)) for v,z in views.items()}
        r=dict(epoch=epoch,threshold=float(threshold),candidate_order=0,checkpoint_order=epoch,
            mean_macro=float(np.mean([x['binary_macro_f1'] for x in values.values()])),mean_changed_f1=float(np.mean([x['changed_f1'] for x in values.values()])),
            clean_macro=values['clean']['binary_macro_f1'],clean_fpr=values['clean']['false_positive_rate'])
        for view,metrics in values.items():
            for key,value in metrics.items():r[f'{view}_{key}']=value
        rows.append(r)
    write_csv(folder/'thresholds.csv',rows)
    best=select_point(rows,plan['clean_floor'],plan['fpr_ceiling'])
    write_json(folder/'validation_complete.json',dict(passed=True,epoch=epoch,selected=best,probability_sha256={v:digest(folder/f'validation_{v}.npz') for v in VIEWS}))
    print('Refinement validation',epoch,None if best is None else {k:best[k] for k in ('threshold','clean_macro','mean_macro','mean_changed_f1')},flush=True)
    return rows


def run(out=DEFAULT):
    out=Path(out);plan=prepare(out)
    if (out/'frozen.json').exists():return
    for name,sha in json.loads((out/'sources/manifest.json').read_text()).items():assert digest(ROOT/'code'/name)==sha,'Refinement code changed after declaration'
    setup_device(plan['seed'],'cpu');model=make_model(plan['source_candidate'])
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.p=plan['dropout']
    initial=torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(initial['model']);del initial
    optimizer=FunctionalAdamW(model.parameters(),lr=.0001,weight_decay=.001)
    banks=[arrays(BASE,0,'train',view) for view in VIEWS]
    for b in banks[1:]:
        for key in ('labels','indices','pair','scan'):np.testing.assert_array_equal(b[key],banks[0][key])
    weights=balanced_training_weights(banks[0]['labels'],banks[0]['pair']);rows=[]
    # Initial checkpoint is a control, evaluated with the same CPU inference path.
    initial_folder=out/'epoch_00';initial_folder.mkdir(exist_ok=True)
    for view in VIEWS:
        source=V3/f"fold_0/learning/{plan['source_candidate']}_seed42/validation_{view}.npz"
        target=initial_folder/f'validation_{view}.npz'
        if not target.exists():shutil.copy2(source,target)
    rows.extend(operating_rows(out,0,model,plan))
    history=[]
    for epoch in range(1,5):
        folder=out/f'epoch_{epoch:02d}';folder.mkdir(exist_ok=True);checkpoint=folder/'checkpoint.pt'
        if not checkpoint.exists():
            if epoch>1:
                state=torch.load(out/f'epoch_{epoch-1:02d}/checkpoint.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);del state
            rate=plan['learning_rates'][epoch-1];optimizer.param_groups[0]['lr']=rate
            rng=np.random.default_rng(plan['seed']+1009*epoch);order=rng.permutation(len(weights));started=time.perf_counter();loss_sum=0.;model.train()
            write_json(out/'execution_state.json',dict(status='running',stage='fine_tuning',epoch=epoch))
            for begin in range(0,len(order),128):
                group=order[begin:begin+128];optimizer.zero_grad()
                for pos in range(0,len(group),16):
                    ids=group[pos:pos+16];views=rng.integers(3,size=len(ids));inputs=batch(banks,ids,'cpu',views)
                    target=torch.from_numpy(np.asarray(banks[0]['labels'][ids],np.float32));w=torch.from_numpy(weights[ids])
                    logits=model(*inputs);loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*w).sum()/len(group)
                    assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())*len(group)
                nn.utils.clip_grad_norm_(model.parameters(),5.,foreach=False);optimizer.step()
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),
                training_loss=loss_sum/len(order),training_seconds=time.perf_counter()-started,learning_rate=rate,**memory_snapshot())
            torch.save(state,folder/'checkpoint_writing.pt');(folder/'checkpoint_writing.pt').replace(checkpoint)
            print('Refinement training epoch',epoch,'seconds',round(state['training_seconds'],1),flush=True)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        history.append({k:state[k] for k in ('epoch','training_loss','training_seconds','learning_rate','process_private_commit_mib')});del state
        rows.extend(operating_rows(out,epoch,model,plan));write_csv(out/'training_history.csv',history)
    chosen=select_point(rows,plan['clean_floor'],plan['fpr_ceiling']);assert chosen is not None
    passed=chosen['mean_macro']>=plan['reference']['mean_macro']+.01 and chosen['mean_changed_f1']>=plan['reference']['mean_changed_f1']
    checkpoint=Path(plan['source_checkpoint']) if chosen['epoch']==0 else out/f"epoch_{chosen['epoch']:02d}/checkpoint.pt"
    frozen=dict(passed=True,promotion_gate_passed=bool(passed),selection=chosen,checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        reference=plan['reference'],design_sha256=digest(out/'design.json'),validation_only=True,new_test_cases_scored=0,source_checkpoint_unchanged=digest(plan['source_checkpoint'])==plan['source_sha256'])
    write_json(out/'all_validation_thresholds.json',rows);write_json(out/'frozen.json',frozen)
    write_json(out/'execution_state.json',dict(status='validation_complete',promotion_gate_passed=bool(passed),next_step='Frozen held-out evaluation and replication' if passed else 'Retain v2; fixed refinement budget exhausted'))
    print('REFINEMENT FROZEN',json.dumps(frozen),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();run(a.out)
