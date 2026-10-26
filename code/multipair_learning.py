"""Controlled nearest/voxel/local-augmentation ablations; validation-only selection."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from siamese_pointnet import SiamesePointNet,setup,tensor_batch,predict
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from multipair_data import ROOT,DEFAULT,PAIRS
from multipair_features import assemble
from multipair_metrics import balanced_training_weights,choose_threshold
from m3c2_baseline import digest,write_json,MemoryTimer

THRESHOLDS=np.r_[.01,.02,np.arange(.05,.951,.05).round(3),.98,.99]
VARIANTS=('nearest','voxel','voxel_aug')


def augment_batch(batch):
    """Local patch augmentation; not a simulation of full-scan visibility changes."""
    p0,m0,p1,m1,aux=[x.clone() for x in batch]
    n=len(p0)
    angle=(torch.rand(n,device=p0.device)*2-1)*torch.pi
    cosine,sine=angle.cos(),angle.sin()
    for p in (p0,p1):
        x=p[:,:,0].clone();y=p[:,:,1].clone()
        p[:,:,0]=cosine[:,None]*x-sine[:,None]*y
        p[:,:,1]=sine[:,None]*x+cosine[:,None]*y
        p.add_(torch.randn_like(p).clamp(-3,3)*(.01/.6))
    for j,(p,mask) in enumerate(((p0,m0),(p1,m1))):
        retention=.6+.4*torch.rand(n,device=p.device)
        mask &= torch.rand(mask.shape,device=p.device)<retention[:,None]
        p.masked_fill_(~mask[:,:,None],0.)
        # Approximate raw density reduction; true full-scan NN prior retained.
        raw=torch.expm1(aux[:,j]*np.log1p(10000))
        aux[:,j]=torch.log1p(raw*retention)/np.log1p(10000)
    return [p0,m0,p1,m1,aux]


def archive_sources(folder):
    dst=folder/'source';dst.mkdir(exist_ok=True)
    names=['multipair_learning.py','multipair_features.py','multipair_metrics.py','siamese_pointnet.py']
    for name in names:shutil.copy2(ROOT/'code'/name,dst/name)
    return {name:digest(dst/name) for name in names}


def neural(out,protocol,variant,seed,pairs,epochs=30):
    out=Path(out);folder=out/f'learning/{protocol}/{variant}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():
        return json.loads((folder/'frozen.json').read_text())
    setup(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    assert device=='cuda','Use the verified existing CUDA environment'
    train=assemble(out,'nearest' if variant=='nearest' else 'voxel','train',pairs)
    val=assemble(out,'nearest' if variant=='nearest' else 'voxel','validation',pairs)
    weights=balanced_training_weights(train['labels'],train['pair'])
    config=dict(protocol=protocol,variant=variant,seed=seed,pairs=list(pairs),epochs=epochs,
        batch_size=128,learning_rate=.001,weight_decay=.001,architecture='Existing shared PointNet 3-32-64; masked mean/max; 387-64-32-1 head',
        training_balance='Equal pair weight; equal present-class weight within each pair',
        checkpoint_selection='Minimum unweighted validation BCE; equal number of cores per pair/direction',
        threshold_selection='Mean pair/direction binary macro F1, validation only',
        threshold_grid=THRESHOLDS.tolist(),
        augmentation=('Common XY patch rotations, independent clipped 1 cm jitter and 60-100% patch retention; density auxiliary approximately adjusted; full-scan NN prior retained'
            if variant=='voxel_aug' else 'none'),
        no_full_scan_augmentation_claim=True,device=device,protocol_sha256=digest(out/'protocol.json'),
        source_sha256=archive_sources(folder),features_have_no_labels_or_absolute_coordinates=True)
    write_json(folder/'config.json',config)
    model=SiamesePointNet().to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    history=[];best=float('inf');start=0
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        start=state['epoch']+1;best=state['best'];history=state['history']
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    for epoch in range(start,epochs):
        model.train();order=np.random.default_rng(seed+epoch*1009).permutation(len(weights));loss_sum=0.
        for begin in range(0,len(order),128):
            ids=order[begin:begin+128];batch=tensor_batch(train,ids,device)
            if variant=='voxel_aug':batch=augment_batch(batch)
            target=torch.from_numpy(train['labels'][ids].astype(np.float32)).to(device)
            weight=torch.from_numpy(weights[ids]).to(device)
            optim.zero_grad();logits=model(*batch)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
            loss_sum+=float(loss.detach())*len(ids)
        probability=predict(model,val,device)
        probability=np.clip(probability,1e-7,1-1e-7);y=val['labels']
        vloss=float(-np.mean(y*np.log(probability)+(1-y)*np.log(1-probability)))
        improved=vloss<best
        if improved:best=vloss
        history.append(dict(epoch=epoch+1,weighted_training_bce=loss_sum/len(weights),validation_bce=vloss))
        state=dict(epoch=epoch,model=model.state_dict(),optimizer=optim.state_dict(),best=best,
            history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(state,folder/'last.pt')
        if improved:torch.save(state,folder/'best.pt')
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        if epoch==0 or (epoch+1)%5==0:print(protocol,variant,seed,'epoch',epoch+1,'validation BCE',round(vloss,4),flush=True)
    state=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
    probability=predict(model,val,device)
    choice,rows=choose_threshold(val['labels'],probability,val['pair']*2+val['scan'],THRESHOLDS)
    pd.DataFrame(rows).to_csv(folder/'thresholds.csv',index=False)
    np.savez_compressed(folder/'validation_predictions.npz',probability=probability,labels=val['labels'],pair=val['pair'],scan=val['scan'],indices=val['indices'])
    result=dict(protocol=protocol,variant=variant,seed=seed,threshold=choice['threshold'],validation_objective=choice['objective'],
        validation_changed_f1=choice['changed_f1'],best_epoch=state['epoch']+1,best_validation_bce=state['best'],
        checkpoint_sha256=digest(folder/'best.pt'),config_sha256=digest(folder/'config.json'),
        training_cores=len(weights),validation_cores=len(val['labels']),test_accessed=False,
        seconds_this_process=time.perf_counter()-t0,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write_json(folder/'frozen.json',result)
    print('Frozen neural:',json.dumps(result),flush=True)
    del train,val,model,optim,state;gc.collect();torch.cuda.empty_cache()
    return result


def boosted(out,protocol,pairs):
    folder=Path(out)/f'learning/{protocol}/geometry_boosted';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():return json.loads((folder/'frozen.json').read_text())
    train=assemble(out,'voxel','train',pairs);val=assemble(out,'voxel','validation',pairs)
    weights=balanced_training_weights(train['labels'],train['pair'])
    config=dict(protocol=protocol,pairs=list(pairs),model='HistGradientBoostingClassifier',max_iter=200,max_leaf_nodes=15,
        learning_rate=.05,l2_regularization=1.,min_samples_leaf=30,early_stopping=False,random_state=42,
        selection='Fixed tabular control, same validation-only threshold objective as neural models',
        threshold_grid=THRESHOLDS.tolist(),
        no_absolute_coordinates=True,source_sha256=archive_sources(folder))
    write_json(folder/'config.json',config)
    with MemoryTimer() as timer,threadpool_limits(limits=2):
        model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.,
            min_samples_leaf=30,early_stopping=False,random_state=42)
        model.fit(train['geo'],train['labels'],sample_weight=weights)
        probability=model.predict_proba(val['geo'])[:,1]
    choice,rows=choose_threshold(val['labels'],probability,val['pair']*2+val['scan'],THRESHOLDS)
    joblib.dump(model,folder/'model.joblib');pd.DataFrame(rows).to_csv(folder/'thresholds.csv',index=False)
    result=dict(protocol=protocol,model='geometry_boosted',threshold=choice['threshold'],validation_objective=choice['objective'],
        validation_changed_f1=choice['changed_f1'],checkpoint_sha256=digest(folder/'model.joblib'),
        config_sha256=digest(folder/'config.json'),training_cores=len(weights),test_accessed=False,timing=timer.stats)
    write_json(folder/'frozen.json',result);print('Frozen geometry classifier:',json.dumps(result),flush=True)
    del train,val,model;gc.collect()
    return result


def run(out=DEFAULT,protocols=('primary','temporal')):
    out=Path(out)
    for protocol in protocols:
        folder=out/f'learning/{protocol}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'selection_frozen.json').exists():continue
        pairs=PAIRS if protocol=='primary' else ('01',)
        candidates=[neural(out,protocol,v,42,pairs) for v in VARIANTS]
        selected=max(candidates,key=lambda c:(c['validation_objective'],-VARIANTS.index(c['variant'])))['variant']
        repeated=[next(c for c in candidates if c['variant']==selected)]
        for seed in (43,44):repeated.append(neural(out,protocol,selected,seed,pairs))
        tabular=boosted(out,protocol,pairs)
        write_json(folder/'selection_frozen.json',dict(protocol=protocol,pairs=list(pairs),
            candidates=candidates,selected_variant=selected,seeds=[42,43,44],repeated_models=repeated,geometry_classifier=tabular,
            threshold_policy='Each seed uses validation only. Architecture fixed from seed42 before test access.',
            test_accessed=False,protocol_sha256=digest(out/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--protocols',nargs='+',default=['primary','temporal']);a=p.parse_args();run(a.out,a.protocols)
