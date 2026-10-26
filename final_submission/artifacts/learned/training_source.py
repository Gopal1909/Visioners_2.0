"""Compact shared PointNet-style paired-patch baseline with geometric NN prior.

One binary prediction per requested original core in each epoch. Two patches are
centered at the SAME query position and keep their relative displacement. The
shared point MLP pools masked mean/max features; a fusion head compares epochs.
No absolute coordinates or labels enter the features. This is a course baseline,
not a reproduction of Siamese KPConv/PGN3DCD and not a full-cloud segmentation run.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial import cKDTree
from study_pipeline import ROOT,DEFAULT,CONDITIONS,load_pair,condition_cloud
from m3c2_baseline import MemoryTimer,write_json,digest,score_pair

RADIUS=.6
K=128


class SiamesePointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(387,64),nn.ReLU(),nn.Dropout(.1),
                                nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def encode(self,p,mask):
        f=self.encoder(p)
        maximum=f.masked_fill(~mask[...,None],-1e9).max(dim=1).values
        maximum=torch.where(mask.any(dim=1,keepdim=True),maximum,torch.zeros_like(maximum))
        mean=(f*mask[...,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
        return torch.cat((maximum,mean),dim=1)

    def forward(self,p0,m0,p1,m1,aux):
        a,b=self.encode(p0,m0),self.encode(p1,m1)
        return self.head(torch.cat((a,b,torch.abs(a-b),aux),dim=1)).squeeze(-1)


def paired_features(source_tree,target_tree,centers,k=K,radius=RADIUS):
    """k nearest within fixed Euclidean radius; zero padding and masks if absent.

    Patch truncation is an explicit learned fixed-input design, distinct from the
    complete-support geometric cylinder estimator. This function cannot access GT.
    """
    patches=[];masks=[];aux=[]
    for tree in (source_tree,target_tree):
        dist,ids=tree.query(centers,k=k,distance_upper_bound=radius,workers=1)
        mask=np.isfinite(dist)
        safe=np.minimum(ids,max(0,len(tree.data)-1))
        coords=tree.data[safe] if len(tree.data) else np.zeros((*safe.shape,3))
        relative=(coords-centers[:,None,:])/radius
        relative[~mask]=0.
        patches.append(relative.astype('float32'));masks.append(mask)
        aux.append(mask.sum(1).astype('float32')/k)
        if len(patches)==2:aux.append(np.minimum(dist[:,0],radius).astype('float32')/radius)
    return dict(p0=patches[0],m0=masks[0],p1=patches[1],m1=masks[1],
                aux=np.column_stack(aux).astype('float32'))


def save_feature_cache(path,trees,centers,labels):
    arrays={};stats={}
    with MemoryTimer() as timer:
        for i in (0,1):
            f=paired_features(trees[i],trees[1-i],centers[i])
            arrays.update({f'{key}_{i}':v for key,v in f.items()})
            arrays[f'y_{i}']=labels[i].astype('float32')
            stats[f'scan{i}_empty_target_fraction']=float((~f['m1'].any(1)).mean())
        np.savez_compressed(path,**arrays)
    return dict(**timer.stats,**stats)


def read_cache(path):
    with np.load(path) as z:
        return {key:np.concatenate([z[f'{key}_{i}'] for i in (0,1)]) for key in ('p0','m0','p1','m1','aux','y')}


def setup(seed=42):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def tensor_batch(features,ids,device):
    return [torch.from_numpy(features[key][ids]).to(device) for key in ('p0','m0','p1','m1','aux')]


@torch.no_grad()
def predict(model,f,device,batch=256):
    model.eval();parts=[]
    for start in range(0,len(f['aux']),batch):
        ids=slice(start,start+batch)
        parts.append(torch.sigmoid(model(*tensor_batch(f,ids,device))).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0,dtype='float32')


def train(out,epochs=30):
    out=Path(out);folder=out/'learned';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model_frozen.json').exists():
        print('Frozen model retained',flush=True);return
    setup();device='cuda' if torch.cuda.is_available() else 'cpu'
    config=dict(seed=42,epochs=epochs,batch_size=128,learning_rate=.001,
        weight_decay=.0001,patch_k=K,patch_radius_m=RADIUS,features='relative XYZ, masks, capped source/target counts, capped target NN distance',
        architecture='shared 3-32-64 point MLP; masked max+mean; concat(a,b,abs(a-b),3 auxiliary geometry); 387-64-32-1 head, dropout .1',
        loss='BCEWithLogitsLoss, positive weight = training negatives / positives',
        model_selection='minimum weighted calibration BCE over 30 predetermined epochs',
        threshold_grid=[.2,.3,.4,.5,.6,.7,.8],threshold_selection='calibration pooled directional 3-class macro F1 only',
        device=device,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        split_sha256=digest(out/'splits.npz'),source_sha256=digest(Path(__file__)),
        no_gt_in_features=True,augmentation='none; clean-only training to measure untrained stress sensitivity')
    write_json(folder/'model_config.json',config)
    splits=dict(np.load(out/'splits.npz'))
    missing=[s for s in ('train','calibration') if not (folder/f'{s}_patches.npz').exists()]
    if missing:
        data=load_pair()
        with MemoryTimer() as timer: trees=[cKDTree(d['xyz']) for d in data]
        tree_stats=timer.stats
        for split in missing:
            centers=[splits[f'{split}_xyz{i}'].astype('float64') for i in (0,1)]
            labels=[splits[f'{split}_gt{i}'] for i in (0,1)]
            stats=save_feature_cache(folder/f'{split}_patches.npz',trees,centers,labels)
            write_json(folder/f'{split}_feature_runtime.json',dict(tree_build=tree_stats,patches=stats))
        del data,trees;gc.collect()
    tr=read_cache(folder/'train_patches.npz');val=read_cache(folder/'calibration_patches.npz')
    y=tr['y'];positive=float(y.sum());negative=float(len(y)-positive)
    if positive<10 or negative<10:raise ValueError('Insufficient training-class support')
    posweight=negative/positive
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posweight,device=device))
    model=SiamesePointNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];start_epoch=0;best=float('inf');last=folder/'last_checkpoint.pt'
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']+1;best=state['best_loss'];history=state['history']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start_time=time.perf_counter()
    with MemoryTimer() as timer:
        for epoch in range(start_epoch,epochs):
            model.train();order=np.random.default_rng(42+epoch).permutation(len(y));losses=[]
            for start in range(0,len(y),128):
                ids=order[start:start+128]
                target=torch.from_numpy(y[ids]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(*tensor_batch(tr,ids,device));loss=criterion(logits,target)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
            prob=predict(model,val,device)
            logits=torch.logit(torch.from_numpy(prob).to(device).clamp(1e-6,1-1e-6))
            vloss=float(criterion(logits,torch.from_numpy(val['y']).to(device)).cpu())
            improved=vloss<best
            if improved:best=vloss
            row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),calibration_loss=vloss,
                     elapsed_seconds=time.perf_counter()-start_time)
            history.append(row);pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            state=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),best_loss=best,
                history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
            torch.save(state,folder/'checkpoint_tmp.pt');(folder/'checkpoint_tmp.pt').replace(last)
            if improved:torch.save(state,folder/'best_checkpoint.pt')
            print(f'Epoch {epoch+1}/{epochs} train={row["train_loss"]:.4f} calibration={vloss:.4f}',flush=True)
    stats=timer.stats
    stats.update(gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,
                 gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0,
                 positive_weight=posweight,training_points=len(y),training_changed=int(positive))
    write_json(folder/'training_runtime.json',stats)
    state=torch.load(folder/'best_checkpoint.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model']);prob=predict(model,val,device)
    lengths=[len(splits[f'calibration_gt{i}']) for i in (0,1)]
    g=[splits[f'calibration_gt{i}'] for i in (0,1)]
    probabilities=np.split(prob,[lengths[0]])
    rows=[]
    for threshold in config['threshold_grid']:
        preds=[(v>=threshold).astype('int8') for v in probabilities]
        rows.append(dict(threshold=threshold,**score_pair(g,preds,[np.ones(len(a),bool) for a in g])))
    df=pd.DataFrame(rows);df.to_csv(folder/'calibration_thresholds.csv',index=False)
    chosen=df.iloc[int(np.argmax(df.combined_macro_f1))]
    write_json(folder/'model_frozen.json',dict(best_epoch=int(state['epoch']+1),threshold=float(chosen.threshold),
        best_calibration_loss=float(state['best_loss']),calibration_macro_f1=float(chosen.combined_macro_f1),
        checkpoint_sha256=digest(folder/'best_checkpoint.pt'),configuration_sha256=digest(folder/'model_config.json'),
        split_sha256=digest(out/'splits.npz'),frozen_before_test_inference=True,
        caveat='Same-scene spatial holdout, one seed and architecture, inherited historical pilot/C2C test exposure'))
    np.savez_compressed(folder/'calibration_predictions.npz',probability=prob,labels=val['y'])
    print('Model frozen:',chosen.threshold,'epoch',state['epoch']+1,flush=True)


def evaluate(out,only=None):
    out=Path(out);folder=out/'learned'
    frozen=json.loads((folder/'model_frozen.json').read_text())
    assert digest(folder/'best_checkpoint.pt')==frozen['checkpoint_sha256']
    setup();device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SiamesePointNet().to(device)
    state=torch.load(folder/'best_checkpoint.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model']);model.eval()
    data=load_pair()
    with MemoryTimer() as timer: tree0=cKDTree(data[0]['xyz'])
    tree0time=timer.stats
    for name,kind,value in CONDITIONS:
        if only and name not in only:continue
        source=out/'geometry'/name
        if not (source/'complete.json').exists():raise FileNotFoundError(f'Geometry checkpoint missing: {name}')
        dst=folder/'evaluation'/name;dst.mkdir(parents=True,exist_ok=True)
        if (dst/'complete.json').exists():
            print(name,'retained',flush=True);continue
        measurements=dict(np.load(source/'measurements.npz'))
        xyz1,ids1=condition_cloud(data,kind,value)
        positions=np.searchsorted(ids1,measurements['indices1'])
        np.testing.assert_array_equal(ids1[positions],measurements['indices1'])
        np.testing.assert_array_equal(xyz1[positions],measurements['xyz1'])
        with MemoryTimer() as timer:trees=[tree0,cKDTree(xyz1)]
        tree1time=timer.stats
        pred=[];prob=[];stats=[]
        for i in (0,1):
            with MemoryTimer() as timer:
                f=paired_features(trees[i],trees[1-i],measurements[f'xyz{i}'])
                if device=='cuda':torch.cuda.synchronize()
                probabilities=predict(model,f,device)
                if device=='cuda':torch.cuda.synchronize()
            stats.append(timer.stats)
            prob.append(probabilities);pred.append((probabilities>=frozen['threshold']).astype('int8'))
            del f
        gt=[measurements[f'gt{i}'] for i in (0,1)]
        masks=[measurements[f'decision{i}_valid'] for i in (0,1)]
        rows=[]
        for protocol in ('A','B') if kind=='occlusion' else ('standard',):
            eligible=[np.ones(len(g),bool) for g in gt]
            if protocol=='B':
                center=data[1]['xyz'].mean(axis=0)
                eligible[0]=np.linalg.norm(measurements['xyz0'][:,:2]-center[:2],axis=1)>value
            g=[x[e] for x,e in zip(gt,eligible)];pp=[x[e] for x,e in zip(pred,eligible)]
            for scope in ('common','all'):
                mm=[m[e] for m,e in zip(masks,eligible)] if scope=='common' else [np.ones(len(x),bool) for x in g]
                rows.append(dict(condition=name,family=kind,severity=value,protocol=protocol,
                    method=f'Siamese PointNet {scope}',**score_pair(g,pp,mm),
                    geometry_seconds=tree0time['seconds']+tree1time['seconds']+sum(s['seconds'] for s in stats),
                    sampled_process_peak_rss_mib=max(s['sampled_peak_rss_mib'] for s in stats)))
        np.savez_compressed(dst/'predictions.npz',probability0=prob[0],probability1=prob[1],
            prediction0=pred[0],prediction1=pred[1],indices0=measurements['indices0'],indices1=measurements['indices1'])
        pd.DataFrame(rows).to_csv(dst/'metrics.csv',index=False)
        write_json(dst/'timing.json',dict(tree0=tree0time,tree1=tree1time,query_patch_inference=stats))
        write_json(dst/'complete.json',dict(checkpoint_sha256=frozen['checkpoint_sha256'],threshold=frozen['threshold'],
            geometry_measurement_sha256=digest(source/'measurements.npz'),no_stress_retuning=True))
        print(name,pd.DataFrame(rows)[['method','protocol','combined_macro_f1','coverage']].to_string(index=False),flush=True)
        del xyz1,ids1,trees,measurements;gc.collect()
    frames=[pd.read_csv(p) for p in (folder/'evaluation').glob('*/metrics.csv') if (p.parent/'complete.json').exists()]
    if frames:pd.concat(frames,ignore_index=True).to_csv(folder/'evaluation_metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','evaluate']);parser.add_argument('--out',default=str(DEFAULT))
    parser.add_argument('--only',nargs='+');parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    if args.stage=='train':train(args.out,args.epochs)
    else:evaluate(args.out,args.only)
