"""Multiscale metric-fusion network and fixed learning ablations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from learning_v2_runtime import ROOT,DEFAULT,VARIANTS,THRESHOLDS,arrays,fold_dir,setup,digest,write_json,write_csv,MemoryTimer,memory_snapshot
from multipair_metrics import balanced_training_weights, choose_threshold, binary_scores

# Exact same frozen grid; avoid importing the older training module and its
# scikit-learn libraries into every CUDA worker merely to obtain this constant.


class MetricFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU())
        self.point_head=nn.Sequential(nn.Linear(1536,128),nn.ReLU(),nn.Dropout(.1))
        self.geometry_head=nn.Sequential(nn.Linear(165,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(192,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x,mask,geo):
        f=self.encoder(x)
        maximum=f.masked_fill(~mask[...,None],-1e9).max(dim=3).values
        maximum=torch.where(mask.any(dim=3,keepdim=True),maximum,torch.zeros_like(maximum))
        mean=(f*mask[...,None]).sum(dim=3)/mask.sum(dim=3,keepdim=True).clamp_min(1)
        pooled=torch.cat((maximum,mean),dim=-1);a,b=pooled[:,:,0],pooled[:,:,1]
        interaction=torch.cat((a,b,torch.abs(a-b),a*b),dim=-1).flatten(1)
        return self.head(torch.cat((self.point_head(interaction),self.geometry_head(geo)),dim=1)).squeeze(-1)


_HOST_BUFFERS={}


def batch_arrays(bank,ids,views=None,reuse=False):
    ids=np.asarray(ids);selected=np.zeros(len(ids),np.int8) if views is None else np.asarray(views).copy()
    for view in (1,2):
        positions=np.flatnonzero(selected==view)
        if len(positions):selected[positions[~np.asarray(bank[view]['available'][ids[positions]],bool)]]=0
    values=[]
    for key in ('x','mask','geo'):
        shape=(len(ids),*bank[0][key].shape[1:]);dtype=bank[0][key].dtype
        cache_key=(key,shape,np.dtype(dtype).str)
        if reuse:
            if cache_key not in _HOST_BUFFERS:_HOST_BUFFERS[cache_key]=(np.empty(shape,dtype),np.empty(shape,dtype))
            result,gather=_HOST_BUFFERS[cache_key]
        else:result=np.empty(shape,dtype);gather=None
        for view in np.unique(selected):
            positions=np.flatnonzero(selected==view)
            if reuse:
                requested=ids[positions]
                assert (requested>=0).all() and (requested<len(bank[int(view)][key])).all()
                np.take(bank[int(view)][key],requested,axis=0,out=gather[:len(positions)],mode='clip')
                result[positions]=gather[:len(positions)]
            else:result[positions]=bank[int(view)][key][ids[positions]]
        values.append(result)
    return values


def batch(bank,ids,device,views=None):
    # Blocking CUDA copies allow a bounded set of host buffers to be reused.
    # CPU callers receive independent storage so autograd inputs cannot alias.
    return [torch.from_numpy(value).to(device,non_blocking=False)
        for value in batch_arrays(bank,ids,views,reuse=str(device).startswith('cuda'))]


@torch.no_grad()
def probabilities(model,data,device='cuda',batch_size=256):
    model.eval();parts=[]
    for start in range(0,len(data['labels']),batch_size):
        ids=np.arange(start,min(start+batch_size,len(data['labels'])))
        parts.append(torch.sigmoid(model(*batch([data],ids,device))).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0,np.float32)


def hard_order(labels,pairs,losses,rng):
    """Preserve each original pair/class group count while emphasizing hard rows."""
    parts=[]
    for pair in np.unique(pairs):
        for label in (0,1):
            ids=np.flatnonzero((pairs==pair)&(labels==label))
            if not len(ids):continue
            hardness=np.asarray(losses[ids],np.float64).clip(.05,5.)
            probability=.7/len(ids)+.3*hardness/hardness.sum()
            parts.append(rng.choice(ids,len(ids),replace=True,p=probability))
    return rng.permutation(np.concatenate(parts))


def archive_sources(folder):
    source=folder/'source';source.mkdir(exist_ok=True)
    names=['learning_v2_data.py','learning_v2_model.py','learning_v2_runtime.py','multipair_metrics.py','multipair_features.py',
        'multipair_data.py','multipair_learning.py','m3c2_baseline.py','c2c_robustness.py','siamese_pointnet.py']
    result={}
    for name in names:shutil.copy2(ROOT/'code'/name,source/name);result[name]=digest(source/name)
    return result


def validation_score(data,probability,threshold):
    keep=data['available'];code=data['pair']*2+data['scan'];scores=[]
    for group in np.unique(code):
        mask=keep&(code==group);scores.append(binary_scores(data['labels'][mask],(probability[mask]>=threshold).astype(np.int8)))
    return dict(binary_macro_f1=float(np.mean([r['binary_macro_f1'] for r in scores])),
        changed_f1=float(np.mean([r['f1_changed'] for r in scores])),available_cores=int(keep.sum()))


def train(out=DEFAULT,fold=0,variant=VARIANTS[0],seed=42,epoch_chunk=None):
    out=Path(out);folder=fold_dir(out,fold)/f'learning/{variant}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():return json.loads((folder/'frozen.json').read_text())
    design=json.loads((out/'design.json').read_text());setup(seed);assert torch.cuda.is_available()
    banks=[arrays(out,fold,'train',v) for v in (('clean',) if variant.endswith('_clean') else ('clean','mild','medium'))]
    val=arrays(out,fold,'validation','clean');labels=np.asarray(banks[0]['labels']);pairs=np.asarray(banks[0]['pair'])
    weights=balanced_training_weights(labels,pairs);n=len(labels)
    for b in banks[1:]:
        for key in ('labels','pair','scan','indices'):np.testing.assert_array_equal(b[key],banks[0][key])
    config=dict(variant=variant,seed=seed,fold=fold,epochs=design['epochs'],batch_size=design['batch_size'],
        learning_rate=.001,weight_decay=.001,architecture=design['architecture'],
        input_channels='relative XYZ (3), bounded opposite-cloud NN displacement (3), capped distance (1), match present (1)',
        loss='Existing equal pair/present-class weighted BCE',checkpoint=design['checkpoint'],
        hard_sampling=design['hard_sampling'] if variant.endswith('_hard') else 'none',
        robust_views=len(banks),threshold_grid=THRESHOLDS.tolist(),design_sha256=digest(out/'design.json'),
        execution='Checkpoint-preserving worker recycling under Windows commit limits; effective batch and total epochs unchanged',
        protocol_sha256=digest(fold_dir(out,fold)/'protocol.json'),source_sha256=archive_sources(folder),test_accessed=False)
    write_json(folder/'config.json',config)
    model=MetricFusion().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');start=0;history=[];hardness=np.ones(n,np.float32)
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        best=state['best'];start=state['epoch']+1;history=state['history'];hardness=state['hardness']
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        del state
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    prior_seconds=float(history[-1].get('seconds',0.)) if history else 0.
    stop_epoch=min(design['epochs'],start+epoch_chunk) if epoch_chunk else design['epochs']
    with MemoryTimer() as timer:
        for epoch in range(start,stop_epoch):
            rng=np.random.default_rng(seed+1009*epoch)
            order=hard_order(labels,pairs,hardness,rng) if variant.endswith('_hard') else rng.permutation(n)
            model.train();loss_sum=0.
            for begin in range(0,n,design['batch_size']):
                ids=order[begin:begin+design['batch_size']]
                views=rng.integers(len(banks),size=len(ids))
                inputs=batch(banks,ids,'cuda',views)
                target=torch.from_numpy(labels[ids].astype(np.float32)).cuda();w=torch.from_numpy(weights[ids]).cuda()
                optimizer.zero_grad(set_to_none=True);logits=model(*inputs)
                raw=nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
                loss=(raw*w).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                hardness[ids]=.7*hardness[ids]+.3*raw.detach().cpu().numpy()
                loss_sum+=float(loss.detach())*len(ids)
            probability=probabilities(model,val).clip(1e-7,1-1e-7);y=val['labels']
            vloss=float(-np.mean(y*np.log(probability)+(1-y)*np.log(1-probability)))
            improved=vloss<best
            if improved:best=vloss
            history.append(dict(epoch=epoch+1,weighted_training_bce=loss_sum/n,clean_validation_bce=vloss,
                seconds=prior_seconds+time.perf_counter()-started,**memory_snapshot()))
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
                history=history,hardness=hardness,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            if improved:
                torch.save(state,folder/'best_tmp.pt');(folder/'best_tmp.pt').replace(folder/'best.pt')
            torch.save(state,folder/'last_tmp.pt');(folder/'last_tmp.pt').replace(folder/'last.pt')
            write_csv(folder/'history.csv',history)
            if epoch==0 or (epoch+1)%5==0:print('V2',fold,variant,seed,'epoch',epoch+1,'clean val BCE',round(vloss,4),flush=True)
        if stop_epoch<design['epochs']:
            print('V2 checkpoint/recycle',fold,variant,seed,'epoch',stop_epoch,flush=True)
            return dict(partial=True,completed_epoch=stop_epoch)
        state=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['model'])
        probability=probabilities(model,val)
        chosen,thresholds=choose_threshold(val['labels'],probability,val['pair']*2+val['scan'],THRESHOLDS)
        write_csv(folder/'thresholds.csv',thresholds)
        validation={}
        for view in ('clean','mild','medium'):
            data=val if view=='clean' else arrays(out,fold,'validation',view)
            prob=probability if view=='clean' else probabilities(model,data)
            validation[view]=validation_score(data,prob,chosen['threshold'])
            np.savez_compressed(folder/f'validation_{view}.npz',probability=prob,labels=data['labels'],pair=data['pair'],
                scan=data['scan'],indices=data['indices'],available=data['available'])
            if view!='clean':del data
    result=dict(variant=variant,seed=seed,fold=fold,threshold=chosen['threshold'],validation_objective=chosen['objective'],
        validation_changed_f1=chosen['changed_f1'],validation_by_view=validation,
        robustness_validation_objective=float(np.mean([v['binary_macro_f1'] for v in validation.values()])),
        best_epoch=state['epoch']+1,best_clean_validation_bce=state['best'],
        checkpoint_sha256=digest(folder/'best.pt'),config_sha256=digest(folder/'config.json'),
        design_sha256=digest(out/'design.json'),test_accessed=False,training_cores=n,
        parameter_count=sum(p.numel() for p in model.parameters()),gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,timing=timer.stats)
    result['timing_scope']='Final worker chunk; history seconds accumulate training time over all chunks'
    write_json(folder/'frozen.json',result);print('V2 frozen',json.dumps(result),flush=True)
    return result


def boosted(out=DEFAULT,fold=0):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    folder=fold_dir(out,fold)/'learning/bounded_geometry';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():return
    train_data=arrays(out,fold,'train','clean');val=arrays(out,fold,'validation','clean')
    weights=balanced_training_weights(train_data['labels'],train_data['pair'])
    config=dict(model='HistGradientBoostingClassifier',max_iter=200,max_leaf_nodes=15,learning_rate=.05,
        l2_regularization=1.,min_samples_leaf=30,early_stopping=False,random_state=42,
        geometry='Same 165 bounded multiscale geometry channels used by neural head; no point encoder',
        source_sha256=archive_sources(folder),design_sha256=digest(Path(out)/'design.json'))
    write_json(folder/'config.json',config)
    with threadpool_limits(limits=2):
        model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.,
            min_samples_leaf=30,early_stopping=False,random_state=42)
        model.fit(train_data['geo'],train_data['labels'],sample_weight=weights)
        probability=model.predict_proba(val['geo'])[:,1]
    choice,rows=choose_threshold(val['labels'],probability,val['pair']*2+val['scan'],THRESHOLDS)
    write_csv(folder/'thresholds.csv',rows);joblib.dump(model,folder/'model.joblib')
    result=dict(threshold=choice['threshold'],validation_objective=choice['objective'],validation_changed_f1=choice['changed_f1'],
        checkpoint_sha256=digest(folder/'model.joblib'),config_sha256=digest(folder/'config.json'),test_accessed=False)
    write_json(folder/'frozen.json',result);print('V2 bounded geometry frozen',json.dumps(result),flush=True)


def freeze_selection(out=DEFAULT,fold=0):
    folder=fold_dir(out,fold)/'learning';path=folder/'selection.json'
    if path.exists():return json.loads(path.read_text())
    candidates=[json.loads((folder/f'{v}_seed42/frozen.json').read_text()) for v in VARIANTS]
    clean=max(candidates,key=lambda c:(c['validation_objective'],-VARIANTS.index(c['variant'])))
    robust=max(candidates,key=lambda c:(c['robustness_validation_objective'],-VARIANTS.index(c['variant'])))
    result=dict(primary_variant=clean['variant'],secondary_robust_variant=robust['variant'],candidates=candidates,
        selection_data='Validation only; clean primary, secondary mean of three views',test_accessed=False,
        design_sha256=digest(Path(out)/'design.json'))
    write_json(path,result);print('V2 selection',fold,clean['variant'],'secondary',robust['variant'],flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','boosted','select'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--variant',default=VARIANTS[0],choices=VARIANTS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epoch-chunk',type=int);a=p.parse_args()
    if a.stage=='train':train(a.out,a.fold,a.variant,a.seed,getattr(a,'epoch_chunk',None))
    elif a.stage=='boosted':boosted(a.out,a.fold)
    else:freeze_selection(a.out,a.fold)
