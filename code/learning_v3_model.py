"""Custom hierarchical local aggregation and cross-scan attention, pure PyTorch.

This is not an EF-Siamese KPConv implementation or a published-model reproduction.
All neighborhoods remain inside the frozen v2 feature support bound.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from learning_v3_runtime import (BASE,DEFAULT,setup,digest,write_json,write_csv,
    ROOT,CANDIDATES,EPOCHS,fold_dir,bank,validation,THRESHOLDS,MemoryTimer,memory_snapshot)
from learning_v2_model import MetricFusion,batch,probabilities,validation_score
from multipair_metrics import balanced_training_weights,choose_threshold
from learning_v3_optim import FunctionalAdamW


def gather(points,indices):
    rows=torch.arange(len(points),device=points.device)
    return points[rows.reshape(-1,*([1]*(indices.ndim-1))),indices]


@torch.no_grad()
def anchors(xyz,mask,count=16):
    """Masked farthest point sampling, starting nearest the original core."""
    distance=xyz.square().sum(-1).masked_fill(~mask,float('inf'))
    choice=distance.argmin(-1);minimum=torch.full_like(distance,float('inf'))
    unused=mask.clone();ids=[];valid=[]
    for _ in range(count):
        present=unused.any(-1);ids.append(choice);valid.append(present)
        point=gather(xyz,choice[:,None]).squeeze(1)
        minimum=torch.minimum(minimum,(xyz-point[:,None]).square().sum(-1))
        unused.scatter_(1,choice[:,None],False)
        choice=minimum.masked_fill(~unused,-1).argmax(-1)
    return torch.stack(ids,1),torch.stack(valid,1)


def masked_pool(value,mask,dim):
    maximum=value.masked_fill(~mask[...,None],-1e4).max(dim).values
    maximum=torch.where(mask.any(dim)[...,None],maximum,torch.zeros_like(maximum))
    mean=(value*mask[...,None]).sum(dim)/mask.sum(dim).clamp_min(1)[...,None]
    return torch.cat((maximum,mean),-1)


class HierarchicalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(8,32),nn.LayerNorm(32),nn.GELU())
        self.local=nn.Sequential(nn.Linear(67,32),nn.GELU(),nn.Linear(32,64),nn.GELU())
        self.query=nn.Linear(64,32,bias=False);self.key=nn.Linear(64,32,bias=False)
        self.fusion=nn.Sequential(nn.Linear(196,64),nn.LayerNorm(64),nn.GELU())
        self.point_head=nn.Sequential(nn.Linear(1536,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.25))
        self.geometry_head=nn.Sequential(nn.Linear(165,64),nn.LayerNorm(64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,1))

    def forward(self,x,mask,geo):
        b,s,t,n,c=x.shape
        flat=x.reshape(-1,n,c);m=mask.reshape(-1,n);xyz=flat[...,:3]
        feature=self.point(flat)
        with torch.no_grad():
            ids,am=anchors(xyz,m)
            pos=gather(xyz,ids)
            distance=torch.cdist(pos,xyz).masked_fill(~m[:,None],1e4)
            values,neighbors=distance.topk(min(8,n),dim=-1,largest=False,sorted=False)
            nm=(values<=.5)&am[...,None]
        local_feature=gather(feature,neighbors);central=gather(feature,ids)
        delta=gather(xyz,neighbors)-pos[:,:,None]
        edge=torch.cat((central[:,:,None].expand_as(local_feature),local_feature-central[:,:,None],delta),-1)
        local=self.local(edge).masked_fill(~nm[...,None],-1e4).max(2).values
        local=torch.where(nm.any(2)[...,None],local,torch.zeros_like(local))
        local=local.reshape(b,s,2,16,64);pos=pos.reshape(b,s,2,16,3);am=am.reshape(b,s,2,16)
        outputs=[]
        for side in (0,1):
            own=local[:,:,side];other=local[:,:,1-side]
            d=pos[:,:,1-side,None]-pos[:,:,side,:,None]
            d2=d.square().sum(-1)
            valid=am[:,:,side,:,None]&am[:,:,1-side,None,:]&(d2<=1.)
            score=torch.matmul(self.query(own),self.key(other).transpose(-1,-2))/math.sqrt(32)-4*d2
            weight=torch.softmax(score.masked_fill(~valid,-1e4),-1)*valid
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            matched=torch.matmul(weight,other);offset=(weight[...,None]*d).sum(-2)
            fused=self.fusion(torch.cat((own,matched,torch.abs(own-matched),offset,valid.any(-1,keepdim=True)), -1))
            outputs.append(masked_pool(own+fused,am[:,:,side],2))
        a,z=outputs
        mixed=torch.cat((a,z,torch.abs(a-z),a*z),-1).flatten(1)
        return self.head(torch.cat((self.point_head(mixed),self.geometry_head(geo)),1)).squeeze(-1)


def make_model(candidate):
    model=MetricFusion() if candidate.startswith('pool_') else HierarchicalFusion()
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.p=.25
    return model


def execution_device(out):
    path=Path(out)/'execution_amendment_03.json'
    return json.loads(path.read_text())['training_device'] if path.exists() else 'cuda'


def setup_device(seed,device):
    if device=='cuda':setup(seed)
    else:
        np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)


def fit_sanity(out=DEFAULT):
    from learning_v2_runtime import arrays
    setup(42);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    source=arrays(BASE,0,'train','clean');rng=np.random.default_rng(9831);ids=[]
    # Genuine construction labels from training regions only. Equal class count.
    for label in (0,1):ids.extend(rng.choice(np.flatnonzero(source['labels']==label),32,replace=False))
    ids=np.asarray(ids);inputs=[torch.from_numpy(np.asarray(source[k][ids]).copy()).cuda() for k in ('x','mask','geo')]
    y=torch.from_numpy(np.asarray(source['labels'][ids],np.float32)).cuda();rows=[]
    for candidate in ('pool_fresh_bce','graph_fresh_bce'):
        setup(42);model=make_model(candidate).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
        torch.cuda.reset_peak_memory_stats();history=[]
        with MemoryTimer() as timer:
            for step in range(400):
                model.train();optimizer.zero_grad(set_to_none=True)
                logits=model(*inputs);loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                if step%10==0 or step==399:
                    model.eval()
                    with torch.no_grad():
                        prediction=model(*inputs);bce=float(nn.functional.binary_cross_entropy_with_logits(prediction,y))
                        accuracy=float(((prediction>=0)==y.bool()).float().mean())
                    history.append(dict(step=step+1,bce=bce,accuracy=accuracy))
                    if accuracy>=.984375 and bce<.05:break
            row=dict(candidate=candidate,passed=accuracy>=.984375 and bce<.05,steps=step+1,training_accuracy=accuracy,
                training_bce=bce,parameter_count=sum(p.numel() for p in model.parameters()),
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,history=history,
                source='BikeParkingConstruction fold0 training-only 32 original cores per class',
                original_ids=ids.tolist(),test_or_validation_accessed=False)
        row['timing']=timer.stats
        rows.append(row);print('V3 sanity',candidate,'accuracy',accuracy,'BCE',bce,'steps',step+1,flush=True)
        del model,optimizer;torch.cuda.empty_cache()
    result=dict(passed=all(r['passed'] for r in rows),models=rows,scope='Training memorization sanity check, not a test result',source_sha256=digest(__file__))
    write_json(out/'sanity_gate.json',result)
    if not result['passed']:raise RuntimeError('Training-subset learnability gate failed; no full experiments allowed')


def loss_values(logits,target,focal=False):
    bce=nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
    return bce*(1-torch.exp(-bce)).square() if focal else bce


def train(out=DEFAULT,fold=0,candidate=CANDIDATES[0],seed=42,epoch_chunk=2):
    out=Path(out);folder=fold_dir(out,fold)/f'learning/{candidate}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():return
    assert json.loads((out/'sanity_gate.json').read_text())['passed']
    device=execution_device(out);setup_device(seed,device)
    if device=='cuda':assert torch.cuda.is_available()
    model=make_model(candidate).to(device)
    optimizer=FunctionalAdamW(model.parameters(),lr=.0006,weight_decay=.01)
    amendment=json.loads((out/'execution_amendment_01.json').read_text()) if (out/'execution_amendment_01.json').exists() else {}
    microbatch=int(amendment.get('microbatch_size',64));validation_batch=int(amendment.get('validation_batch_size',128))
    if not (folder/'config.json').exists():
        src=folder/'source';src.mkdir(exist_ok=True);manifest={}
        files=[*sorted((ROOT/'code').glob('learning_v3*.py')),*[ROOT/'code'/n for n in
            ('learning_v2_model.py','learning_v2_runtime.py','learning_v2_data.py','multipair_metrics.py','multipair_data.py','c2c_robustness.py')]]
        for p in files:shutil.copy2(p,src/p.name);manifest[p.name]=digest(src/p.name)
        write_json(folder/'config.json',dict(candidate=candidate,fold=fold,seed=seed,design_sha256=digest(out/'design.json'),
            fresh_features_verification_sha256=digest(fold_dir(out,fold)/'fresh_features_verification.json'),
            execution_amendment_sha256=digest(out/'execution_amendment_01.json') if amendment else None,
            optimizer_execution_amendment_sha256=digest(out/'execution_amendment_02.json') if (out/'execution_amendment_02.json').exists() else None,
            training_device=device,device_amendment_sha256=digest(out/'execution_amendment_03.json') if (out/'execution_amendment_03.json').exists() else None,
            torch_version=torch.__version__,native_adamw_kernel_sha256=digest(Path(torch.__file__).parent/'optim/adamw.py'),
            microbatch_size=microbatch,effective_batch_size=128,validation_batch_size=validation_batch,
            source_sha256=manifest,loss='Pair/class weighted focal gamma2' if candidate.endswith('focal') else 'Pair/class weighted BCE',
            test_accessed=False,parameter_count=sum(p.numel() for p in model.parameters())))
    config=json.loads((folder/'config.json').read_text());assert config['design_sha256']==digest(out/'design.json')
    assert config['training_device']==device,'Do not switch devices during a fit'
    for name in ('learning_v3_model.py','learning_v3_runtime.py'):
        assert config['source_sha256'][name]==digest(ROOT/'code'/name),'Training code changed after fit started'
    best=float('inf');start=0;history=[]
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);best=state['best']
        start=state['epoch']+1;history=state['history'];torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        del state
    val=validation(fold,'clean');stop=min(EPOCHS,start+epoch_chunk)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    with MemoryTimer() as timer:
        for epoch in range(start,stop):
            data=bank(out,fold,epoch);weights=balanced_training_weights(data['labels'],data['pair'])
            learning_rate=.0006*(.1+.9*.5*(1+math.cos(math.pi*epoch/(EPOCHS-1))))
            for group in optimizer.param_groups:group['lr']=learning_rate
            order=np.random.default_rng(seed+epoch*1009).permutation(len(data['labels']))
            model.train();loss_sum=0.;started=time.perf_counter()
            # Accumulation preserves the effective 128-core optimization batch.
            for begin in range(0,len(order),128):
                group=order[begin:begin+128];optimizer.zero_grad(set_to_none=True)
                for pos in range(0,len(group),microbatch):
                    ids=group[pos:pos+microbatch];inputs=batch([data],ids,device)
                    y=torch.from_numpy(np.asarray(data['labels'][ids],np.float32)).to(device)
                    w=torch.from_numpy(np.asarray(weights[ids],np.float32)).to(device)
                    logits=model(*inputs);loss=(loss_values(logits,y,candidate.endswith('focal'))*w).sum()/len(group)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();loss_sum+=float(loss.detach())*len(group)
                nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);optimizer.step()
            probability=probabilities(model,val,device=device,batch_size=validation_batch).clip(1e-7,1-1e-7);y=val['labels']
            vloss=float(-np.mean(y*np.log(probability)+(1-y)*np.log(1-probability)));improved=vloss<best
            if improved:best=vloss
            history.append(dict(epoch=epoch+1,training_loss=loss_sum/len(order),clean_validation_bce=vloss,
                learning_rate=learning_rate,seconds=time.perf_counter()-started,**memory_snapshot()))
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,history=history,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
            if improved:torch.save(state,folder/'best_tmp.pt');(folder/'best_tmp.pt').replace(folder/'best.pt')
            torch.save(state,folder/'last_tmp.pt');(folder/'last_tmp.pt').replace(folder/'last.pt')
            write_csv(folder/'history.csv',history);del data
            print('V3 epoch',fold,candidate,seed,epoch+1,'val BCE',round(vloss,4),'seconds',round(history[-1]['seconds'],1),flush=True)
        if stop<EPOCHS:return
        state=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
        probability=probabilities(model,val,device=device,batch_size=validation_batch)
        chosen,rows=choose_threshold(val['labels'],probability,val['pair']*2+val['scan'],THRESHOLDS);write_csv(folder/'thresholds.csv',rows)
        by_view={}
        for view in ('clean','mild','medium'):
            view_data=val if view=='clean' else validation(fold,view)
            p=probability if view=='clean' else probabilities(model,view_data,device=device,batch_size=validation_batch)
            by_view[view]=validation_score(view_data,p,chosen['threshold'])
            np.savez_compressed(folder/f'validation_{view}.npz',probability=p,labels=view_data['labels'],pair=view_data['pair'],
                scan=view_data['scan'],indices=view_data['indices'],available=view_data['available'])
            if view!='clean':del view_data
    meta=dict(candidate=candidate,fold=fold,seed=seed,threshold=chosen['threshold'],validation_objective=chosen['objective'],
        validation_changed_f1=chosen['changed_f1'],validation_by_view=by_view,
        robustness_validation_objective=float(np.mean([v['binary_macro_f1'] for v in by_view.values()])),
        best_epoch=state['epoch']+1,best_clean_validation_bce=state['best'],test_accessed=False,
        parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=digest(folder/'best.pt'),
        config_sha256=digest(folder/'config.json'),design_sha256=digest(out/'design.json'),
        timing_scope='Final worker chunk; per-epoch seconds in history',timing=timer.stats,
        training_device=device,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0.)
    write_json(folder/'frozen.json',meta);print('V3 frozen',json.dumps(meta),flush=True)


def select(out=DEFAULT,fold=0):
    out=Path(out);folder=fold_dir(out,fold)/'learning';path=folder/'selection.json'
    if path.exists():return
    candidates=[json.loads((folder/f'{c}_seed42/frozen.json').read_text()) for c in CANDIDATES]
    best=max(candidates,key=lambda x:(x['robustness_validation_objective'],-CANDIDATES.index(x['candidate'])))
    reference=json.loads((BASE/f'fold_{fold}/learning/selection.json').read_text())
    old=next(x for x in reference['candidates'] if x['variant']==reference['secondary_robust_variant'])
    gate=best['validation_objective']>=old['validation_objective']-.005 and best['robustness_validation_objective']>=old['robustness_validation_objective']+.01
    write_json(path,dict(candidate=best['candidate'],seed=42,fold=fold,candidates=candidates,pilot_gate_passed=bool(gate),
        baseline_clean_macro=old['validation_objective'],baseline_three_view_macro=old['robustness_validation_objective'],
        chosen_clean_macro=best['validation_objective'],chosen_three_view_macro=best['robustness_validation_objective'],
        selection='Three-view validation macro; candidate order tie break',test_accessed=False,design_sha256=digest(out/'design.json')))
    print('V3 selection',fold,best['candidate'],'pilot gate',gate,'clean',best['validation_objective'],'robust',best['robustness_validation_objective'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['sanity','train','select'])
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--candidate',choices=CANDIDATES,default=CANDIDATES[0]);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epoch-chunk',type=int,default=2);a=parser.parse_args()
    if a.stage=='sanity':fit_sanity(a.out)
    elif a.stage=='train':train(a.out,a.fold,a.candidate,a.seed,a.epoch_chunk)
    else:select(a.out,a.fold)
