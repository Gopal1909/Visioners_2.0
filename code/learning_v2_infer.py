"""GPU-only prediction path: no full-cloud or plotting modules are imported."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import sys
import numpy as np
from learning_v2_runtime import BASE,DEFAULT,ROOT,VARIANTS,fold_dir,digest,write_json,write_csv,setup
from learning_v2_scores import metric_rows


def load_models(out,fold):
    import torch,joblib
    from learning_v2_model import MetricFusion
    folder=fold_dir(out,fold)/'learning';selection=json.loads((folder/'selection.json').read_text())
    names=[v+'_seed42' for v in VARIANTS]
    if fold==0:names.extend(selection['primary_variant']+f'_seed{s}' for s in (43,44))
    models={}
    for name in names:
        meta=json.loads((folder/name/'frozen.json').read_text());path=folder/name/'best.pt'
        assert digest(path)==meta['checkpoint_sha256']
        model=MetricFusion().cuda();state=torch.load(path,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);model.eval();del state
        models[name]=(model,meta['threshold'],meta['checkpoint_sha256'])
    geo=folder/'bounded_geometry';meta=json.loads((geo/'frozen.json').read_text())
    assert digest(geo/'model.joblib')==meta['checkpoint_sha256']
    models['bounded_geometry']=(joblib.load(geo/'model.joblib'),meta['threshold'],meta['checkpoint_sha256'])
    return models


def inference(out=DEFAULT,fold=0,pair='01',stress=False,max_cases=8):
    from learning_v2_model import probabilities
    out=Path(out);root=fold_dir(out,fold);base=BASE if fold==0 else BASE/f'spatial_cv/fold_{fold}'
    folders=[p for p in sorted((root/f'evaluation/pair_{pair}').iterdir()) if p.is_dir() and
        ((p.name!='clean_seed42') if stress else p.name=='clean_seed42') and not (p/'complete.json').exists()]
    if not folders:return
    setup(42);models=load_models(out,fold)
    for dst in folders[:max_cases]:
        receipt=json.loads((dst/'feature_complete.json').read_text());assert digest(dst/'features.npz')==receipt['sha256']
        with np.load(dst/'features.npz') as z:saved={k:z[k] for k in z.files}
        labels=[saved[f'labels{i}'] for i in (0,1)];prediction={};probability={};timing={}
        condition=json.loads((dst/'condition.json').read_text())
        for name,(model,threshold,_) in models.items():
            started=time.perf_counter();p=[]
            for scan in (0,1):
                f={k:saved[f'{k}{scan}'] for k in ('x','mask','geo','labels')}
                p.append(model.predict_proba(f['geo'])[:,1] if name=='bounded_geometry' else probabilities(model,f))
            probability[name]=p;prediction[name]=[(a>=threshold).astype(np.int8) for a in p];timing[name]=time.perf_counter()-started
        old=base/f'evaluation/primary/pair_{pair}'/dst.name
        with np.load(old/'predictions.npz') as z:common=[z[f'm3c2_calibrated_prediction{i}']>=0 for i in (0,1)]
        rows=[];name,seed=dst.name.rsplit('_seed',1)
        for visibility in ('A','B') if condition['kind']=='occlusion' else ('standard',):
            masks=[np.ones(len(y),bool) for y in labels]
            if visibility=='B':masks[0]=np.linalg.norm(saved['xyz0'][:,:2]-np.array(condition['blind_center_m'])[:2],axis=1)>condition['radius_m']
            metadata=dict(experiment='learning_v2',spatial_fold=fold,pair=pair,condition=name,family=condition['kind'],
                severity=condition['severity'],perturbation_seed=int(seed),visibility=visibility)
            rows.extend(metric_rows(labels,prediction,common,metadata,masks,condition['target_requested_cores'],condition.get('requested_label_counts')))
        result={k:v for k,v in saved.items() if not k.startswith(('x0','x1','mask','geo'))}
        for scan in (0,1):
            result[f'common_valid{scan}']=common[scan]
            for method in prediction:
                result[f'{method}_prediction{scan}']=prediction[method][scan];result[f'{method}_probability{scan}']=probability[method][scan]
        np.savez_compressed(dst/'predictions.npz',**result);write_csv(dst/'metrics.csv',rows)
        write_json(dst/'complete.json',dict(passed=True,predictions_sha256=digest(dst/'predictions.npz'),
            model_sha256={k:m[2] for k,m in models.items()},feature_sha256=receipt['sha256'],
            selection_sha256=digest(root/'learning/selection.json'),no_test_retuning=True,inference_seconds=timing,
            gpu_worker_sha256=digest(__file__),scoring_kernel_sha256=digest(ROOT/'code/learning_v2_scores.py')))
        print('V2 inference',fold,pair,dst.name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inference'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);p.add_argument('--pair',default='01');p.add_argument('--stress',action='store_true')
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:inference(a.out,a.fold,a.pair,a.stress)
    else:
        root=fold_dir(a.out,a.fold)/f'evaluation/pair_{a.pair}'
        def pending():return [p for p in root.iterdir() if p.is_dir() and
            ((p.name!='clean_seed42') if a.stress else p.name=='clean_seed42') and not (p/'complete.json').exists()]
        while pending():
            command=[sys.executable,str(Path(__file__).resolve()),'inference','--out',str(a.out),'--fold',str(a.fold),'--pair',a.pair,'--worker']
            if a.stress:command.append('--stress')
            before=len(pending());subprocess.run(command,check=True);assert len(pending())<before,'Inference worker made no progress'


if __name__=='__main__':main()
