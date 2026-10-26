"""Execute the predeclared five spatial folds with independent model selection.

Fold 0 is the main run. Folds 1-4 use one training seed to control compute; the
main run separately repeats its selected architecture with three training seeds.
All stages run sequentially in separate processes to release laptop memory.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/multipair_2026-10-25_v1'


def fit_fold(out):
    from multipair_learning import neural,boosted,VARIANTS
    from multipair_data import PAIRS
    from m3c2_baseline import write_json,digest
    folder=out/'learning/primary';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection_frozen.json').exists():return
    candidates=[neural(out,'primary',variant,42,PAIRS) for variant in VARIANTS]
    selected=max(candidates,key=lambda c:(c['validation_objective'],-VARIANTS.index(c['variant'])))
    tabular=boosted(out,'primary',PAIRS)
    write_json(folder/'selection_frozen.json',dict(protocol='primary',pairs=list(PAIRS),candidates=candidates,
        selected_variant=selected['variant'],seeds=[42],repeated_models=[selected],geometry_classifier=tabular,
        threshold_policy='Independent fold validation only; no tuning from any fold test outcomes',
        test_accessed=False,protocol_sha256=digest(out/'protocol.json')))


def run(out=DEFAULT,folds=(1,2,3,4)):
    design=json.loads((out/'crossvalidation_design.json').read_text())
    original=json.loads((out/'protocol.json').read_text())
    for number in folds:
        if number==0:continue
        config=next(f for f in design['folds'] if f['fold']==number)
        folder=out/f'spatial_cv/fold_{number}';folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'protocol.json').exists():
            protocol=dict(original,cells=config['cells'],spatial_fold_id=number,
                parent_protocol_sha256=design['protocol_sha256'])
            (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
            shutil.copy2(out/'data_audit.json',folder/'data_audit.json')
        stages=[('prepare',[str(ROOT/'code/multipair_data.py'),'prepare','--out',str(folder)]),
            ('geometry',[str(ROOT/'code/multipair_geometry.py'),'--out',str(folder)]),
            ('features',[str(ROOT/'code/multipair_features.py'),'--out',str(folder)]),
            ('learning',[str(Path(__file__).resolve()),'--fit-fold','--out',str(folder)]),
            ('clean_test',[str(ROOT/'code/multipair_evaluate.py'),'--out',str(folder)])]
        for stage,args in stages:
            marker=folder/f'pipeline_{stage}_complete.json'
            if marker.exists():continue
            print('CROSS-VALIDATION FOLD',number,'STAGE',stage,flush=True)
            start=time.perf_counter()
            with open(folder/f'{stage}_console.log','a',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                for line in process.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
            if code:
                (folder/'pipeline_error.json').write_text(json.dumps(dict(stage=stage,exit_code=code)))
                raise RuntimeError(f'Fold {number} {stage} failed; completed stages preserved')
            marker.write_text(json.dumps(dict(passed=True,seconds=time.perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4]);p.add_argument('--fit-fold',action='store_true');a=p.parse_args()
    fit_fold(a.out) if a.fit_fold else run(a.out,a.folds)
