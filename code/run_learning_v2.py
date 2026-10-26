"""Resume-safe sequential multiscale learning experiment; results only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from learning_v2_runtime import ROOT,DEFAULT,PAIRS,VARIANTS,fold_dir,digest,write_json


def run(out=DEFAULT):
    out=Path(out);logs=out/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    if (out/'PROTOCOL_SUPERSEDED.json').exists():raise RuntimeError('Preserved development attempt is superseded; use the coverage study')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LOKY_MAX_CPU_COUNT='2',PYTHONIOENCODING='utf-8')
    def stage(name,script,args=()):
        marker=logs/f'{name}_complete.json'
        if marker.exists():return
        args=list(map(str,args));training=script=='learning_v2_model.py' and args[0]=='train'
        epoch_chunk=3;failures=0
        if training:
            fold=int(args[args.index('--fold')+1]);variant=args[args.index('--variant')+1]
            seed=int(args[args.index('--seed')+1]) if '--seed' in args else 42
            frozen=fold_dir(out,fold)/f'learning/{variant}_seed{seed}/frozen.json'
        command=[sys.executable,str(ROOT/'code'/script),*args,'--out',str(out)]
        write_json(out/'execution_state.json',dict(status='running',stage=name,command=command))
        print('V2 STAGE',name,flush=True);start=time.perf_counter()
        with open(logs/f'{name}.log','a',encoding='utf-8') as log:
            while True:
                actual=command+(['--epoch-chunk',str(epoch_chunk)] if training else [])
                process=subprocess.Popen(actual,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
                if code and training and failures<2:
                    failures+=1;epoch_chunk=1
                    print('Retry with one-epoch worker; last complete checkpoint retained',flush=True);continue
                if code or not training or frozen.exists():break
        if code:
            write_json(out/'execution_state.json',dict(status='failed',stage=name,exit_code=code))
            raise RuntimeError(f'{name} failed; completed caches and checkpoints retained')
        write_json(marker,dict(passed=True,seconds=time.perf_counter()-start,script_sha256=digest(ROOT/'code'/script)))
    stage('prepare','learning_v2_data.py',['prepare'])
    for fold in range(5):
        for pair in PAIRS:
            for split in ('train','validation'):
                for view in ('clean','mild','medium'):
                    stage(f'cache_f{fold}_p{pair}_{split}_{view}','learning_v2_data.py',
                        ['cache','--fold',fold,'--pair',pair,'--split',split,'--view',view])
                    if fold==0 and pair=='01' and split=='train' and view=='clean':
                        stage('measured_patch_coverage','learning_v2_data.py',['check-coverage'])
        for split in ('train','validation'):
            for view in ('clean','mild','medium'):
                stage(f'assemble_f{fold}_{split}_{view}','learning_v2_data.py',['assemble','--fold',fold,'--split',split,'--view',view])
        for variant in VARIANTS:
            stage(f'train_f{fold}_{variant}_s42','learning_v2_model.py',['train','--fold',fold,'--variant',variant])
        stage(f'boosted_f{fold}','learning_v2_model.py',['boosted','--fold',fold])
        stage(f'select_f{fold}','learning_v2_model.py',['select','--fold',fold])
        if fold==0:
            choice=json.loads((fold_dir(out,fold)/'learning/selection.json').read_text())['primary_variant']
            for seed in (43,44):
                stage(f'train_f0_{choice}_s{seed}','learning_v2_model.py',['train','--fold',0,'--variant',choice,'--seed',seed])
    # No new test predictions are generated until every fold choice is frozen.
    for fold in range(5):
        for pair in PAIRS:
            stage(f'clean_features_f{fold}_p{pair}','learning_v2_evaluate.py',['features','--fold',fold,'--pair',pair])
        for pair in PAIRS:
            stage(f'clean_inference_f{fold}_p{pair}','learning_v2_evaluate.py',['inference','--fold',fold,'--pair',pair])
    for pair in PAIRS:
        stage(f'stress_features_p{pair}','learning_v2_evaluate.py',['features','--fold',0,'--pair',pair,'--stress'])
    for pair in PAIRS:
        stage(f'stress_inference_p{pair}','learning_v2_evaluate.py',['inference','--fold',0,'--pair',pair,'--stress'])
    stage('analysis','learning_v2_evaluate.py',['analyse'])
    stage('verification','learning_v2_evaluate.py',['verify'])
    # Optional external data are never read until every model and threshold is
    # frozen. Re-running this driver after data arrive skips completed work.
    external_plan=out/'external_test_plan.json';external_passed=False
    if external_plan.exists():
        plan=json.loads(external_plan.read_text());scene=Path(plan.get('expected_folder','__not_supplied__'))
        if scene.is_dir() and all(len(list(scene.rglob(f'pointCloud{i}.ply')))==1 for i in (0,1)):
            stage('independent_scene_features','learning_v2_external.py',['cpu'])
            stage('independent_scene_inference','learning_v2_external.py',['inference'])
            stage('independent_scene_verification','learning_v2_external.py',['verify'])
            external_passed=True
    write_json(out/'complete_evidence.json',dict(passed=True,primary_matrix_sha256=digest(out/'verification.json'),
        independent_scene_evaluated=external_passed,
        independent_verification_sha256=digest(out/'external_open_space/verification.json') if external_passed else None,
        reports_and_presentations_untouched=True))
    source=logs/'final_sources';source.mkdir(exist_ok=True);manifest={}
    names=sorted({p.name for p in (ROOT/'code').glob('learning_v2*.py')}|{'run_learning_v2.py','test_learning_v2.py'})
    for name in names:
        shutil.copy2(ROOT/'code'/name,source/name);manifest[name]=digest(source/name)
    write_json(source/'manifest.json',manifest)
    write_json(out/'execution_state.json',dict(status='complete',all_planned_stages_passed=True))
    index=ROOT/'results/active_experiment.json'
    write_json(index,dict(study=str(out.relative_to(ROOT)),status='complete',state=str((out/'execution_state.json').relative_to(ROOT)),
        resume_script='code/run_learning_v2.py',previous_study='results/multipair_nested_2026-10-25_v1',
        report_and_presentation_authoring=False,verification=str((out/'complete_evidence.json').relative_to(ROOT)),
        independent_scene_evaluated=external_passed))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);run(p.parse_args().out)
