"""Resume-safe construction-only pilot; expand only after its validation gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from learning_v3_control import ROOT,BASE,DEFAULT,PAIRS,CANDIDATES,EPOCHS,fold_dir,digest,write_json


def run(out=DEFAULT):
    out=Path(out);logs=out/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    def stage(name,script,args=(),training=False):
        marker=logs/f'{name}_complete.json'
        if marker.exists():return
        caching=script=='learning_v3_data.py' and args[0]=='cache'
        if caching:
            a=list(map(str,args));cache_fold=int(a[a.index('--fold')+1]);cache_pair=a[a.index('--pair')+1]
        command=[sys.executable,str(ROOT/'code'/script),*map(str,args),'--out',str(out)]
        write_json(out/'execution_state.json',dict(status='running',stage=name,command=command));print('V3 STAGE',name,flush=True)
        start=time.perf_counter()
        with open(logs/f'{name}.log','a',encoding='utf-8') as log:
            while True:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait()
                if code:
                    write_json(out/'execution_state.json',dict(status='failed',stage=name,exit_code=code));raise RuntimeError(f'{name} failed; completed work retained')
                if caching:
                    if len(list((fold_dir(out,cache_fold)/'epochs').glob(f'epoch_*/pair_{cache_pair}.json')))==EPOCHS:break
                    continue
                if not training:break
                a=list(map(str,args));fold=int(a[a.index('--fold')+1]);candidate=a[a.index('--candidate')+1];seed=int(a[a.index('--seed')+1]) if '--seed' in a else 42
                if (fold_dir(out,fold)/f'learning/{candidate}_seed{seed}/frozen.json').exists():break
        write_json(marker,dict(passed=True,seconds=time.perf_counter()-start,source_sha256=digest(ROOT/'code'/script)))
    stage('sanity','learning_v3_model.py',['sanity'])
    stage('prepare','learning_v3_data.py',['prepare'])
    write_json(ROOT/'results/active_experiment.json',dict(study=str(out.relative_to(ROOT)),status='running',previous_study=str(BASE.relative_to(ROOT)),
        resume_script='code/run_learning_v3.py',scope='Construction-only; no Office/car-parking data used',state=str((out/'execution_state.json').relative_to(ROOT)),report_and_presentation_authoring=False))
    for fold in range(5):
        for pair in PAIRS:stage(f'fresh_f{fold}_p{pair}','learning_v3_data.py',['cache','--fold',fold,'--pair',pair])
        stage(f'assemble_f{fold}','learning_v3_data.py',['assemble','--fold',fold])
        stage(f'fresh_audit_f{fold}','learning_v3_audit.py',['features','--fold',fold])
        for candidate in CANDIDATES:stage(f'train_f{fold}_{candidate}_s42','learning_v3_model.py',['train','--fold',fold,'--candidate',candidate],training=True)
        stage(f'select_f{fold}','learning_v3_model.py',['select','--fold',fold])
        if fold==0:
            selection=json.loads((fold_dir(out,0)/'learning/selection.json').read_text())
            if not selection['pilot_gate_passed']:
                stage('pilot_analysis','learning_v3_audit.py',['pilot'])
                finish(out,logs,'pilot_complete_gate_not_met');return
    choice=json.loads((fold_dir(out,0)/'learning/selection.json').read_text())['candidate']
    for seed in (43,44):stage(f'repeat_f0_s{seed}','learning_v3_model.py',['train','--fold',0,'--candidate',choice,'--seed',seed],training=True)
    for fold in range(5):
        for pair in PAIRS:stage(f'clean_f{fold}_p{pair}','learning_v3_evaluate.py',['inference','--fold',fold,'--pair',pair])
    for pair in PAIRS:stage(f'stress_p{pair}','learning_v3_evaluate.py',['inference','--fold',0,'--pair',pair,'--stress'])
    stage('analysis','learning_v3_evaluate.py',['analyse'])
    stage('verification','learning_v3_audit.py',['verify'])
    finish(out,logs,'complete')


def finish(out,logs,status):
    source=logs/'final_sources';source.mkdir(exist_ok=True);manifest={}
    for path in sorted((ROOT/'code').glob('*learning_v3*.py')):
        shutil.copy2(path,source/path.name);manifest[path.name]=digest(source/path.name)
    write_json(source/'manifest.json',manifest)
    write_json(out/'execution_state.json',dict(status=status,pilot_completed=True,full_matrix_completed=status=='complete'))
    index=json.loads((ROOT/'results/active_experiment.json').read_text());index['status']=status;index['verification']=str((out/('verification.json' if status=='complete' else 'pilot_verification.json')).relative_to(ROOT));write_json(ROOT/'results/active_experiment.json',index)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);run(p.parse_args().out)
