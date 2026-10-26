"""Execute fixed-outer-test, adequately supported nested spatial experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/multipair_nested_2026-10-25_v1'
PROTOTYPE=ROOT/'results/multipair_2026-10-25_v1'


def run(out=DEFAULT):
    out=Path(out);logs=out/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LOKY_MAX_CPU_COUNT='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    def stage(name,script,args=(),where=None):
        marker=logs/f'{name}_complete.json'
        if marker.exists():return
        command=[sys.executable,str(ROOT/'code'/script),*args,'--out',str(where or out)]
        (out/'execution_state.json').write_text(json.dumps(dict(status='running',stage=name,command=command),indent=2))
        start=time.perf_counter();print('NESTED STAGE',name,flush=True)
        with open(logs/f'{name}.log','a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        if code:
            (out/'execution_state.json').write_text(json.dumps(dict(status='failed',stage=name,exit_code=code),indent=2))
            raise RuntimeError(f'{name} failed; all checkpoints retained')
        marker.write_text(json.dumps(dict(passed=True,seconds=time.perf_counter()-start)))
    stage('prepare','multipair_nested.py')
    for fold in range(5):
        folder=out if fold==0 else out/f'spatial_cv/fold_{fold}'
        stage(f'fold{fold}_calibration','multipair_geometry.py',where=folder)
        stage(f'fold{fold}_features','multipair_features.py',where=folder)
        stage(f'fold{fold}_learning','multipair_learning.py',('--protocols','primary'),folder) if fold==0 else stage(
            f'fold{fold}_learning','multipair_crossvalidate.py',('--fit-fold',),folder)
        stage(f'fold{fold}_test','multipair_evaluate.py',where=folder)
    # Keep the already-frozen independent-epoch experiment exactly as it was.
    # Reusing its evidence does not turn it into a freshly untouched test.
    for relative in ['evaluation/temporal','learning/temporal']:
        src=PROTOTYPE/relative;dst=out/relative
        if not dst.exists():shutil.copytree(src,dst)
    shutil.copy2(PROTOTYPE/'temporal_geometry_frozen.json',out/'temporal_geometry_frozen.json')
    from m3c2_baseline import digest,write_json
    files={str(p.relative_to(PROTOTYPE)).replace('\\','/'):digest(p)
        for relative in ['evaluation/temporal','learning/temporal'] for p in (PROTOTYPE/relative).rglob('*') if p.is_file()}
    write_json(out/'temporal_reference_manifest.json',dict(source=str(PROTOTYPE),files=files,
        original_model_selection_pairs=['01'],original_test_pair='23',not_retrained_or_retuned=True))
    from multipair_operating_points import freeze_catalog
    freeze_catalog(out)
    stage('repeated_robustness','multipair_evaluate.py',('--stress',))
    stage('analysis','multipair_analysis.py')
    stage('verification','multipair_verify.py')
    (out/'execution_state.json').write_text(json.dumps(dict(status='complete',all_planned_stages_passed=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);run(p.parse_args().out)
