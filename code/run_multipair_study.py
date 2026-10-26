"""Sequential, resumable experiment execution; no report or slide authoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/multipair_2026-10-25_v1'


def run(out=DEFAULT):
    out=Path(out);logs=out/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    if (out/'PROTOCOL_SUPERSEDED.json').exists():
        raise RuntimeError('This prototype is preserved. Use run_nested_multipair.py for the revised inner-validation protocol.')
    environment=dict(os.environ,LOKY_MAX_CPU_COUNT='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    stages=[('prepare',['multipair_data.py','prepare']),('geometry_calibration',['multipair_geometry.py']),
        ('training_features',['multipair_features.py']),('learning',['multipair_learning.py']),
        ('clean_spatial',['multipair_evaluate.py']),('clean_temporal',['multipair_evaluate.py','--protocol','temporal']),
        ('spatial_crossvalidation',['multipair_crossvalidate.py']),
        ('repeated_robustness',['multipair_evaluate.py','--stress']),
        ('analysis',['multipair_analysis.py']),('verification',['multipair_verify.py'])]
    for stage,args in stages:
        marker=logs/f'{stage}_complete.json'
        if marker.exists():continue
        command=[sys.executable,str(ROOT/'code'/args[0]),*args[1:],'--out',str(out)]
        (out/'execution_state.json').write_text(json.dumps(dict(status='running',stage=stage,command=command),indent=2))
        start=time.perf_counter();print('EXPERIMENT STAGE',stage,flush=True)
        with open(logs/f'{stage}.log','a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        if code:
            (out/'execution_state.json').write_text(json.dumps(dict(status='failed',stage=stage,exit_code=code),indent=2))
            raise RuntimeError(f'{stage} failed; checkpoints retained; see {logs/stage}')
        marker.write_text(json.dumps(dict(passed=True,seconds=time.perf_counter()-start)))
    (out/'execution_state.json').write_text(json.dumps(dict(status='complete',all_planned_stages_passed=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);run(p.parse_args().out)
