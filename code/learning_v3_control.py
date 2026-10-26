"""Standard-library-only controller: no numerical runtimes resident in parent."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/learning_v2_coverage_2026-10-25_v1'
DEFAULT=ROOT/'results/learning_v3_construction_2026-10-25_v1'
PAIRS=('01','02','03','12','13','23')
CANDIDATES=('pool_fresh_bce','graph_fresh_bce','graph_fresh_focal')
EPOCHS=16
CORES=1500


def fold_dir(out,fold):return Path(out)/f'fold_{fold}'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def inference_cli():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--fold',type=int,default=0);p.add_argument('--pair',default='01');p.add_argument('--stress',action='store_true');a=p.parse_args()
    src=BASE/f'fold_{a.fold}/evaluation/pair_{a.pair}';dst=fold_dir(a.out,a.fold)/f'evaluation/pair_{a.pair}'
    def pending():return [s for s in src.iterdir() if s.is_dir() and ((s.name!='clean_seed42') if a.stress else s.name=='clean_seed42') and not (dst/s.name/'complete.json').exists()]
    while pending():
        command=[sys.executable,str(ROOT/'code/learning_v3_evaluate.py'),'inference','--out',str(a.out),'--fold',str(a.fold),'--pair',a.pair,'--worker']
        if a.stress:command.append('--stress')
        before=len(pending());subprocess.run(command,check=True);assert len(pending())<before
