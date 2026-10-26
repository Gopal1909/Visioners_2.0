"""Small runtime shared by memory-constrained GPU workers (no plotting/SciPy)."""
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import psutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/multipair_nested_2026-10-25_v1'
DEFAULT=ROOT/'results/learning_v2_coverage_2026-10-25_v1'
VARIANTS=('multiscale_clean','multiscale_robust','multiscale_robust_hard')
PAIRS=('01','02','03','12','13','23')
THRESHOLDS=np.r_[.01,.02,np.arange(.05,.951,.05).round(3),.98,.99]


def fold_dir(out,fold):return Path(out)/f'fold_{fold}'


def arrays(out,fold,split,view):
    folder=fold_dir(out,fold)/f'assembled/{split}_{view}'
    assert (folder/'complete.json').exists()
    return {key:np.load(folder/f'{key}.npy',mmap_mode='r') for key in ('x','mask','geo','labels','indices','available','pair','scan')}


def setup(seed=42):
    import torch
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        if rows:
            fields=list(dict.fromkeys(key for row in rows for key in row))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def memory_snapshot():
    info=psutil.Process().memory_info()
    result=dict(process_private_commit_mib=getattr(info,'private',info.vms)/2**20,process_rss_mib=info.rss/2**20)
    import os
    if os.name=='nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),*[(k,ctypes.c_ulonglong) for k in
                ('total_phys','avail_phys','total_pagefile','avail_pagefile','total_virtual','avail_virtual','avail_extended')]]
        s=Status();s.length=ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):result['system_commit_available_mib']=s.avail_pagefile/2**20
    return result


class MemoryTimer:
    def __enter__(self):
        self.process=psutil.Process();self.start_rss=self.peak=self.process.memory_info().rss
        self.stop=threading.Event();self.t0=time.perf_counter()
        def poll():
            while not self.stop.wait(.01):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();info=self.process.memory_info();self.peak=max(self.peak,info.rss)
        self.stats=dict(seconds=time.perf_counter()-self.t0,rss_start_mib=self.start_rss/2**20,
            sampled_peak_rss_mib=self.peak/2**20,sampled_rss_growth_mib=max(0,self.peak-self.start_rss)/2**20,
            process_lifetime_peak_mib=getattr(info,'peak_wset',0)/2**20)
