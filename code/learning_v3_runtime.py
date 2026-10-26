"""Construction-focused follow-up; lightweight runtime for separate CPU/GPU jobs."""
from pathlib import Path
from learning_v2_runtime import (THRESHOLDS,setup,digest,write_json,
    write_csv,MemoryTimer,memory_snapshot)

from learning_v3_control import ROOT,BASE,DEFAULT,PAIRS,CANDIDATES,EPOCHS,CORES


def fold_dir(out,fold):return Path(out)/f'fold_{fold}'


def bank(out,fold,epoch):
    import numpy as np
    folder=fold_dir(out,fold)/f'epochs/epoch_{epoch:02d}/assembled'
    assert (folder/'complete.json').exists()
    return {k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in
        ('x','mask','geo','labels','pair','scan','indices','available')}


def validation(fold,view):
    from learning_v2_runtime import arrays
    return arrays(BASE,fold,'validation',view)
