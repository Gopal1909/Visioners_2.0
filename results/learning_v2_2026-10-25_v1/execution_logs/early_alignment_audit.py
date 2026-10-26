"""Independent spot audit before test scoring; run from D:/CV_Project."""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData

root=Path('results/learning_v2_2026-10-25_v1')
audit=json.loads((root/'data_audit.json').read_text())
raw={(r['pair'],r['scan']):PlyData.read(r['path'],mmap='r')['params'].data
     for r in audit['clouds'] if r['pair'] in ('01','23')}
observations=0;probes=0
for fold in (0,1):
    for pair,group in [('01',0),('23',5)]:
        for split in ('train','validation'):
            with np.load(root/f'fold_{fold}/cache/pair_{pair}_{split}_clean.npz') as z:
                for scan in (0,1):
                    ids=z[f'indices{scan}'];record=raw[(pair,scan)]
                    np.testing.assert_array_equal(z[f'labels{scan}'],record['cd_type'][ids])
                    np.testing.assert_array_equal(z[f'xyz{scan}'],np.column_stack([record[k][ids] for k in ('x','y','z')]))
                    observations+=len(ids)
                    selected=np.random.default_rng(144+scan).choice(len(ids),8,replace=False)
                    offset=(2*group+scan)*len(ids)
                    for key in ('x','mask','geo','labels','indices','available'):
                        mapped=np.load(root/f'fold_{fold}/assembled/{split}_clean/{key}.npy',mmap_mode='r')
                        np.testing.assert_array_equal(mapped[offset+selected],z[f'{key}{scan}'][selected])
                        del mapped
                    probes+=len(selected)
result=dict(passed=True,folds=[0,1],pairs=['01','23'],original_gt_xyz_observations_checked=observations,
            independently_matched_feature_row_probes=probes,test_predictions_accessed=False)
(root/'early_alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
