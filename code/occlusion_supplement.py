"""All-scene geometric occlusion supplement using the retained pilot core indices.

The spatial test stripe lies outside the planned blind cylinders, so it cannot
measure a local deletion effect. This separate exploratory supplement does;
it is not presented as a learned spatial-holdout result or used for tuning.
"""
from pathlib import Path
import json
import numpy as np
from study_pipeline import ROOT,geometry
from m3c2_baseline import digest,write_json


def run():
    out=ROOT/'results/occlusion_supplement_2026-10-25_v1'
    out.mkdir(parents=True,exist_ok=True)
    pilot=ROOT/'results/pair_01/m3c2/pilot_2026-10-25_v2'
    if not (out/'splits.npz').exists():
        with np.load(pilot/'selected_cores.npz') as c:
            arrays={}
            for i in (0,1):
                arrays[f'test_indices{i}']=c[f'indices{i}']
                arrays[f'test_xyz{i}']=c[f'xyz{i}']
                arrays[f'test_gt{i}']=c[f'cd_type{i}']
        np.savez_compressed(out/'splits.npz',**arrays)
        frozen=json.loads((ROOT/'results/study_2026-10-25_v1/geometry_frozen.json').read_text())
        frozen.update(evaluation_scope='exploratory all-scene pilot cores; geometric-only occlusion supplement',
            split_hash=digest(out/'splits.npz'),selection='Reuse original pilot spatially balanced indices unchanged',
            rationale='Buffered test stripe is outside planned blind cylinders; add a separately labelled geometric footprint study rather than imply a no-effect result demonstrates robustness.')
        write_json(out/'geometry_frozen.json',frozen)
    geometry(out,['clean','occlusion_4.6','occlusion_7.6'])


if __name__=='__main__':run()
