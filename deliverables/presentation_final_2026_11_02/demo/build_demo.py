"""Export frozen, aligned saved arrays only; never run research inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = ROOT / 'results/learning_v2_coverage_2026-10-25_v1'
OLD = ROOT / 'results/multipair_nested_2026-10-25_v1'
REG = ROOT / 'results/construction_change_regions_2026-10-26_v1'
SOURCES = {}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def source(p):
    SOURCES[str(p.relative_to(ROOT))] = digest(p)
    return p

def read(p):
    return json.loads(source(p).read_text())

def write(p, data):
    p.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')

def main():
    assets = HERE / 'assets'
    models = read(REG / 'model_manifest.json')
    design = read(REG / 'design.json')
    source(ROOT / 'code/demo.py')
    source(ROOT / 'code/construction_change_regions.py')
    source(ROOT / 'code/learning_v2_evaluate.py')
    choices = []
    for fold in range(5):
        r = V2 / f'fold_{fold}'
        s = read(r / 'learning/selection.json')
        name = s['primary_variant'] + '_seed42'
        frozen = read(r / f'learning/{name}/frozen.json')
        assert models[str(fold)]['model'] == name
        assert models[str(fold)]['checkpoint_sha256'] == frozen['checkpoint_sha256']
        choices.append(name)
    with source(REG / 'candidate_regions.csv').open(newline='') as f:
        regions = list(csv.DictReader(f))
    catalog = {'pairs': [], 'models': models, 'regionDesign': design,
               'notice': 'Saved sampled held-out predictions. No live inference, training or full-cloud rendering.',
               'reportingCheckout': '5477619 (reporting only; arrays from local results)'}
    conditions = ['clean_seed42', 'combined_large_seed42', 'retention_0.25_seed42']
    checks = []
    for pair in ['01', '02', '03', '12', '13', '23']:
        catalog['pairs'].append(pair)
        for cond in conditions:
            scans = [[], []]
            support = []
            for fold, name in enumerate(choices):
                r = V2 / f'fold_{fold}/evaluation/pair_{pair}' / cond
                if not (r / 'complete.json').exists():
                    assert cond != 'clean_seed42', f'Missing clean fold: {r}'
                    continue
                receipt = read(r / 'complete.json')
                p = source(r / 'predictions.npz')
                assert digest(p) == receipt['predictions_sha256']
                assert receipt['model_sha256'][name] == models[str(fold)]['checkpoint_sha256']
                condition = read(r / 'condition.json')
                support.append(condition)
                old = OLD if fold == 0 else OLD / f'spatial_cv/fold_{fold}'
                op = source(old / f'evaluation/primary/pair_{pair}/{cond}/predictions.npz')
                with np.load(p, allow_pickle=False) as z, np.load(op, allow_pickle=False) as oz:
                    for scan in [0, 1]:
                        for key in ['indices', 'xyz', 'labels']:
                            np.testing.assert_array_equal(z[f'{key}{scan}'], oz[f'{key}{scan}'])
                        xyz = z[f'xyz{scan}']
                        assert np.isfinite(xyz).all()
                        pred = z[f'{name}_prediction{scan}']
                        prob = z[f'{name}_probability{scan}']
                        valid = z[f'common_valid{scan}']
                        c2c = oz[f'c2c_calibrated_prediction{scan}']
                        for i in range(len(xyz)):
                            scans[scan].append([*map(float, xyz[i]), int(z[f'labels{scan}'][i]),
                                int(pred[i]), int(c2c[i]), int(valid[i]), float(prob[i]), fold,
                                int(z[f'indices{scan}'][i]), -2])
            if cond == 'clean_seed42':
                for scan, kind in enumerate(['removed', 'added']):
                    with np.load(source(REG / f'pair_{pair}_{kind}_cores.npz'), allow_pickle=False) as rz:
                        np.testing.assert_array_equal(np.array(scans[scan])[:, :3], rz['xyz'])
                        np.testing.assert_array_equal(np.array(scans[scan])[:, 4], rz['prediction'])
                        np.testing.assert_array_equal(np.array(scans[scan])[:, 9], rz['original_indices'])
                        for row, component in zip(scans[scan], rz['component_index']):
                            row[10] = int(component)
            data = {'pair': pair, 'condition': cond, 'scans': scans, 'support': support,
                    'schema': ['x','y','z','gt','v2','c2c','common_valid','model_score','fold','original_index','component'],
                    'regions': [r for r in regions if r['pair'] == pair] if cond == 'clean_seed42' else []}
            write(assets / f'{pair}_{cond}.json', data)
            checks.append({'pair': pair, 'condition': cond, 'counts': list(map(len, scans)), 'alignment': True})
    write(assets / 'catalog.json', catalog)
    write(assets / 'provenance.json', {'sources_sha256': SOURCES, 'checks': checks,
        'selection': 'Per-fold clean-validation-selected primary variant, seed42; may itself be robust or robust_hard.',
        'c2c': 'Saved c2c_calibrated_prediction arrays, aligned by original IDs, coordinates and labels.',
        'stress': 'Actual saved perturbed core coordinates and predictions; support counts from condition receipts. Full degraded support clouds are not exported.',
        'regions': 'Original clean candidate components only, verified against exported selected predictions; not recomputed for stress.'})
    manifest = {p.name: digest(p) for p in [HERE / 'index.html', HERE / 'style.css', HERE / 'viewer.js']}
    manifest.update({str(p.relative_to(HERE)).replace('\\', '/'): digest(p) for p in sorted(assets.glob('*.json'))})
    write(HERE / 'asset_manifest.json', manifest)
    print(f'Exported {len(checks)} views; verified {len(SOURCES)} source hashes and all alignments.')

if __name__ == '__main__':
    main()
