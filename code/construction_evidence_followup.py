"""Bounded validation-only test of geometric corroboration and model agreement.

All inputs are existing frozen v2 outputs. No training, new neighbourhoods,
support expansion, pair-specific thresholds, or changes to evaluation masks.
Test application is forbidden until an immutable validation decision exists.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from learning_v3_control import ROOT, BASE, digest, write_json
from learning_v2_runtime import write_csv
from multipair_metrics import binary_scores


DEFAULT = ROOT / 'results/construction_evidence_2026-10-26_v1'
VIEWS = ('clean', 'mild', 'medium')
RULES = ('primary', 'robust', 'agreement', 'mean_margin')
MIN_DISTANCES = (0., .01, .02, .04)
FIELDS = ('labels', 'indices', 'available', 'pair', 'scan')


def candidate_grid():
    return [dict(name=f'{rule}_d{round(distance*1000):03d}mm', rule=rule,
                 minimum_distance_m=distance, order=order)
            for order, (rule, distance) in enumerate(
                (r, d) for r in RULES for d in MIN_DISTANCES)]


def corroborated_prediction(primary, robust, primary_threshold, robust_threshold,
                            distance_m, rule='primary', minimum_distance_m=0.):
    """Decision-only operation: no labels, coordinates, or neighbouring cores."""
    primary = np.asarray(primary, dtype=np.float64)
    robust = np.asarray(robust, dtype=np.float64)
    distance_m = np.asarray(distance_m)
    assert primary.shape == robust.shape == distance_m.shape
    assert np.isfinite(primary).all() and np.isfinite(robust).all()
    assert ((primary >= 0) & (primary <= 1)).all()
    assert ((robust >= 0) & (robust <= 1)).all()
    assert np.isfinite(distance_m).all() and (distance_m >= 0).all()
    assert 0 < primary_threshold < 1 and 0 < robust_threshold < 1
    assert 0 <= minimum_distance_m <= .25
    a = primary >= primary_threshold
    b = robust >= robust_threshold
    if rule == 'primary':
        positive = a
    elif rule == 'robust':
        positive = b
    elif rule == 'agreement':
        positive = a & b
    elif rule == 'mean_margin':
        def logit(p):
            p = np.clip(p, 1e-7, 1-1e-7)
            return np.log(p) - np.log1p(-p)
        # Normalize each score by its OWN previously frozen decision threshold.
        positive = (logit(primary) - logit(primary_threshold)
                    + logit(robust) - logit(robust_threshold)) >= 0
    else:
        raise ValueError(rule)
    return (positive & (distance_m >= minimum_distance_m)).astype(np.int8)


def summarize(data, prediction):
    keep = np.asarray(data['available'], bool)
    groups = np.asarray(data['pair']) * 2 + data['scan']
    rows = []
    for code in np.unique(groups):
        mask = keep & (groups == code)
        row = binary_scores(data['labels'][mask], prediction[mask])
        row.update(pair=f'{int(code)//2:02d}', scan=int(code)%2)
        rows.append(row)
    totals = {key: sum(r[key] for r in rows) for key in ('tp','fp','fn','tn')}
    tp, fp, fn, tn = (totals[k] for k in ('tp','fp','fn','tn'))
    result = dict(binary_macro_f1=float(np.mean([r['binary_macro_f1'] for r in rows])),
                  changed_f1=float(np.mean([r['f1_changed'] for r in rows])),
                  changed_precision=tp/max(1,tp+fp), changed_recall=tp/max(1,tp+fn),
                  false_positive_rate=fp/max(1,fp+tn), available_cores=int(keep.sum()), **totals)
    return result, rows


def summarize_views(scores):
    result = {f'{view}_{key}': value for view, row in scores.items()
              for key, value in row.items()}
    for key in ('binary_macro_f1','changed_f1','changed_precision','changed_recall','false_positive_rate'):
        result['mean_'+key] = float(np.mean([scores[v][key] for v in VIEWS]))
    return result


def eligible(row, reference, lane):
    """Predeclared gains and precision/recall guards, applied before ranking."""
    if lane == 'clean':
        return (row['clean_binary_macro_f1'] >= reference['clean_binary_macro_f1']+.01-1e-12
                and row['clean_changed_f1'] >= reference['clean_changed_f1']-1e-12
                and row['clean_false_positive_rate'] <= reference['clean_false_positive_rate']+1e-12
                and row['clean_changed_recall'] >= reference['clean_changed_recall']-.02-1e-12
                and row['mean_binary_macro_f1'] >= reference['mean_binary_macro_f1']-.005-1e-12)
    if lane == 'robust':
        return (row['mean_binary_macro_f1'] >= reference['mean_binary_macro_f1']+.01-1e-12
                and row['mean_changed_f1'] >= reference['mean_changed_f1']-1e-12
                and row['clean_binary_macro_f1'] >= reference['clean_binary_macro_f1']-.005-1e-12
                and row['clean_false_positive_rate'] <= reference['clean_false_positive_rate']+.005+1e-12
                and row['mean_changed_recall'] >= reference['mean_changed_recall']-.02-1e-12)
    raise ValueError(lane)


def choose(rows, reference, lane):
    valid = [r for r in rows if eligible(r, reference, lane)]
    if not valid:
        return None
    prefix = 'clean_' if lane == 'clean' else 'mean_'
    return max(valid, key=lambda r: (r[prefix+'binary_macro_f1'], r[prefix+'changed_f1'],
                                    -r['clean_false_positive_rate'], -r['order']))


def input_paths():
    for fold in range(5):
        folder = BASE / f'fold_{fold}'
        yield folder / 'protocol.json'
        selection_path = folder / 'learning/selection.json'
        yield selection_path
        selection = json.loads(selection_path.read_text())
        for name in set(selection[k] for k in ('primary_variant','secondary_robust_variant')):
            source = folder / 'learning' / (name+'_seed42')
            yield source / 'frozen.json'
            yield source / 'best.pt'
            for view in VIEWS:
                yield source / f'validation_{view}.npz'
        for view in VIEWS:
            source = folder / f'assembled/validation_{view}'
            for field in (*FIELDS, 'geo'):
                yield source / f'{field}.npy'


def prepare(out=DEFAULT):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    if (out/'design.json').exists():
        return
    # Freeze all choices before looking at any new candidate scores.
    sources = {str(p): digest(p) for p in sorted(set(input_paths()))}
    protected = {str(p): digest(p) for p in (ROOT/'code').glob('*.py')}
    for root in (BASE, ROOT/'results/learning_v3_construction_2026-10-25_v1',
                 ROOT/'results/operating_point_followup_2026-10-26_v1',
                 ROOT/'results/construction_curriculum_refinement_2026-10-26_v1',
                 ROOT/'results/construction_change_regions_2026-10-26_v1'):
        for name in ('final_review.json','verification.json','frozen.json','execution_state.json'):
            path = root/name
            if path.exists():
                protected[str(path)] = digest(path)
    write_json(out/'input_manifest.json', sources)
    write_json(out/'protected_manifest.json', protected)
    snapshot = out/'sources'
    snapshot.mkdir(exist_ok=True)
    for name in ('construction_evidence_followup.py','learning_v2_runtime.py',
                 'learning_v3_control.py','multipair_metrics.py'):
        shutil.copy2(ROOT/'code'/name, snapshot/name)
    write_json(out/'design.json', dict(
        purpose='Reduce coherent false positives with bounded geometric corroboration and frozen-model agreement',
        source_study=str(BASE), candidates=candidate_grid(), folds=list(range(5)), views=list(VIEWS),
        distance='Original core-to-opposite-full-scan NN distance capped at .25 m, geo[:,50]*.25; repeated columns 105 and 160 must agree',
        frozen_networks='Per-fold original seed42 primary and secondary robust best checkpoints and original thresholds; no new fitting',
        ensemble='Positive only if both frozen decisions agree, or mean of threshold-centered logit margins >=0',
        selection='One global rule/distance per lane across all five validation folds and six pairs; no fold/pair-specific changes',
        clean_gate='Clean mean-fold macro +.01; changed-F1 noninferior; FPR nonincreasing; recall loss <=.02; three-view macro loss <=.005 relative to original primary',
        robust_gate='Three-view mean-fold macro +.01; changed-F1 noninferior; clean macro loss <=.005; clean FPR rise <=.005; three-view recall loss <=.02 relative to original secondary robust',
        tie_break='Lane macro, lane changed-F1, lower clean FPR, original candidate order',
        exposure='Previously inspected same-site held-out errors motivated this hypothesis. This is follow-up development, not an untouched benchmark.',
        labels='Validation labels used for selection. Previously viewed test errors are descriptive only. No pair-specific exceptions.',
        masks='Original availability and original IDs/labels retained; no abstention added and no points excluded',
        support='No new neighbourhoods or support dependencies, so original buffered-split support guard remains applicable',
        heldout_policy='Freeze all selections first. Apply only lanes passing their gates. Reuse saved predictions/features; do not run neural inference or regenerate geometry.',
        test_access_before_freeze=False, input_manifest_sha256=digest(out/'input_manifest.json'),
        source_sha256=digest(__file__), reports_or_presentations=False,
        data='BikeParkingConstruction only; no downloads or additional data required'))
    write_json(out/'execution_state.json', dict(stage='prepared', complete=False))
    print('Frozen evidence-follow-up design: 16 candidates, five validation folds', flush=True)


def validation_inputs(fold, view):
    folder = BASE / f'fold_{fold}'
    selection = json.loads((folder/'learning/selection.json').read_text())
    data = {}
    thresholds = {}
    for kind, key in (('primary','primary_variant'),('robust','secondary_robust_variant')):
        source = folder/'learning'/(selection[key]+'_seed42')
        meta = json.loads((source/'frozen.json').read_text())
        assert digest(source/'best.pt') == meta['checkpoint_sha256']
        thresholds[kind] = meta['threshold']
        with np.load(source/f'validation_{view}.npz') as z:
            if not data:
                data = {field: z[field] for field in FIELDS}
            else:
                for field in FIELDS:
                    np.testing.assert_array_equal(data[field], z[field])
            data[kind] = z['probability']
    assembled = folder/f'assembled/validation_{view}'
    for field in FIELDS:
        np.testing.assert_array_equal(data[field], np.load(assembled/f'{field}.npy', mmap_mode='r'))
    geo = np.load(assembled/'geo.npy', mmap_mode='r')
    assert geo.shape == (len(data['labels']),165)
    for offset in (55,110):
        np.testing.assert_array_equal(geo[:,50], geo[:,offset+50])
        np.testing.assert_array_equal(geo[:,54], geo[:,offset+54])
    data['distance_m'] = np.asarray(geo[:,50], np.float64)*.25
    data['has_near_match'] = np.asarray(geo[:,54], bool)
    data['source_log_density'] = np.asarray(geo[:,48])
    data['opposite_log_density'] = np.asarray(geo[:,49])
    return data, thresholds


def predict(data, thresholds, spec):
    return corroborated_prediction(data['primary'], data['robust'], thresholds['primary'],
                                   thresholds['robust'], data['distance_m'], spec['rule'], spec['minimum_distance_m'])


def error_slices(data, prediction, fold, view, model):
    """Disjoint geometric bins of validation errors, without claiming their cause."""
    bins = (0.,.01,.02,.04,.08,.25,np.inf)
    rows = []
    for pair in np.unique(data['pair']):
        for scan in (0,1):
            support = data['available'] & (data['pair']==pair) & (data['scan']==scan)
            for truth, estimate, outcome in ((0,0,'TN'),(0,1,'FP'),(1,0,'FN'),(1,1,'TP')):
                group = support & (data['labels']==truth) & (prediction==estimate)
                for lo,hi in zip(bins[:-1],bins[1:]):
                    mask = group & (data['distance_m']>=lo) & (data['distance_m']<hi)
                    count = int(mask.sum())
                    rows.append(dict(fold=fold,view=view,model=model,pair=f'{int(pair):02d}',scan=scan,outcome=outcome,
                        distance_bin=f'{lo:g} to {hi:g} m',count=count,core_match_within_25cm=int(data['has_near_match'][mask].sum()),
                        mean_raw_model_score=float(data[model][mask].mean()) if count else None,
                        mean_source_log_density=float(data['source_log_density'][mask].mean()) if count else None,
                        mean_opposite_log_density=float(data['opposite_log_density'][mask].mean()) if count else None))
    return rows


def validate(out=DEFAULT):
    out = Path(out)
    if (out/'frozen_selection.json').exists():
        print('Validation already frozen; not repeating', flush=True)
        return
    plan = json.loads((out/'design.json').read_text())
    assert plan['candidates'] == candidate_grid() and plan['source_sha256'] == digest(__file__)
    started = time.perf_counter()
    fold_rows, direction_rows, slices = [], [], []
    for fold in plan['folds']:
        scores = {spec['name']: {} for spec in plan['candidates']}
        for view in VIEWS:
            data, thresholds = validation_inputs(fold,view)
            for spec in plan['candidates']:
                p = predict(data,thresholds,spec)
                metrics, directions = summarize(data,p)
                scores[spec['name']][view] = metrics
                direction_rows.extend(dict(fold=fold,view=view,candidate=spec['name'],**r) for r in directions)
                if spec['minimum_distance_m']==0 and spec['rule'] in ('primary','robust'):
                    slices.extend(error_slices(data,p,fold,view,spec['rule']))
            print('Evidence validation',fold,view,flush=True)
        for spec in plan['candidates']:
            fold_rows.append(dict(fold=fold,**spec,**summarize_views(scores[spec['name']])))
    rows = []
    for spec in plan['candidates']:
        selected = [r for r in fold_rows if r['name']==spec['name']]
        columns = [k for k in selected[0] if k not in ('fold',*spec)]
        rows.append(dict(**spec,**{k:float(np.mean([r[k] for r in selected])) for k in columns}))
    by_name = {r['name']:r for r in rows}
    choices = {}
    for lane,reference_name in (('clean','primary_d000mm'),('robust','robust_d000mm')):
        reference = by_name[reference_name]
        selected = choose(rows,reference,lane)
        choices[lane] = dict(promoted=selected is not None,reference=reference,
                             selected=selected if selected is not None else reference)
    write_csv(out/'validation_candidate_summary.csv',rows)
    write_csv(out/'validation_per_fold.csv',fold_rows)
    write_csv(out/'validation_per_direction.csv',direction_rows)
    write_csv(out/'validation_error_slices.csv',slices)
    frozen = dict(validation_only=True,choices=choices,design_sha256=digest(out/'design.json'),
                  candidate_summary_sha256=digest(out/'validation_candidate_summary.csv'),
                  frozen_before_any_new_test_application=True,global_settings_not_pair_specific=True,
                  validation_seconds=time.perf_counter()-started,source_sha256=digest(__file__))
    write_json(out/'frozen_selection.json',frozen)
    write_json(out/'execution_state.json',dict(stage='validation_frozen',complete=False,
        clean_promoted=choices['clean']['promoted'],robust_promoted=choices['robust']['promoted']))
    for lane,choice in choices.items():
        r=choice['selected']
        print(lane,'promoted',choice['promoted'],r['name'],'clean',r['clean_binary_macro_f1'],'three-view',r['mean_binary_macro_f1'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','validate'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    {'prepare':prepare,'validate':validate}[args.stage](args.out)


if __name__=='__main__':
    main()
