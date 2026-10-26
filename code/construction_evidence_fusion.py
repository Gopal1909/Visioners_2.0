"""Final bounded validation check: frozen v2 neural + fresh match-distribution scores."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from learning_v3_control import ROOT, BASE, digest, write_json
from learning_v2_runtime import THRESHOLDS, write_csv
from construction_match_distributions import DEFAULT as PILOT
from construction_evidence_followup import VIEWS, FIELDS, summarize, summarize_views

DEFAULT=ROOT/'results/construction_evidence_fusion_2026-10-26_v1'
WEIGHTS=(0.,.25,.5,.75,1.)


def blend(neural,geometry,weight):
    neural=np.asarray(neural,dtype=np.float64);geometry=np.asarray(geometry,dtype=np.float64)
    assert neural.shape==geometry.shape and 0<=weight<=1
    assert np.isfinite(neural).all() and np.isfinite(geometry).all()
    assert ((neural>=0)&(neural<=1)&(geometry>=0)&(geometry<=1)).all()
    return (1-weight)*neural+weight*geometry


def run(out=DEFAULT):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'frozen_selection.json').exists():return
    selection=json.loads((BASE/'fold_0/learning/selection.json').read_text())
    neural=BASE/'fold_0/learning'/(selection['secondary_robust_variant']+'_seed42')
    geometry=PILOT/'models/fresh_match_distributions'
    assert json.loads((PILOT/'verification.json').read_text())['passed']
    if not (out/'design.json').exists():
        inputs={str(p):digest(p) for folder in (neural,geometry) for p in folder.glob('validation_*.npz')}
        for p in (neural/'frozen.json',geometry/'fit_complete.json',geometry/'model.joblib',neural/'best.pt',PILOT/'verification.json'):
            inputs[str(p)]=digest(p)
        write_json(out/'input_manifest.json',inputs)
        write_json(out/'design.json',dict(
            purpose='Check complementary errors between frozen neural and new match-distribution classifier without further fitting',
            neural=str(neural),geometry=str(geometry),fold=0,weights=list(WEIGHTS),thresholds=THRESHOLDS.tolist(),
            scores='Convex combination of raw model outputs; scores are not calibrated event probabilities',
            selection='One global weight and threshold shared by all six pairs, directions, and clean/mild/medium validation conditions',
            clean_constraints='Clean macro >= original v2 robust reference minus .005; clean FPR <= reference plus .005',
            pilot_gate='Three-view mean macro >= original v2 robust reference plus .01; mean changed-F1 noninferior',
            tie_break='Mean macro, mean changed-F1, clean macro, lower geometry weight, higher threshold',
            heldout_policy='Only if the validation gate passes, apply the frozen recipe once to existing fold0 six-pair clean and stress caches. No full-cloud preprocessing or neural inference. Scope remains fold0 follow-up; no five-fold promotion.',
            exposure='Additional development on previously reused same-site validation; previous held-out findings motivated the investigation. No untouched benchmark claim.',
            extra_fits=False,mask_or_label_changes=False,new_support_dependencies=False,
            input_manifest_sha256=digest(out/'input_manifest.json'),source_sha256=digest(__file__),reports_or_presentations=False))
        sources=out/'sources';sources.mkdir(exist_ok=True)
        for name in ('construction_evidence_fusion.py','construction_evidence_followup.py','multipair_metrics.py'):
            shutil.copy2(ROOT/'code'/name,sources/name)
    plan=json.loads((out/'design.json').read_text());assert plan['source_sha256']==digest(__file__)
    data={};reference={};meta=json.loads((neural/'frozen.json').read_text())
    for view in VIEWS:
        with np.load(neural/f'validation_{view}.npz') as z:data[view]={key:z[key] for key in (*FIELDS,'probability')}
        with np.load(geometry/f'validation_{view}.npz') as z:
            for key in FIELDS:np.testing.assert_array_equal(data[view][key],z[key])
            data[view]['geometry_probability']=z['probability']
        reference[view]=summarize(data[view],(data[view]['probability']>=meta['threshold']).astype(np.int8))[0]
    reference=summarize_views(reference);rows=[];directions=[]
    for weight in plan['weights']:
        for threshold in plan['thresholds']:
            views={}
            for view in VIEWS:
                d=data[view];p=blend(d['probability'],d['geometry_probability'],weight)
                views[view],grouped=summarize(d,(p>=threshold).astype(np.int8))
                directions.extend(dict(weight=weight,threshold=threshold,view=view,**r) for r in grouped)
            r=dict(weight=weight,threshold=threshold,**summarize_views(views))
            r['eligible']=(r['clean_binary_macro_f1']>=reference['clean_binary_macro_f1']-.005-1e-12
                           and r['clean_false_positive_rate']<=reference['clean_false_positive_rate']+.005+1e-12)
            rows.append(r)
    eligible=[r for r in rows if r['eligible']]
    ranking=lambda r:(r['mean_binary_macro_f1'],r['mean_changed_f1'],r['clean_binary_macro_f1'],-r['weight'],r['threshold'])
    selected=max(eligible,key=ranking) if eligible else None
    passed=(selected is not None and selected['mean_binary_macro_f1']>=reference['mean_binary_macro_f1']+.01-1e-12
            and selected['mean_changed_f1']>=reference['mean_changed_f1']-1e-12)
    write_csv(out/'validation_grid.csv',rows);write_csv(out/'validation_per_direction.csv',directions)
    frozen=dict(pilot_gate_passed=bool(passed),selected=selected,reference=reference,
        unconstrained_diagnostic=max(rows,key=ranking),design_sha256=digest(out/'design.json'),
        validation_grid_sha256=digest(out/'validation_grid.csv'),frozen_before_any_new_test_scoring=True,
        original_v2_unchanged=True,no_five_fold_model_promotion=True)
    write_json(out/'frozen_selection.json',frozen)
    write_json(out/'execution_state.json',dict(stage='validation_frozen',pilot_gate_passed=bool(passed),new_test_cases_scored=0))
    print('Frozen fusion pilot gate',passed,flush=True)
    print(json.dumps(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();run(a.out)
