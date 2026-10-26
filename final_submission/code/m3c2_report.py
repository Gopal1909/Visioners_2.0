"""Regenerate pilot summaries/figures from saved cores and measurements only.

No dataset reads, geometric reruns, parameter selection or modification of Stage 2/3.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def md_table(frame):
    # Do not require the optional tabulate package in the user's existing env.
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x)
    rows = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def generate_report(directory):
    p = Path(directory)
    config = json.loads((p/'run_config.json').read_text())
    cfg = config['config']
    j = cfg['projection_radii_m'].index(.10)
    metrics = pd.read_csv(p/'configuration_metrics.csv')
    cores = dict(np.load(p/'selected_cores.npz'))
    raw = [dict(np.load(p/f'raw_scan{i}.npz')) for i in (0,1)]
    normals = [dict(np.load(p/f'normals_scan{i}.npz')) for i in (0,1)]
    primary = dict(np.load(p/'predictions_r0.10_f0.02.npz'))
    runtime = json.loads((p/'runtime_memory.json').read_text())
    sampling = json.loads((p/'sampling.json').read_text())
    diagnostic = json.loads((p/'half_length_diagnostic.json').read_text())
    tests = json.loads((p/'synthetic_tests.json').read_text())
    gt = [cores[f'cd_type{i}'] for i in (0,1)]
    quality_rows, support_rows, reason_rows, example_rows = [], [], [], []
    for i in (0,1):
        n = normals[i]
        selected = n['normal_reason'] == 0
        for k, radius in enumerate(cfg['normal_radii_m']):
            adequate = n['adequate'][:,k]
            angles = n['stability_angle_deg'][:,k]
            finite_angles = angles[np.isfinite(angles)]
            quality_rows.append(dict(scan=i, radius_m=radius, cores=len(selected),
                count_ge10=int((n['counts'][:,k]>=10).sum()), adequate=int(adequate.sum()),
                selected=int(np.sum(n['radius_m']==radius)),
                median_count=float(np.median(n['counts'][:,k])),
                median_curvature_adequate=float(np.median(n['curvature'][adequate,k])) if adequate.any() else np.nan,
                median_eigengap_adequate=float(np.median(n['eigengap'][adequate,k])) if adequate.any() else np.nan,
                median_nearest_cross_scale_angle_deg=float(np.median(finite_angles)) if len(finite_angles) else np.nan))
        for k, radius in enumerate(cfg['projection_radii_m']):
            r = raw[i]
            valid = r['reason'][:,k] == 0
            for tag in ('source','target'):
                counts = r[f'count_{tag}'][:,k]
                spread = r[f'spread_{tag}_m'][:,k]
                usable = np.isfinite(spread)
                support_rows.append(dict(scan=i, projection_radius_m=radius, epoch_role=tag,
                    normal_valid=int(selected.sum()), distance_valid=int(valid.sum()),
                    count_below10_on_normal_valid=int(np.sum(selected & (counts<10))),
                    median_count_normal_valid=float(np.median(counts[selected])),
                    max_count=int(counts.max()),
                    median_spread_m=float(np.median(spread[usable])) if usable.any() else np.nan,
                    p95_spread_m=float(np.percentile(spread[usable],95)) if usable.any() else np.nan))
            codes, counts = np.unique(r['reason'][:,k], return_counts=True)
            for code, count in zip(codes, counts):
                if code in (0,1,2,3):
                    name = {0:'valid',1:'normal insufficient',2:'normal degenerate/quality rejected',3:'normal unstable'}[int(code)]
                else:
                    name = ' + '.join(name for bit,name in ((4,'source support'),(8,'target support'),
                        (16,'invalid uncertainty'),(32,'resource limit'),(64,'invalid normal')) if code & bit)
                reason_rows.append(dict(scan=i, projection_radius_m=radius, reason_code=int(code),
                                        reason=name, count=int(count), fraction=float(count/len(valid))))
        pred = primary[f'scan{i}_prediction']
        categories = np.where(pred < 0, 'unknown',
            np.where(gt[i] == pred, np.where(pred == 1, 'TP', 'TN'),
                     np.where(pred == 1, 'FP', 'FN')))
        # Examples are selected after scoring for diagnosis, never to tune parameters.
        for category in ('TP','TN','FP','FN','unknown'):
            found = np.flatnonzero(categories == category)
            if not len(found):
                continue
            a = int(found[0])
            example_rows.append(dict(scan=i, category=category, original_index=int(cores[f'indices{i}'][a]),
                sample_position=a, cd_type=int(gt[i][a]), prediction=int(pred[a]),
                x_m=cores[f'xyz{i}'][a,0], y_m=cores[f'xyz{i}'][a,1], z_m=cores[f'xyz{i}'][a,2],
                signed_distance_m=raw[i]['signed_distance_m'][a,j],
                lod95_m=primary[f'scan{i}_lod95_m'][a],
                count_source=int(raw[i]['count_source'][a,j]), count_target=int(raw[i]['count_target'][a,j]),
                normal_radius_m=n['radius_m'][a], reason=int(primary[f'scan{i}_reason'][a])))
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(p/'normal_quality_summary.csv',index=False)
    support = pd.DataFrame(support_rows)
    support.to_csv(p/'neighbourhood_support_summary.csv',index=False)
    reasons = pd.DataFrame(reason_rows)
    reasons.to_csv(p/'reason_summary.csv',index=False)
    examples = pd.DataFrame(example_rows)
    examples.to_csv(p/'diagnostic_examples.csv',index=False)

    # Audit saved decisions and exact support without invoking the geometry again.
    checked = 0
    for radius in cfg['projection_radii_m']:
        k = cfg['projection_radii_m'].index(radius)
        for floor in config['effect_floors_m']:
            with np.load(p/f'predictions_r{radius:.2f}_f{floor:.2f}.npz') as d:
                for i in (0,1):
                    valid = d[f'scan{i}_valid']
                    pred = d[f'scan{i}_prediction']
                    assert np.array_equal(valid, d[f'scan{i}_reason'] == 0)
                    assert np.all(pred[~valid] == -1)
                    assert np.isfinite(d[f'scan{i}_lod95_m'][valid]).all()
                    assert np.isfinite(raw[i]['signed_distance_m'][valid,k]).all()
                    assert np.all(raw[i]['count_source'][valid,k]>=10)
                    assert np.all(raw[i]['count_target'][valid,k]>=10)
                    expected = np.abs(raw[i]['signed_distance_m'][valid,k]) > np.maximum(
                        d[f'scan{i}_lod95_m'][valid], floor)
                    assert np.array_equal(pred[valid], expected)
                    pairs = metrics[(metrics.projection_radius_m==radius)&(metrics.effect_floor_m==floor)&
                                    (metrics.registration_error_m==cfg['registration_error_m'])]
                    pair = pairs[pairs.method.isin(['M3C2-style','C2C common support'])]
                    assert len(pair)==2 and pair[f'scan{i}_valid'].nunique()==1
                    assert int(pair[f'scan{i}_valid'].iloc[0])==int(valid.sum())
                    checked += len(pred)
    (p/'saved_output_audit.json').write_text(json.dumps(dict(
        passed=True, audited_predictions=checked, configurations=9,
        checks=['unknown sentinel and reason/valid agreement','at least 10 neighbours per epoch',
                'finite valid measurements/LoD','strict LoD and floor decisions',
                'exact per-direction common-support counts']),indent=2))
    selected = metrics[(metrics.projection_radius_m==.10)&(metrics.effect_floor_m==.02)&
                        (metrics.registration_error_m==cfg['registration_error_m'])]
    selected.to_csv(p/'primary_comparison.csv',index=False)
    primary_m = selected[selected.method=='M3C2-style'].iloc[0]
    primary_c = selected[selected.method=='C2C common support'].iloc[0]
    sweep = metrics[(metrics.method=='M3C2-style') &
                    (metrics.registration_error_m==cfg['registration_error_m'])]
    timing_rows = [dict(stage=k, **v) for k,v in runtime.items() if isinstance(v,dict)]
    pd.DataFrame(timing_rows).to_csv(p/'runtime_memory_stages.csv',index=False)
    plot_metrics(p, metrics, quality, examples, primary)
    directional = []
    for _,row in selected.iterrows():
        for i in (0,1):
            directional.append(dict(method=row.method, scan=i, precision=row[f'scan{i}_precision'],
                recall=row[f'scan{i}_recall'], F1=row[f'scan{i}_f1'], changed_IoU=row[f'scan{i}_iou_changed'],
                valid=int(row[f'scan{i}_valid']), coverage=row[f'scan{i}_coverage']))
    class_coverage = pd.DataFrame([dict(scan=i, label=c, total=int(primary_m[f'scan{i}_gt{c}_total']),
        valid_coverage=primary_m[f'scan{i}_gt{c}_coverage']) for i in (0,1) for c in (0,1)])
    max_peak = max(v['sampled_peak_rss_mib'] for v in runtime.values() if isinstance(v,dict))
    mtime = runtime['m3c2_geometry_all_radii_seconds']
    ctime = runtime['c2c_primary_search_seconds']
    process_peak = max(v['process_lifetime_peak_mib'] for v in runtime.values() if isinstance(v,dict))
    direction_time = sum(runtime[k]['seconds'] for k in ('normal_scan0','normal_scan1','projection_scan0','projection_scan1'))
    body = f'''# Stage 4 bounded M3C2-style pilot

25 October 2026, Asia/Kolkata. Implemented and tested; this clean pilot does **not** complete the broader matched robustness stage.

The frozen M3C2-style configuration scores macro F1 **{primary_m.combined_macro_f1:.4f}** versus C2C **{primary_c.combined_macro_f1:.4f}** on exactly the same {int(primary_m.scan0_valid+primary_m.scan1_valid):,} valid directional observations. Coverage is **{primary_m.coverage:.2%}**; **{primary_m.unknown_rate:.2%}** of M3C2 predictions are unknown. These results do not support superiority over C2C.

## Environment, geometry and tests

Windows interpreter: `{config['executable']}`. Python {config['python'].split()[0]}, NumPy {config['numpy']}, SciPy {config['scipy']}; plyfile verified in this environment. No environment/package changes. All **{tests['tests']} synthetic tests passed** before dataset evaluation. They cover identical and offset planes (both directions), tangential motion, flipped normals, common translation, oblique/wall normals, controlled noise, analytical population spread/registration LoD, strict effect floor, exact cylinder bounds, empty/degenerate/insufficient support, invalid uncertainty/normals, query limits, cross-scale stability, reproducible original-index sampling, directional labels and explicit abstention metrics.

PCA uses source-epoch spherical neighbourhoods at radii 0.10/0.20/0.40 m, at least 10 neighbours. Retain scales with curvature <=0.10, eigengap >=0.10 and tangent eigenvalue ratio >=0.05. A scale must agree within 30 degrees with at least one other adequate scale; select lowest curvature, breaking ties by smaller radius. This stricter two-scale agreement is an explicit implementation choice. Orient the largest absolute normal component positive (ties X/Y/Z). Signed displacement is target mean minus source mean along that normal; its sign alone does not encode added/removed.

Both epochs use the same core-centered cylinder, with radial distance <=r and absolute axial displacement <=L. Project **all** eligible points; means, population SDs and counts come from complete neighbourhoods. Source Scan 0 change means candidate removal; source Scan 1 change means candidate addition. Prediction -1 is unknown, 0 unchanged, 1 changed. Missing support is not automatically physical change.

`LoD95 = 1.96 * (sqrt(s_source^2/n_source + s_target^2/n_target) + registration_error_m)`

`changed = abs(signed_distance) > max(LoD95, minimum_effect_m)`

Reference pinned to **CloudCompare v2.13.2, revision 49dbbb6**. [Process source](https://github.com/CloudCompare/CloudCompare/blob/49dbbb6/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp) confirms mean-difference sign, registration placement, strict significance and shared symmetric single-pass cylinders. [Statistics source](https://github.com/CloudCompare/CloudCompare/blob/49dbbb6/plugins/core/Standard/qM3C2/src/qM3C2Tools.cpp) confirms population spread (`ddof=0`). CloudCompare takes projection diameter (ours is radius), and axial depth is a half-length. Normal-scale selection and unknown policy differ from the reference. No independent CloudCompare application output comparison was performed: **M3C2-style, not software-equivalent**.

## Sampling and frozen settings

Exactly {config['cores_per_scan']:,} original points per epoch, balanced round-robin over occupied 2 m 3D tiles with random within-tile selections without replacement. Seeds are {sampling[0]['seed']} and {sampling[1]['seed']}; all {sampling[0]['occupied_tiles']} / {sampling[1]['occupied_tiles']} occupied tiles are represented. Original indices, coordinates, labels and seeds are in `selected_cores.npz`. Full 4,444,935 / 5,089,793 point clouds provide support. Spatial balancing deliberately overweights sparse tiles relative to point-uniform sampling; metrics estimate this sampled distribution, not full-pair accuracy.

The primary projection radius 0.10 m, half-length 1.0 m, minimum effect 0.02 m and registration error {cfg['registration_error_m']:.3f} m were frozen **before label scoring** in `run_config.json`. No selection using evaluation-label scores, calibration tile tuning or held-out claim. All nine planned settings are displayed below; no best-row substitution for the primary result. This is exploratory within-pair evaluation. C2C remains Euclidean NN >0.075 m, historically selected on the full same pair.

Zero registration uncertainty means LoD is conditional on **no modelled registration uncertainty**; 0.0164 m historical median C2C is not a registration RMS. Sensitivities at 0.01/0.02 m are hypothetical, not calibrated. Correlated LiDAR samples invalidate a literal independent-sample 95% guarantee.

## Fair primary comparison

{md_table(pd.DataFrame(directional))}

{md_table(selected[['method','directional_macro_f1','combined_macro_f1','combined_macro_mIoU','coverage','unknown_rate','conservative_accuracy_lower_bound']])}

Directional macro F1 is the arithmetic mean of removal/addition F1. Combined macro F1/mIoU pool the unchanged/removed/added directional classes using the existing Stage 2 functions. C2C has zero detector abstentions; its common-support coverage excludes points where M3C2 abstained. `support_excluded_rate` and detector `unknown_rate` are saved separately. The conservative lower bound counts every excluded directional observation as an error for both matched rows; C2C all-sampled accuracy is also saved separately. No full-support macro F1 is assigned to abstaining M3C2.

Validity is class-dependent:

{md_table(class_coverage)}

This preferential exclusion of changed points limits interpretation of valid-only recall. `configuration_metrics.csv` retains TP/FP/FN/TN, class-specific coverage and precision/recall/F1/IoU for every configuration and both methods.

## Nine configurations and hypothetical registration sensitivity

{md_table(sweep[['projection_radius_m','effect_floor_m','combined_macro_f1','combined_macro_mIoU','coverage']])}

{md_table(metrics[(metrics.method=='M3C2-style') & metrics.hypothetical_registration_sensitivity][['registration_error_m','combined_macro_f1','combined_macro_mIoU','coverage']])}

## Normal/support diagnostics and failure inspection

{md_table(quality[['scan','radius_m','count_ge10','adequate','selected','median_count','median_nearest_cross_scale_angle_deg']])}

{md_table(reasons[reasons.projection_radius_m==.10][['scan','reason','count','fraction']])}

The primary result has {int(primary_m.scan0_FP+primary_m.scan1_FP):,} false positives among valid observations. Large projected counts can make the conditional LoD very small despite systematic alignment, normal or mixed-surface effects. This is a plausible mechanism, not a calibrated causal attribution. `neighbourhood_support_summary.csv` reports projected counts/spreads; per-core and per-scale arrays retain all quality values, rejection reasons and normal selections. No candidate sphere exceeded the resource cap if the reason table contains no resource-limit row.

The 0.5 m half-length diagnostic used 128 fixed sample positions per epoch, without labels or retuning. Common-valid counts were {diagnostic[0]['common_valid']} / {diagnostic[1]['common_valid']}; median absolute distance differences were {diagnostic[0]['median_absolute_distance_difference_m']:.6f} / {diagnostic[1]['median_absolute_distance_difference_m']:.6f} m, and maxima {diagnostic[0]['max_absolute_distance_difference_m']:.4f} / {diagnostic[1]['max_absolute_distance_difference_m']:.4f} m. The large tail supports sensitivity to distant surfaces along some cylinders. L=1.0 m remains the frozen primary choice; its finite range can miss counterparts, and longer cylinders can combine unrelated surfaces.

`diagnostic_examples.csv` gives the first original-index example for each available TP/TN/FP/FN/unknown category after scoring, including signed distance, LoD, support and reason. This post-hoc diagnostic selection is not parameter selection. `diagnostic_examples.png` labels those examples on the 3D sampled scene. `pilot_3d.png` shows ground truth and both methods on common support, with gray unknown/excluded cores. `distance_uncertainty.png` and `configuration_comparison.png` display displacement/LoD and the full small sweep.

## Runtime, memory and scope

Shared full-cloud tree build: {runtime['shared_full_support_trees']['seconds']:.3f} s. M3C2 normal/projection geometry for all three radii plus this build: **{mtime:.3f} s**. C2C queries for the same 4,000 cores plus this build: **{ctime:.3f} s**. M3C2 includes all radii jointly; no single-radius or independent-process timing claim. Data loading, spatial sampling, archive verification, plots and report processing are separate. Experiment elapsed through the main plots: {runtime['experiment_seconds_through_plotting']:.3f} s (report generation/hash finalization excluded).

Highest sampled process RSS through those plots: {max_peak:.1f} MiB; Windows lifetime peak reported during recorded stages: {process_peak:.1f} MiB. Absolute RSS contains both full clouds, both trees and previously allocated arrays. Scoped RSS growth and 10 ms sampled peaks are in `runtime_memory_stages.csv`; they are not isolated algorithm memory allocations. Transient neighbourhood memory is bounded by one core/epoch candidate list at a time, at most {cfg['max_candidates']:,} candidates. Oversized lists abstain; no truncation or full pairwise matrix. Approximate observed throughput is {4000/direction_time:.1f} directional cores/s for normals and all radii. Linear projection to 20,000 cores/epoch would be about {direction_time*10+runtime['shared_full_support_trees']['seconds']:.1f} s for geometry under identical density, but this is an untested estimate and not a full-cloud feasibility claim.

Historical C2C full-pair results remain separate in `historical_c2c_full_pair.csv` (macro F1 about 0.7359, mIoU about 0.6046). C2C all 4,000 sampled cores is separately listed in `configuration_metrics.csv`. Neither is the primary matched comparison above. Fresh C2C distances were checked against cached distances and original labels on exactly these core indices; maximum differences are under 0.000002 m (float32 cache rounding).

No full-cloud M3C2 core sweep, robustness rerun, 20,000-core expansion, learning-model training or original-data modification. The requested bounded pilot is complete. Broader Stage 4 still needs independent reference-output validation if available, geometry/failure review, and matched robustness experiments with freshly estimated perturbed normals and common masks. The Stage 3 helpers were inspected but are not invoked because this request's pilot is clean. A future run must reuse those helpers, seeds and occlusion protocols, without copying or silently changing them. Parameter changes after seeing these scores require a new versioned configuration and explicit exploratory disclosure.

## Reproduction and artifact schema

From `D:\\CV_Project` in Windows PowerShell:

```powershell
& 'C:\\Users\\prani\\anaconda3\\envs\\dl_env\\python.exe' code/m3c2_baseline.py --smoke-test
& 'C:\\Users\\prani\\anaconda3\\envs\\dl_env\\python.exe' code/m3c2_baseline.py --output-dir results/pair_01/m3c2/new_run
```

The output directory must not already exist. CLI permits at most 20,000 cores/epoch and reruns the synthetic gate before loading the dataset. Configurable Python `Config` exposes all normal/cylinder/count/quality/resource parameters; CLI registration uncertainty is independent of the effect floor.

`normals_scan*.npz`: rows match selected original cores, columns match normal radii; unit normals, selected radii, exclusive normal reason, per-scale neighbour counts/eigenvalues/curvature/eigengap/tangent ratio, adequate masks and nearest cross-scale angles.

`raw_scan*.npz`: rows match cores, columns match projection radii; signed distances, both mean positions, population spreads, counts and combined reason masks. Source/target are relative to the direction (source Scan 1 reverses epoch roles). Distances/uncertainties are NaN when unknown.

`predictions_r*_f*.npz`: per-scan binary predictions with -1 sentinel, validity, combined reasons and LoD95. `c2c_same_cores.npz`: fresh full-support NN distances and fixed-threshold predictions. Normal reasons 1/2/3 are exclusive, projection support flags 4/8/16/32/64 can combine; see `run_config.json`.

`saved_output_audit.json`: {checked:,} saved predictions audited for validity, support, strict decision rule and matched-support counts. `input_source_manifest.json` hashes dataset inputs and source scripts; preservation verification lives in the parent M3C2 directory. The first exploratory v1 run is retained; the final run adds reporting/metric semantics and diagnostics without changing geometry or the frozen detector configuration.
'''
    (p/'pilot_report.md').write_text(body, encoding='utf-8')
    brief = f'''# Bounded M3C2-style pilot: findings

25 October 2026. The implementation and requested clean pilot are complete. The geometry passes **{tests['tests']} synthetic tests**, but this pilot does **not** support an accuracy advantage over C2C.

Exactly 2,000 original cores per scan were selected across all occupied 2 m 3D tiles using seeds 42/43. Full clouds supplied all neighbourhoods. The primary configuration was fixed before scoring: PCA radii 0.10/0.20/0.40 m, lowest-curvature scale with adequate eigengap and cross-scale agreement, projection radius 0.10 m, half-length 1.0 m, effect floor 0.02 m, and at least 10 projected neighbours per epoch.

| Quantity | M3C2-style | C2C on identical valid cores |
|---|---:|---:|
| Pooled 3-class macro F1 | {primary_m.combined_macro_f1:.4f} | {primary_c.combined_macro_f1:.4f} |
| Pooled macro mIoU | {primary_m.combined_macro_mIoU:.4f} | {primary_c.combined_macro_mIoU:.4f} |
| Directional macro F1 | {primary_m.directional_macro_f1:.4f} | {primary_c.directional_macro_f1:.4f} |
| Accuracy on valid support | {primary_m.combined_overall_accuracy:.2%} | {primary_c.combined_overall_accuracy:.2%} |
| Correct valid / all 4,000 cores | {primary_m.conservative_accuracy_lower_bound:.2%} | {primary_c.conservative_accuracy_lower_bound:.2%} |

Both score the same {int(primary_m.scan0_valid):,}/{int(primary_m.scan1_valid):,} directional cores: **{primary_m.coverage:.2%} coverage**. M3C2 abstains on **{primary_m.unknown_rate:.2%}**. C2C itself has zero abstentions; common-support exclusions are reported separately.

{md_table(pd.DataFrame(directional)[['method','scan','precision','recall','F1','changed_IoU']])}

The LoD uses population projected spread and `1.96 * (sqrt(s0²/n0 + s1²/n1) + registration_error)`, checked against [CloudCompare v2.13.2 / revision 49dbbb6](https://github.com/CloudCompare/CloudCompare/blob/v2.13.2/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp). This is **M3C2-style**, with no independent reference-application output validation. Registration error is a separate parameter, set to zero here: uncertainty is conditional on no modelled registration error. The 1.64 cm historical NN median is not used as registration RMS. Hypothetical 1/2 cm registration sensitivities are saved separately.

The nine radius/effect configurations, raw signed distances, LoDs, support/normal diagnostics, explicit unknown reason masks and per-scan predictions are saved. Geometry took **{mtime:.2f} s** for normals/all three radii including shared trees; C2C search took **{ctime:.2f} s** including the same tree build. Sampled process RSS peaked at **{max_peak:.0f} MiB** through the main plots; these are shared-process measurements, not isolated algorithm peaks.

Limitations: primary false positives are frequent; changed-class valid coverage is only {primary_m.scan0_gt1_coverage:.1%}/{primary_m.scan1_gt1_coverage:.1%}. Correlated samples, unmodelled alignment uncertainty, unreliable normals and mixed surfaces can undermine conditional LoD. Halving cylinder length changed some distances by {max(x['max_absolute_distance_difference_m'] for x in diagnostic):.3f} m, despite zero median change, supporting concern about distant surfaces in some cylinders. Spatial balancing overweights sparse regions, so sampled scores do not estimate full-pair accuracy. No held-out or cross-scene claim, and no score-based primary retuning.

See [full report](pilot_report.md), [metrics](configuration_metrics.csv), [3D view](pilot_3d.png), [sweep figure](configuration_comparison.png), and [failure examples](diagnostic_examples.png). Historical full-pair C2C stays separately labelled. Broader Stage 4 matched robustness and independent software validation remain outstanding. Original data and Stage 2/3 outputs are preserved; no model training or full-cloud M3C2 sweep was run.
'''
    (p/'pilot_summary.md').write_text(brief,encoding='utf-8')


def plot_metrics(p, metrics, quality, examples, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = metrics[(metrics.method=='M3C2-style') & ~metrics.hypothetical_registration_sensitivity]
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for radius,g in m.groupby('projection_radius_m'):
        axes[0].plot(g.effect_floor_m*100,g.combined_macro_f1,'o-',label=f'M3C2 r={radius:g} m')
        c = metrics[(metrics.method=='C2C common support') & (metrics.projection_radius_m==radius) &
                    ~metrics.hypothetical_registration_sensitivity].iloc[0]
        axes[0].axhline(c.combined_macro_f1,ls='--',alpha=.6,label=f'C2C common support r={radius:g}')
    axes[0].set(xlabel='Minimum effect floor (cm)',ylabel='Combined 3-class macro F1',ylim=(0,1),
                title='Nine settings; frozen primary r=0.10 m / floor=2 cm')
    axes[0].legend(fontsize=7)
    g=m[m.effect_floor_m==.02]
    axes[1].bar(g.projection_radius_m.astype(str),g.coverage,color='#2563eb',label='valid')
    axes[1].bar(g.projection_radius_m.astype(str),1-g.coverage,bottom=g.coverage,color='#9ca3af',label='unknown')
    axes[1].set(xlabel='Projection radius (m)',ylabel='Fraction of 4,000 cores',ylim=(0,1),title='Coverage is separate from valid-only accuracy')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(p/'configuration_comparison.png',dpi=170); plt.close(fig)
    cores=dict(np.load(p/'selected_cores.npz'))
    fig=plt.figure(figsize=(13,6))
    palette={'TP':'#16a34a','TN':'#2563eb','FP':'#dc2626','FN':'#9333ea','unknown':'#6b7280'}
    for i in (0,1):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        xyz=cores[f'xyz{i}']
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=2,c='#cbd5e1',alpha=.4)
        for _,r in examples[examples.scan==i].iterrows():
            ax.scatter([r.x_m],[r.y_m],[r.z_m],s=60,c=palette[r.category],label=f'{r.category}: index {r.original_index}')
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f'Scan {i}: deterministic diagnostic examples')
        ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.1))
        ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Examples selected after scoring for diagnosis; no detector retuning')
    fig.tight_layout(); fig.savefig(p/'diagnostic_examples.png',dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    generate_report(parser.parse_args().directory)
