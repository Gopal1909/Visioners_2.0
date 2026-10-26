# Stage 4 bounded M3C2-style pilot

25 October 2026, Asia/Kolkata. Implemented and tested; this clean pilot does **not** complete the broader matched robustness stage.

The frozen M3C2-style configuration scores macro F1 **0.3529** versus C2C **0.6274** on exactly the same 2,534 valid directional observations. Coverage is **63.35%**; **36.65%** of M3C2 predictions are unknown. These results do not support superiority over C2C.

## Environment, geometry and tests

Windows interpreter: `C:\Users\prani\anaconda3\envs\dl_env\python.exe`. Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3; plyfile verified in this environment. No environment/package changes. All **19 synthetic tests passed** before dataset evaluation. They cover identical and offset planes (both directions), tangential motion, flipped normals, common translation, oblique/wall normals, controlled noise, analytical population spread/registration LoD, strict effect floor, exact cylinder bounds, empty/degenerate/insufficient support, invalid uncertainty/normals, query limits, cross-scale stability, reproducible original-index sampling, directional labels and explicit abstention metrics.

PCA uses source-epoch spherical neighbourhoods at radii 0.10/0.20/0.40 m, at least 10 neighbours. Retain scales with curvature <=0.10, eigengap >=0.10 and tangent eigenvalue ratio >=0.05. A scale must agree within 30 degrees with at least one other adequate scale; select lowest curvature, breaking ties by smaller radius. This stricter two-scale agreement is an explicit implementation choice. Orient the largest absolute normal component positive (ties X/Y/Z). Signed displacement is target mean minus source mean along that normal; its sign alone does not encode added/removed.

Both epochs use the same core-centered cylinder, with radial distance <=r and absolute axial displacement <=L. Project **all** eligible points; means, population SDs and counts come from complete neighbourhoods. Source Scan 0 change means candidate removal; source Scan 1 change means candidate addition. Prediction -1 is unknown, 0 unchanged, 1 changed. Missing support is not automatically physical change.

`LoD95 = 1.96 * (sqrt(s_source^2/n_source + s_target^2/n_target) + registration_error_m)`

`changed = abs(signed_distance) > max(LoD95, minimum_effect_m)`

Reference pinned to **CloudCompare v2.13.2, revision 49dbbb6**. [Process source](https://github.com/CloudCompare/CloudCompare/blob/49dbbb6/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp) confirms mean-difference sign, registration placement, strict significance and shared symmetric single-pass cylinders. [Statistics source](https://github.com/CloudCompare/CloudCompare/blob/49dbbb6/plugins/core/Standard/qM3C2/src/qM3C2Tools.cpp) confirms population spread (`ddof=0`). CloudCompare takes projection diameter (ours is radius), and axial depth is a half-length. Normal-scale selection and unknown policy differ from the reference. No independent CloudCompare application output comparison was performed: **M3C2-style, not software-equivalent**.

## Sampling and frozen settings

Exactly 2,000 original points per epoch, balanced round-robin over occupied 2 m 3D tiles with random within-tile selections without replacement. Seeds are 42 and 43; all 270 / 238 occupied tiles are represented. Original indices, coordinates, labels and seeds are in `selected_cores.npz`. Full 4,444,935 / 5,089,793 point clouds provide support. Spatial balancing deliberately overweights sparse tiles relative to point-uniform sampling; metrics estimate this sampled distribution, not full-pair accuracy.

The primary projection radius 0.10 m, half-length 1.0 m, minimum effect 0.02 m and registration error 0.000 m were frozen **before label scoring** in `run_config.json`. No selection using evaluation-label scores, calibration tile tuning or held-out claim. All nine planned settings are displayed below; no best-row substitution for the primary result. This is exploratory within-pair evaluation. C2C remains Euclidean NN >0.075 m, historically selected on the full same pair.

Zero registration uncertainty means LoD is conditional on **no modelled registration uncertainty**; 0.0164 m historical median C2C is not a registration RMS. Sensitivities at 0.01/0.02 m are hypothetical, not calibrated. Correlated LiDAR samples invalidate a literal independent-sample 95% guarantee.

## Fair primary comparison

| method | scan | precision | recall | F1 | changed_IoU | valid | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M3C2-style | 0 | 0.1479 | 0.9766 | 0.2569 | 0.1474 | 1246 | 0.6230 |
| M3C2-style | 1 | 0.1612 | 1.0000 | 0.2776 | 0.1612 | 1288 | 0.6440 |
| C2C common support | 0 | 0.3892 | 0.5078 | 0.4407 | 0.2826 | 1246 | 0.6230 |
| C2C common support | 1 | 0.4586 | 0.5845 | 0.5139 | 0.3458 | 1288 | 0.6440 |

| method | directional_macro_f1 | combined_macro_f1 | combined_macro_mIoU | coverage | unknown_rate | conservative_accuracy_lower_bound |
| --- | --- | --- | --- | --- | --- | --- |
| M3C2-style | 0.2673 | 0.3529 | 0.2212 | 0.6335 | 0.3665 | 0.2680 |
| C2C common support | 0.4773 | 0.6274 | 0.4978 | 0.6335 | 0.0000 | 0.5530 |

Directional macro F1 is the arithmetic mean of removal/addition F1. Combined macro F1/mIoU pool the unchanged/removed/added directional classes using the existing Stage 2 functions. C2C has zero detector abstentions; its common-support coverage excludes points where M3C2 abstained. `support_excluded_rate` and detector `unknown_rate` are saved separately. The conservative lower bound counts every excluded directional observation as an error for both matched rows; C2C all-sampled accuracy is also saved separately. No full-support macro F1 is assigned to abstaining M3C2.

Validity is class-dependent:

| scan | label | total | valid_coverage |
| --- | --- | --- | --- |
| 0 | 0 | 1676 | 0.6671 |
| 0 | 1 | 324 | 0.3951 |
| 1 | 0 | 1612 | 0.7109 |
| 1 | 1 | 388 | 0.3660 |

This preferential exclusion of changed points limits interpretation of valid-only recall. `configuration_metrics.csv` retains TP/FP/FN/TN, class-specific coverage and precision/recall/F1/IoU for every configuration and both methods.

## Nine configurations and hypothetical registration sensitivity

| projection_radius_m | effect_floor_m | combined_macro_f1 | combined_macro_mIoU | coverage |
| --- | --- | --- | --- | --- |
| 0.0500 | 0.0000 | 0.2540 | 0.1463 | 0.5380 |
| 0.0500 | 0.0200 | 0.3813 | 0.2459 | 0.5380 |
| 0.0500 | 0.0500 | 0.5517 | 0.4078 | 0.5380 |
| 0.1000 | 0.0000 | 0.2051 | 0.1143 | 0.6335 |
| 0.1000 | 0.0200 | 0.3529 | 0.2212 | 0.6335 |
| 0.1000 | 0.0500 | 0.5410 | 0.3957 | 0.6335 |
| 0.2000 | 0.0000 | 0.1823 | 0.1009 | 0.7007 |
| 0.2000 | 0.0200 | 0.3326 | 0.2035 | 0.7007 |
| 0.2000 | 0.0500 | 0.5243 | 0.3766 | 0.7007 |

| registration_error_m | combined_macro_f1 | combined_macro_mIoU | coverage |
| --- | --- | --- | --- |
| 0.0100 | 0.4008 | 0.2623 | 0.6335 |
| 0.0200 | 0.5331 | 0.3879 | 0.6335 |

## Normal/support diagnostics and failure inspection

| scan | radius_m | count_ge10 | adequate | selected | median_count | median_nearest_cross_scale_angle_deg |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1000 | 1880 | 1103 | 28 | 159.0000 | 3.5149 |
| 0 | 0.2000 | 1980 | 1625 | 270 | 656.5000 | 2.9763 |
| 0 | 0.4000 | 1998 | 1595 | 1222 | 2534.0000 | 3.3750 |
| 1 | 0.1000 | 1932 | 1249 | 26 | 156.0000 | 3.3417 |
| 1 | 0.2000 | 1982 | 1644 | 326 | 642.0000 | 2.5376 |
| 1 | 0.4000 | 2000 | 1531 | 1181 | 2474.5000 | 2.8789 |

| scan | reason | count | fraction |
| --- | --- | --- | --- |
| 0 | valid | 1246 | 0.6230 |
| 0 | normal insufficient | 2 | 0.0010 |
| 0 | normal degenerate/quality rejected | 196 | 0.0980 |
| 0 | normal unstable | 282 | 0.1410 |
| 0 | source support | 36 | 0.0180 |
| 0 | target support | 202 | 0.1010 |
| 0 | source support + target support | 36 | 0.0180 |
| 1 | valid | 1288 | 0.6440 |
| 1 | normal degenerate/quality rejected | 240 | 0.1200 |
| 1 | normal unstable | 227 | 0.1135 |
| 1 | source support | 20 | 0.0100 |
| 1 | target support | 211 | 0.1055 |
| 1 | source support + target support | 14 | 0.0070 |

The primary result has 1,459 false positives among valid observations. Large projected counts can make the conditional LoD very small despite systematic alignment, normal or mixed-surface effects. This is a plausible mechanism, not a calibrated causal attribution. `neighbourhood_support_summary.csv` reports projected counts/spreads; per-core and per-scale arrays retain all quality values, rejection reasons and normal selections. No candidate sphere exceeded the resource cap if the reason table contains no resource-limit row.

The 0.5 m half-length diagnostic used 128 fixed sample positions per epoch, without labels or retuning. Common-valid counts were 76 / 77; median absolute distance differences were 0.000000 / 0.000000 m, and maxima 0.4265 / 0.4165 m. The large tail supports sensitivity to distant surfaces along some cylinders. L=1.0 m remains the frozen primary choice; its finite range can miss counterparts, and longer cylinders can combine unrelated surfaces.

`diagnostic_examples.csv` gives the first original-index example for each available TP/TN/FP/FN/unknown category after scoring, including signed distance, LoD, support and reason. This post-hoc diagnostic selection is not parameter selection. `diagnostic_examples.png` labels those examples on the 3D sampled scene. `pilot_3d.png` shows ground truth and both methods on common support, with gray unknown/excluded cores. `distance_uncertainty.png` and `configuration_comparison.png` display displacement/LoD and the full small sweep.

## Runtime, memory and scope

Shared full-cloud tree build: 2.344 s. M3C2 normal/projection geometry for all three radii plus this build: **16.100 s**. C2C queries for the same 4,000 cores plus this build: **2.382 s**. M3C2 includes all radii jointly; no single-radius or independent-process timing claim. Data loading, spatial sampling, archive verification, plots and report processing are separate. Experiment elapsed through the main plots: 19.272 s (report generation/hash finalization excluded).

Highest sampled process RSS through those plots: 941.5 MiB; Windows lifetime peak reported during recorded stages: 941.5 MiB. Absolute RSS contains both full clouds, both trees and previously allocated arrays. Scoped RSS growth and 10 ms sampled peaks are in `runtime_memory_stages.csv`; they are not isolated algorithm memory allocations. Transient neighbourhood memory is bounded by one core/epoch candidate list at a time, at most 250,000 candidates. Oversized lists abstain; no truncation or full pairwise matrix. Approximate observed throughput is 290.8 directional cores/s for normals and all radii. Linear projection to 20,000 cores/epoch would be about 139.9 s for geometry under identical density, but this is an untested estimate and not a full-cloud feasibility claim.

Historical C2C full-pair results remain separate in `historical_c2c_full_pair.csv` (macro F1 about 0.7359, mIoU about 0.6046). C2C all 4,000 sampled cores is separately listed in `configuration_metrics.csv`. Neither is the primary matched comparison above. Fresh C2C distances were checked against cached distances and original labels on exactly these core indices; maximum differences are under 0.000002 m (float32 cache rounding).

No full-cloud M3C2 core sweep, robustness rerun, 20,000-core expansion, learning-model training or original-data modification. The requested bounded pilot is complete. Broader Stage 4 still needs independent reference-output validation if available, geometry/failure review, and matched robustness experiments with freshly estimated perturbed normals and common masks. The Stage 3 helpers were inspected but are not invoked because this request's pilot is clean. A future run must reuse those helpers, seeds and occlusion protocols, without copying or silently changing them. Parameter changes after seeing these scores require a new versioned configuration and explicit exploratory disclosure.

## Reproduction and artifact schema

From `D:\CV_Project` in Windows PowerShell:

```powershell
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' code/m3c2_baseline.py --smoke-test
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' code/m3c2_baseline.py --output-dir results/pair_01/m3c2/new_run
```

The output directory must not already exist. CLI permits at most 20,000 cores/epoch and reruns the synthetic gate before loading the dataset. Configurable Python `Config` exposes all normal/cylinder/count/quality/resource parameters; CLI registration uncertainty is independent of the effect floor.

`normals_scan*.npz`: rows match selected original cores, columns match normal radii; unit normals, selected radii, exclusive normal reason, per-scale neighbour counts/eigenvalues/curvature/eigengap/tangent ratio, adequate masks and nearest cross-scale angles.

`raw_scan*.npz`: rows match cores, columns match projection radii; signed distances, both mean positions, population spreads, counts and combined reason masks. Source/target are relative to the direction (source Scan 1 reverses epoch roles). Distances/uncertainties are NaN when unknown.

`predictions_r*_f*.npz`: per-scan binary predictions with -1 sentinel, validity, combined reasons and LoD95. `c2c_same_cores.npz`: fresh full-support NN distances and fixed-threshold predictions. Normal reasons 1/2/3 are exclusive, projection support flags 4/8/16/32/64 can combine; see `run_config.json`.

`saved_output_audit.json`: 36,000 saved predictions audited for validity, support, strict decision rule and matched-support counts. `input_source_manifest.json` hashes dataset inputs and source scripts; preservation verification lives in the parent M3C2 directory. The first exploratory v1 run is retained; the final run adds reporting/metric semantics and diagnostics without changing geometry or the frozen detector configuration.
