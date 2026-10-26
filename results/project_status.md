# Project status: Robust 3D Change Detection for Construction Progress Monitoring

Updated 25 October 2026 (Asia/Kolkata). Presentation: **2 November 2026**. The bounded three-method robustness study, compact CUDA-trained learned baseline, report, presentation and offline demo are now assembled. Read **`results/final_status.md`** and `final_submission/README.md` for the authoritative current state. Earlier audit/pilot sections below are retained as history and superseded where they describe pending work or prohibit the training now explicitly authorized by the master instruction.

## Current final study

`results/study_2026-10-25_v1/` contains the buffered same-scene split, calibration freeze, 4,000 requested original cores per epoch, 11 matched clean/stress conditions, learned training/checkpoints and audited analyses. On common clean support, macro F1 is **0.4118 C2C, 0.3817 Siamese PointNet + NN, 0.2342 M3C2-style**, with 66.875% coverage and 33.125% M3C2 detector unknowns. No superiority or cross-scene claim follows. `results/occlusion_supplement_2026-10-25_v1/` separately evaluates geometric occlusion on all-scene pilot indices because the spatial-test stripe misses the blind cylinders. ConSLAM v2 is qualitative within one acquisition sequence.

The new baseline actually trained for 30 fixed epochs on the RTX 4050; calibration selected epoch 1 and threshold 0.7 before test inference. All 25 geometry, learning and plotting tests pass. The historical C2C/Stage 3 results and original datasets are preserved. No additional data downloaded: official missing-pair links redirected to sign-in. `final_submission/` contains a six-page PDF/editable source, 11-slide editable PPTX with eight-minute notes, real tables/plots, model weights and reproducible offline demo. See the final verification evidence before submission.

## Latest: Stage 4 bounded pilot

Read `results/pair_01/m3c2/pilot_2026-10-25_v2/pilot_summary.md` for the current concise result; `pilot_report.md` contains full conventions, diagnostics and limitations. Source: `code/m3c2_baseline.py`, `code/test_m3c2_synthetic.py`, `code/m3c2_report.py`. Existing Windows `dl_env` Python, NumPy, SciPy and plyfile execution is now verified. All **19 synthetic tests passed** before the final dataset run.

Exactly 2,000 spatially balanced original cores per epoch (seeds 42/43) use complete input clouds as neighbourhood support. Primary geometry was frozen before label scoring: PCA radii 0.10/0.20/0.40 m with quality and cross-scale stability checks; projection radius 0.10 m; axial half-length 1.0 m; minimum effect 0.02 m; at least 10 projected neighbours per epoch; registration uncertainty 0 m, explicitly conditional on no modelled registration error. CloudCompare v2.13.2 / revision 49dbbb6 source conventions were checked, but independent reference-application output validation has not been done.

| Primary matched-support result | M3C2-style | C2C, fixed 0.075 m |
|---|---:|---:|
| Pooled directional three-class macro F1 | 0.3529 | 0.6274 |
| Pooled macro mIoU | 0.2212 | 0.4978 |
| Mean removal/addition F1 | 0.2673 | 0.4773 |
| Common evaluation coverage | 63.35% | 63.35% |
| Detector unknown rate | 36.65% | 0% |

The methods score identical 1,246/1,288 valid original cores. C2C's common-support exclusions are distinct from detector unknowns. Conservative correct-valid/all-core accuracy is 26.80% / 55.30%; the all-sampled C2C row is saved separately. Historical full-pair C2C numbers below are not directly comparable. No superiority claim is supported. Spatial balancing and class-dependent coverage limit full-pair interpretation; changed-point coverage is only 39.5% / 36.6%.

All nine planned radius/effect settings, hypothetical registration sensitivities, saved measurements/predictions/quality reasons, timing/RSS records, 3D and diagnostic figures are under the v2 directory. The 0.5 m half-length diagnostic shows maximum distance shifts of about 0.427/0.416 m, motivating mixed-surface inspection. Geometry for all three radii plus shared trees took 16.10 s; C2C sampled queries plus the same tree build took 2.38 s. Peak sampled process RSS through the main plots was about 941 MiB; this is shared-process memory, not isolated per-method memory. The v1 exploratory run is retained; v2 improves reporting and metric semantics with unchanged detector geometry/configuration.

`results/pair_01/m3c2/preservation_verified.json` confirms all 27 protected dataset, Stage 2/3 source and result files remained byte-identical. No full-cloud M3C2 core sweep, 20,000-core expansion or model training was run. Stage 4 is **pilot complete**, not complete under the broader plan's robustness gates.

The research question remains detection of genuine 3D changes under registration error, sparsity, noise and occlusion. Quantitative results are restricted to MultiChange3D BikeParkingConstruction `pair_01`. ConSLAM remains a qualitative construction-site visualization source.

## Completed work and outputs

| Stage | Status | Reuse these locations |
|---|---|---|
| 1: dataset/environment inspection | Completed historically; current audit confirmed PLY headers/file sizes and reviewed inspection code/report | `code/inspect_multichange_pair.py`; `results/pair_01/inspection_report.txt`; `scans_spatial_overview.png`, `scans_overlay_alignment.png`, `ground_truth_change_labels.png`, `feature_distributions.png` in that folder |
| 2: bidirectional C2C and threshold sweep | Completed; saved clean metrics cross-checked against Stage 3 | `code/c2c_baseline.py`; `results/pair_01/c2c_baseline/threshold_metrics.csv`, `c2c_predictions.npz`, `baseline_report.txt`, and three diagnostic PNGs |
| 3: C2C robustness | Implemented; saved-result audit and final reporting completed with explicit methodological limits | `code/c2c_robustness.py`; six CSVs and `robustness_curves.png` in `results/pair_01/robustness_reviewed/`; use its audited `robustness_report_final.txt` for reporting |
| 4: M3C2-style | Audited; bounded clean and matched robustness study complete | `results/study_2026-10-25_v1/`; all-scene occlusion supplement; historical pilot retained |
| 5: compact Siamese PointNet-style model with NN prior | CUDA training and buffered calibration/test plus matched stresses complete | `results/study_2026-10-25_v1/learned/` |

Historical `results/pair_01/robustness/` is an earlier experiment set. Preserve it; use `robustness_reviewed/` for the current comparison. The original reviewed `robustness_report.txt` and source report writer retain stale prose. Rerunning the script overwrites its default results and can reproduce those inconsistencies; use a new output directory and separately review the writer before future regeneration.

The previously existing final report was preserved as `results/pair_01/robustness_reviewed/robustness_report_final.pre_audit_2026-10-25.txt`. `stage3_audit_manifest.json` in the same folder records checks and SHA-256 hashes verifying preservation of inspected source scripts, CSVs, historical reports and the robustness plot. Dataset payloads were not loaded or modified; only their headers and sizes were inspected.

## Actual C2C pilot baseline

Source of truth: `results/pair_01/c2c_baseline/threshold_metrics.csv`, `threshold_m=0.075`. This was the best macro F1 among the 11 saved thresholds on **the same pair**, not an independent test result.

| Quantity | Value |
|---|---:|
| Scan 0 / Scan 1 original point counts | 4,444,935 / 5,089,793 |
| Scan 0 removal precision / recall / F1 / changed IoU | 0.5450 / 0.6674 / 0.6000 / 0.4286 |
| Scan 1 addition precision / recall / F1 / changed IoU | 0.6064 / 0.7530 / 0.6718 / 0.5058 |
| Pooled three-class unchanged F1 | 0.9358 |
| Combined macro F1 / macro mIoU | **0.7359 / 0.6046** |
| Overall accuracy | 89.12% |

Combined macro F1 averages unchanged, removed and added class F1 over pooled directional observations. It is not the mean of removal/addition F1, and observations in the two scans are not unique physical sites.

## Stage 3 audit findings

All six reviewed CSVs were read: translation-only, rotation-only, combined registration, sparsity, noise and dual-protocol occlusion. There are 37 rows, including 32 with directional confusion counts. Independent checks of 552 metric values agree within absolute tolerance `1e-7` (maximum difference approximately `5.05e-9`). All 125 shared numeric values in the five standard clean-reference rows exactly match Stage 2 as parsed; zero-radius occlusion macro metrics also match. Occlusion combined metrics cannot be independently reconstructed because that CSV does not contain confusion counts. This is a saved-result audit, not a full-cloud rerun.

| Condition | Audited combined macro F1 |
|---|---:|
| Translation only: 5 / 10 / 20 / 50 cm | 0.7568 / 0.7222 / 0.4544 / 0.2620 |
| Rotation only: 0.5 / 1 / 2 degrees | 0.7677 / 0.5721 / 0.4092 |
| Combined: 50 cm + 2 degrees | 0.3028 |
| Sparsity: 10% of Scan 1 retained | 0.7166 |
| Noise: 7.5 cm standard deviation per coordinate | 0.6091 |
| Occlusion: 49.5481% of Scan 1 points removed, protocol A / B | 0.5101 / 0.6465 |

Corrected stale translation/rotation prose, unsupported causal claims, the prior final report's unverified translation vector, rotation displacement interpretation, and occlusion/noise explanations. Registration uses seed 42 and fixed axes/directions. In combined tests rotation consumes the first random draw, so translation uses a different direction from the isolated translation experiment. The main routine uses the Scan 1 **mean** as both rotation and deletion centre; the occlusion helper's median default is not used.

For the tested endpoints, the four-family ordering is registration, strict occlusion A, noise, then sparsity. Isolated translation is worse than combined registration at the largest tested stress. Under occlusion B, noise is worse than masked occlusion. This is a range- and protocol-dependent observation, not a universal robustness ranking.

Protocol A scores all Scan 0 points against deleted Scan 1, exposing false removals caused by missing data. Protocol B reuses the predictions and excludes Scan 0 points in the known synthetic blind cylinder. It changes evaluation support, not the detector, and does not establish mutual sensor visibility. The removed percentage measures point count rather than site area. The original six-panel plot visually agrees with saved trends/endpoints; its combined panel requires the paired rotation table, and its occlusion axis should be described as point loss.

## Environment and reusable code

The Stage 1 report records Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1, an RTX 4050 Laptop GPU and Open3D 0.20.0 in `C:\Users\prani\anaconda3\envs\dl_env`. The executable exists. Launch from this audit's WSL shell failed with a socket error, so current package/GPU execution was not reverified. Standard-library host Python handled only CSV/report processing. No environment, package or CUDA changes were made. Resume experiments in the existing Windows `dl_env`.

Use `load_point_cloud_ply` from `code/c2c_baseline.py`: these PLY headers use `element params` with XYZ, intensity and floating `cd_type`; the standard Open3D PLY reader is unsuitable for these files. Label meaning is inherited from the verified Stage 1/project context: `0` unchanged; `1` removed in Scan 0 and added in Scan 1. The upstream label convention was not independently rechecked in this audit.

Reuse the baseline metric functions and cached C2C predictions/distances. The NPZ writer defines `distances_01`, `distances_10`, `gt0`, `gt1`, `best_threshold`, `pred0_best`, `pred1_best`; the payload was not reloaded here. Use the existing Stage 3 perturbation helpers, with explicit awareness of seeds, directions, one-sided perturbations and support masks. The early `inspect_dataset.py`, `compare_scans.py` and `visualize_pointcloud.py` target ConSLAM Sequence 3 and are qualitative utilities, not quantitative MultiChange3D baselines.

## Limits and outstanding work

Only one labelled pair is evaluated; the threshold was selected on that pair. There are no repeated perturbation trials, uncertainty intervals across seeds, independent scenes, or physically derived visibility masks. Scan 1 alone is degraded. Small perturbation improvements are specific observations, not evidence of general benefit from registration error. Point removal changes class mix and pooled epoch weights. Historical scanner-specific accuracy claims and runtime statements were not validated by this audit. C2C proximity on unchanged samples is not an independent registration-error calibration.

M3C2-style local normals, multiscale projection, conditional uncertainty decisions, explicit invalid support, synthetic validation and a frozen clean pilot comparison are now implemented (see latest update above). Remaining Stage 4 work is failure/geometry review, independent reference-output validation if available, and matched robustness comparisons using common indices and reported coverage. ConSLAM has no equivalent labels for the quantitative evaluation. Learning-based implementation/training and its data split remain outstanding, time permitting; no such training is authorized by this task.

## Next work

The requested bounded pilot is complete. Review its false positives, class-dependent abstentions and cylinder-length sensitivity before further experiments. For future matched robustness work follow `results/stage4_m3c2_plan.md`, reuse the existing perturbation helpers/seeds/masks and recompute normals on perturbed source geometry. Keep C2C fixed at `0.075 m`; retain the saved M3C2 primary configuration unless a new, explicitly versioned exploratory study documents a change. Do not promote a score-selected pilot row to a held-out result.

Aim to finish matched geometric comparisons by 29 October, figures/report by 30 October, and presentation preparation by 1 November. Treat these as proposed work allocations. Stage 5 is optional only after the comparison and presentation materials are complete. Preserve all original data, code and results throughout.
