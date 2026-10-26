# Final project status

25 October 2026, Asia/Kolkata. Final presentation: **2 November 2026**.

The requested submission is assembled under `D:\CV_Project\final_submission`. It completes a bounded empirical comparison of C2C, an audited M3C2-style implementation and a compact learned paired-patch baseline, including representative matched robustness experiments and qualified ConSLAM context. It does not claim reference-network reproduction, cross-scene generalization, exact CloudCompare equivalence or certified construction progress.

## Completed stages and evidence

| Stage | Verified state | Authoritative evidence |
|---|---|---|
| Environment | Existing native Windows dl_env, CUDA and RTX 4050 usable; no reinstallation | Package versions in submission; actual CUDA training log and checkpoint |
| 1 dataset inspection | Existing pair_01 reused; official 0/1 labels and classical binary metric conventions rechecked | `results/data_sources_and_access.md`; original inspection report; study dataset SHA-256/schema manifest |
| 2 C2C | Historical full-pair sweep preserved; fixed 0.075 m used on every new sample/condition | Historical CSV/NPZ and fresh same-index cache checks |
| 3 historical robustness | Existing final report reviewed; 416 metric values independently recalculated from 32 confusion-count rows; max error <5e-9 | `results/master_audit_2026-10-25/audit.json`; existing six CSVs and final report |
| 4 M3C2-style | 19 geometry tests, six independent streamed real-cylinder checks; no geometry bug found in tested paths | `code/test_m3c2_synthetic.py`, audit JSON; earlier pilot retained |
| 4 bounded comparison | Calibration support checked, configuration frozen; 4,000 original spatial-test cores per epoch, full-cloud geometry support; 11 clean/stress conditions | `results/study_2026-10-25_v1/geometry_frozen.json`, split config, per-condition measurements/metrics/checkpoints |
| Occlusion supplement | Main stripe misses blind cylinders, explicitly disclosed; added geometric-only all-scene 2,000-core/epoch study, strict A and masked B | `results/occlusion_supplement_2026-10-25_v1/` |
| 5 learning | Compact shared PointNet-style encoder with NN prior trained on CUDA; 16,000 training cores, weighted BCE, 30 epochs; calibration selects epoch 1 and threshold 0.7 before test inference | `learned/best_checkpoint.pt`, frozen config, archived training source, logs and per-condition predictions |
| Analysis | 39 common-support rows recomputed from saved predictions; longitudinal deltas use identical clean/current valid retained indices | `analysis/metric_audit.json`, `matched_clean_deltas.csv`, all-scope result table |
| ConSLAM | Existing assets only; 6,014 PCDs, 527 available pose files, 520 exact matches; first two matched frames and bounded 50k-point TLS sample | `results/conslam_qualitative_2026-10-25_v2/inspection.json` and figures |
| Submission | Six-page PDF and editable source; 11-slide editable PowerPoint, eight-minute notes split evenly; code, configs, weights, tables, plots and offline 3D demo | `final_submission/`; final delivery verification JSON |

## Actual clean comparison

These rows share **2,563 Scan 0 and 2,787 Scan 1 valid cores**, 66.875% of 8,000 requested directional observations.

| Method | Pooled macro F1 | Macro mIoU | Mean removal/addition F1 | Detector unknown |
|---|---:|---:|---:|---:|
| C2C | 0.4118 | 0.3340 | 0.1699 | 0% |
| Siamese PointNet + NN | 0.3817 | 0.3091 | 0.1340 | 0% |
| M3C2-style | 0.2342 | 0.1561 | 0.0654 | 33.125% |

C2C and learned common-support exclusions are not detector abstentions. Their all-sampled macro F1 values are 0.4457 and 0.3936, kept separately. Conservative correct-valid/all-core accuracy on matched support is 54.48%, 52.48% and 27.61% respectively. Changed-class valid coverage for M3C2 is only 24.3% removals and 31.5% additions, so high valid-only recall is selective.

Historical full-pair C2C macro F1 0.7359/mIoU 0.6046 and all-scene pilot M3C2/C2C macro F1 0.3529/0.6274 have distinct distributions and selection histories. The master summary CSV labels these scopes and does not rank them together.

At 20 cm translation, longitudinal macro-F1 deltas on identical intersected support are -0.1045 C2C, -0.1386 M3C2-style and -0.0446 learned. At 10% retention, they are -0.0052, +0.0038 and -0.0956. These are single seeded simulations, with changing class counts, not universal method rankings. The full condition table and coverage are bundled.

## Tests and delivery validation

- **25 tests**: 19 geometry/uncertainty/abstention tests, four learning input/trainability tests and two severity-plot ordering regressions.
- A plotting bug connected retention levels in the wrong order. It was fixed in both scientific figures and native PowerPoint charts; measured values and metric CSVs did not change.
- Original data and 27 protected C2C/Stage 3 files remain byte-identical to the saved pre-pilot manifest.
- Every PDF page and every final slide was rendered and visually reviewed. PowerPoint package/layout/font/native-chart workbook/import validation passes. Charts use six-decimal display snapshots; full-precision metrics remain in CSVs. Native Microsoft PowerPoint execution is not claimed.
- The offline demo runs a live 0.12 m synthetic normal-offset check and displays measured original-core predictions. In an isolated headless installed Chrome session, method/epoch controls, common-support masking, rotation, zoom and reset passed without JavaScript page errors. The in-app/browser-tool kernels were unavailable; the headless test is the actual browser evidence.
- See `final_submission/artifacts/audit/final_verification.json` for the final requirement checks, source/weight/artifact hashes and file inventories. Private intermediate renders remain under `results/submission_build/` and are not the deliverables.

## Outputs and one demo command

Main files:

- `final_submission/report/project_report.pdf`
- `final_submission/report/project_report.md`
- `final_submission/report/build_report.py`
- `final_submission/presentation/project_presentation.pptx`
- `final_submission/presentation/build_presentation.mjs`
- `final_submission/presentation/speaker_notes.md`
- `final_submission/artifacts/all_evaluated_results.csv`
- `final_submission/artifacts/figures/`
- `final_submission/artifacts/learned/best_checkpoint.pt`
- `final_submission/code/`
- `final_submission/README.md` and `reproduction.md`

From `D:\CV_Project`:

```powershell
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' final_submission/code/demo.py
```

Then open `final_submission/demo/change_demo.html`. The demo needs no original dataset, GPU or network and does not pretend that saved experimental predictions are recomputed live.

## Data actually used and access limits

No new dataset downloads were performed. MultiChange3D pair_01 was already present and is the only labelled quantitative pair. The five missing construction pair folders and optional Open Space MLS official links redirected to Microsoft sign-in. Storage was sufficient; access was not available and was not bypassed. No cross-pair/cross-scene score is claimed. Existing local ConSLAM Sequence 3 assets provide qualitative context only, not per-point change ground truth or a cross-month comparison. Dataset/software citations and academic/noncommercial licensing limits are recorded; full clouds are excluded from the small submission bundle.

## Remaining limitations and deliberate scope decisions

1. One pair/scene, one stress/training seed, spatially balanced sampling and historical same-pair threshold exposure. No confidence intervals across acquisitions or generalization claim.
2. M3C2-style source conventions checked, but no independent CloudCompare output comparison; zero modelled registration uncertainty, correlated points, edge normals and mixed cylinder surfaces limit nominal LoD interpretation.
3. Learned calibration macro F1 0.7649 versus 0.3936 on spatial-test all-core support, with later calibration-loss growth. Its three-method common-support test score is 0.3817, labelled separately. No official reference-network reproduction; their extra dependency stack was inspected but not installed.
4. Main test stripe lies outside both blind cylinders. The separate all-scene geometric supplement supplies deletion evidence; it is not passed off as learned held-out data.
5. Legacy Stage 3 float32 mean center differs from a float64 mean by about 0.27 m in X. Exact legacy arithmetic remains frozen for method comparability; a future changed center requires a new protocol.
6. ConSLAM TLS-to-normalized LiDAR frame was not independently validated; only adjacent-frame qualitative proximity and site context are reported. Missing counterpart points never become official change annotations.
7. No full-cloud M3C2 sweep, additional 20,000-core expansion, BIM quantity estimation, DBSCAN postprocessing or semantic completion estimator. The bounded 11-condition study prioritizes methodology and finished submission over these extensions.

All required core work is represented by actual outputs. Conditional download and reference-validation limits remain explicit rather than being described as completed experiments.
