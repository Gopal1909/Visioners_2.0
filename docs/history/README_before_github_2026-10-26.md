# Robust 3D Change Detection for Construction Progress Monitoring

CV5570 semester project by Vishwanath Venugopal (CE23B124) and Shaik Koduru Asif (CE23B114). Presentation: 2 November 2026.

The project compares C2C, an audited M3C2-style detector and a compact Siamese PointNet-style network with an NN prior. It studies registration error, density loss, noise and occlusion. Outputs localize **candidate geometric changes**; they do not certify construction quantities, structural completion or schedule progress.

## Current submission

Start with `final_submission/README.md`. The folder contains a six-page academic PDF and editable manuscript/build source, an 11-slide editable PowerPoint with eight-minute speaker notes, measured result tables and figures, trained checkpoint, code, and an offline interactive 3D demo.

The new buffered spatial-test comparison uses 4,000 original cores per epoch, with complete input clouds as geometric support. All three methods score the same 5,350 valid cores out of 8,000 requested observations:

| Method | Pooled three-class macro F1 | Macro mIoU | Detector unknown rate |
|---|---:|---:|---:|
| C2C | 0.4118 | 0.3340 | 0% |
| Siamese PointNet + NN | 0.3817 | 0.3091 | 0% |
| M3C2-style | 0.2342 | 0.1561 | 33.125% |

Common evaluation coverage is 66.875%; C2C and learned mask exclusions are not detector abstentions. Their all-sampled results are retained separately. Historical full-pair C2C (macro F1 0.7359) and earlier all-scene pilot values belong to different scopes and cannot be ranked against this table. The study does not support M3C2 superiority.

## Existing environment

Use native Windows PowerShell and the existing interpreter. No environment reinstall is needed.

```powershell
$py = 'C:\Users\prani\anaconda3\envs\dl_env\python.exe'
& $py -c "import torch,numpy,scipy,plyfile,psutil; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(psutil.virtual_memory())"
```

Verified Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.5.1+cu121, CUDA, RTX 4050 Laptop GPU (6 GB), Open3D 0.20.0, plyfile, pandas, psutil and matplotlib. System RAM is 15.6 GiB. About 194 GiB disk and 4.8 GiB RAM were available at the start. Exact package versions accompany the submission. CPU geometry peaks near 0.9 GiB process RSS; learned inference peaks near 1.6 GiB in its separate process. Measured training GPU allocation peaked near 89 MiB; feature preprocessing is separate. Do not run several memory-heavy jobs concurrently.

## Data and licensing

Expected default data location:

```text
dataset/MultiChange3D/BikeParkingConstruction/pair_01/pointCloud0.ply
dataset/MultiChange3D/BikeParkingConstruction/pair_01/pointCloud1.ply
dataset/Sequence 3/Copy of lidar-pcd/lidar-pcd/*.pcd
dataset/Sequence 3/Copy of lidar_pose_normalized/normalized/*.txt
dataset/Sequence 3/Copy of groundtruth_scan.ply
```

The PLY element is `params`, with XYZ, intensity and `cd_type`. `cd_type=0` means unchanged and `1` means changed; changed Scan 0 points are removals and changed Scan 1 points are additions. Official preparation and classical metric scripts were checked. Original files are never rewritten. New study loaders preserve all original record fields. For an external data directory, set `$env:CV_PROJECT_DATA_ROOT = 'D:\CV_Project\dataset'` before running `study_pipeline.py` or `siamese_pointnet.py`.

Only existing pair_01 and local ConSLAM assets were used. **No additional datasets were downloaded.** Official links for missing construction pairs pair_02/03/12/13/23 and optional Open Space MLS redirected to Microsoft sign-in; access was not bypassed. See `results/data_sources_and_access.md` for official URLs, schema verification, citations and access limits. MultiChange3D data are CC BY-NC-SA 4.0; its code is MIT. ConSLAM software is academic-use-only; broader data redistribution rights are not inferred. Full original datasets are excluded from the submission.

## Reproduce or resume the stages

Run commands from `D:\CV_Project`. All new experiment commands use versioned output folders and resume completed checkpoints. To obtain a fresh rerun, pass the **same new `--out` directory** to prepare, geometry, training, evaluation and analysis. The historical C2C archive and geometric occlusion supplement remain available for audit/aggregation. Do not rerun legacy scripts with their default output directories, because their old writers overwrite historical outputs.

```powershell
# Geometry, learning invariants and plotting regressions: 25 tests
& $py -m unittest discover -s code -p 'test_*.py' -v

# Stage 1 historical inspection, if needed (existing report is retained)
& $py code/inspect_multichange_pair.py --help

# Stage 2 historical baseline: choose a NEW output directory if rerunning
& $py code/c2c_baseline.py --output_dir results/pair_01/c2c_reproduction_new

# Stage 3 saved-result audit and independent full-support cylinder spot checks
# --out must not already exist
& $py code/project_audit.py --out results/audit_new

# Stages 4 and 5: existing study checkpoints are retained
& $py code/study_pipeline.py prepare
& $py code/study_pipeline.py geometry
& $py code/occlusion_supplement.py
& $py code/siamese_pointnet.py train
& $py code/siamese_pointnet.py evaluate
& $py code/analyze_study.py

# Example of a fresh spatial study, preserving the existing run
& $py code/study_pipeline.py prepare --out results/reproduction_new
& $py code/study_pipeline.py geometry --out results/reproduction_new
& $py code/siamese_pointnet.py train --out results/reproduction_new
& $py code/siamese_pointnet.py evaluate --out results/reproduction_new
& $py code/analyze_study.py --out results/reproduction_new

# ConSLAM qualitative script, no change accuracy claims
& $py code/conslam_qualitative.py

# Self-contained final demo: no original dataset, GPU, network or retraining
& $py final_submission/code/demo.py
```

Open `final_submission/demo/change_demo.html` in a browser. The command verifies bundled checksums, runs a live synthetic PCA/cylinder offset check, and writes the offline viewer of **saved measured predictions**. It is not a live full-dataset detector. Method/epoch selection, support masking, rotation, zoom and reset were tested in an isolated headless installed Chrome session.

## Stages and authoritative outputs

| Stage | Inputs and method | Verified outputs |
|---|---|---|
| 1 | pair_01 original PLYs and labels | `results/pair_01/inspection_report.txt` and existing figures |
| 2 | bidirectional full-pair C2C threshold sweep | `results/pair_01/c2c_baseline/`; historical same-pair exploratory selection |
| 3 | controlled one-sided C2C stresses | `results/pair_01/robustness_reviewed/robustness_report_final.txt` and six CSVs; no unnecessary full rerun |
| 4 pilot | source normals, common cylinders, conditional LoD, explicit unknowns | `results/pair_01/m3c2/pilot_2026-10-25_v2/`; prior run preserved |
| 4 bounded study | frozen calibration configuration and 11 clean/stress conditions | `results/study_2026-10-25_v1/geometry/`, `geometry_metrics.csv` |
| Occlusion supplement | original whole-scene pilot indices, clean plus 4.6/7.6 m deletion, protocols A/B | `results/occlusion_supplement_2026-10-25_v1/`; geometric-only exploratory scope |
| 5 | shared paired-patch encoder, clean training and buffered calibration | `results/study_2026-10-25_v1/learned/`; 30 epochs, best checkpoint, frozen threshold and all stress predictions |
| Final analysis | saved per-core predictions on identical support | `results/study_2026-10-25_v1/analysis/`; all-scope tables, class coverage, matched-clean deltas and figures |
| ConSLAM | existing timestamp-matched PCD/poses, memory-mapped TLS sample | `results/conslam_qualitative_2026-10-25_v2/`; within one acquisition only |

The initial wider-gap ConSLAM diagnostic is retained under v1; reduced overlap limits its interpretation. A plotting regression in retention-level ordering was corrected without changing measurements or scores; two tests prevent its recurrence.

## Splits, seeds and fixed parameters

The geometry-only X regions are train [27.7864,40.2857], calibration [42.5077,45.2854], and test [47.5074,55.5625] m. Both epochs appear in each split. Gaps of 2.2221 m exceed twice the largest clean M3C2 support halo and the learned patch diameter. Cores are balanced over occupied 2 m tiles, using `20261025 + 100*split_order + scan` seeds. This is a same-scene spatial holdout, not a cross-scene test. Prior pilot inspection and historical C2C threshold selection are disclosed test exposure.

C2C remains NN distance >0.075 m. M3C2 uses normal radii 0.10/0.20/0.40 m, fixed quality/stability checks, projection radius 0.10 m, axial half-length 1.0 m, at least 10 points per epoch, effect floor 0.02 m and separate registration error 0 m. CloudCompare v2.13.2 revision 49dbbb6 mean-mode conventions were checked; application equivalence was not established. A historical NN median is not registration calibration.

The learned model uses masked paired 128-point patches within 0.6 m, relative XYZ and an NN prior, shared 3-32-64 MLP, masked mean/max pooling and 387-64-32-1 head. Labels are supervision only. Seed 42, weighted BCE (positive weight 3.6976), AdamW 0.001, batch 128 and 30 fixed epochs. Calibration loss selects epoch 1; calibration F1 selects threshold 0.7. The actual training-source hash and archived source are included; the later source change only records evaluation metadata. No test/stress retuning occurs.

Stresses reuse Stage 3 helpers and seed 42 exactly. The stored legacy float32 mean center differs from a float64 mean by about 0.27 m in X; it is retained for comparability, not described as an independently calibrated center. Only Scan 1 changes. Original indices are maintained through retention/deletion; normals are recomputed on actual perturbed source geometry. The main spatial stripe is outside both deletion cylinders, so the separate all-scene geometric supplement supplies meaningful occlusion evidence.

## Metric definitions and limitations

Binary precision, recall, F1 and changed IoU come from TP/FP/FN/TN at original indices. Directional macro F1 averages removal/addition F1. Pooled three-class macro F1 and mIoU average unchanged/removed/added class values over the two directional observations. This pooling is a project-defined directional summary, distinct from the upstream geometric nearest-neighbour dual-direction protocol. Absent classes contribute zero under the existing epsilon convention. No prediction interpolation inflates sampled-core results into full-cloud scores.

Unknown is -1, with saved validity/reason masks. Every matched row uses identical per-direction original indices for all methods. Conservative correct-valid/all-requested accuracy counts exclusions as errors. Longitudinal deltas additionally intersect clean/current validity and retained indices. All-core rows for C2C and the learned model remain separate. Spatially balanced and class-dependent valid sampling prevents extrapolating these scores to full-pair accuracy. Registration uncertainty is not calibrated; the learned model overfits and shifts spatially; there is one pair, scene and training/stress seed. ConSLAM has no accuracy labels. Geometric changes require semantics or BIM quantities for a progress claim.

## Submission authoring

`code/build_report.py` produces Markdown and a six-page PDF from bundled metrics using reportlab. The preferred authoring runtime is the Codex bundled Python. `final_submission/presentation/build_presentation.mjs` uses the bundled Artifact Tool and exports native editable tables/charts with embedded workbooks. A fresh destination and receipt are required for a rebuild; validated presentation files are deliberately not overwritten. The browser/desktop tool kernels were unavailable, so the demo was tested through the bundled Playwright library in an isolated installed Chrome session. PDF pages and every final slide were rendered and visually reviewed.

See `results/final_status.md` for the completion audit and remaining limitations.
