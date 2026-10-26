# CV5570 final submission

**Robust 3D Change Detection for Construction Progress Monitoring**

Vishwanath Venugopal CE23B124 and Shaik Koduru Asif CE23B114. Presentation 2 November 2026.

## Start here

- `report/project_report.pdf`: six-page academic report, with editable `project_report.md` and `build_report.py`.
- `presentation/project_presentation.pptx`: 11 editable slides with native charts/tables, embedded data workbooks and speaker notes. `speaker_notes.md` divides eight minutes evenly between both members. `build_presentation.mjs` preserves the editable authoring source.
- `artifacts/all_evaluated_results.csv`: the single summary table, with evaluation scopes kept separate. `results_summary.md` provides a readable overview.
- `artifacts/figures/`: real measured change maps, robustness, support, training and ConSLAM figures.
- `artifacts/learned/best_checkpoint.pt`: the actual selected compact model checkpoint, configuration, threshold, training history and archived training source.
- `code/`: original reusable functions and the added audited experiment, learning, analysis, test and demo scripts.
- `reproduction.md`, `environment_versions.json` and `requirements.txt`: commands, dependencies, data/split/seeds and limitations.
- `artifacts/audit/`: numeric, preservation, presentation and final-delivery verification evidence.

## Offline demo

From `D:\CV_Project` in Windows PowerShell:

```powershell
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' final_submission/code/demo.py
```

Or, from inside this submission folder:

```powershell
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' code/demo.py
```

Open `demo/change_demo.html`. It works offline without the original point clouds, GPU, model downloads or retraining. The command checks artifact hashes and performs a **live synthetic cylinder/PCA test**, then displays **saved real experiment predictions**. Select the method and epoch, toggle common support, drag to rotate and scroll to zoom. Gray means M3C2 unknown or evaluation exclusion. The viewer does not run a full-cloud detector or infer construction completion.

## Main measured comparison

On identical clean spatial-test support (5,350 valid cores out of 8,000 requested):

| Method | Macro F1 | Macro mIoU | Detector unknown |
|---|---:|---:|---:|
| C2C | 0.4118 | 0.3340 | 0% |
| Siamese PointNet + NN | 0.3817 | 0.3091 | 0% |
| M3C2-style | 0.2342 | 0.1561 | 33.125% |

Common evaluation coverage is 66.875%. These same-scene spatial-core results are not directly comparable to historical full-pair or all-scene pilot scores. C2C and learned all-core rows remain separate. The learned baseline is a compact course implementation with an NN prior, not a reproduction of a published reference network. M3C2-style reference-software equivalence is unverified.

Only existing MultiChange3D pair_01 and local ConSLAM Sequence 3 assets were used. No additional data were downloaded: the official missing-pair and optional scene links redirected to sign-in. No cross-pair/cross-scene experiment or ConSLAM accuracy result is claimed. The bundle excludes original full datasets; see `artifacts/data_sources_and_access.md` for sources, licenses and citations.

The full working project remains in `D:\CV_Project`. Detailed raw per-core stress measurements and prediction checkpoints remain in its versioned `results/study_2026-10-25_v1/` directory; this small bundle includes representative cores, aggregate evidence and reproducible code.
