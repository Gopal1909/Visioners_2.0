# Robust 3D Change Detection for Construction Progress Monitoring

CV5570 semester project. Original project attribution: Vishwanath Venugopal (CE23B124) and Shaik Koduru Asif (CE23B114).

We identify unchanged, candidate added, and candidate removed geometry between construction-site scans. Geometric change alone does not certify completed work, installed quantities, or schedule progress.

**Status: 26 October 2026. V2 remains the retained workflow.** Later candidates, including unsuccessful experiments, are preserved. This README supersedes the obsolete one-pair availability and results descriptions in historical documents.

## Start here

- [Dataset downloads and exact layout](docs/DATASETS.md)
- [Installation and reproducibility](docs/REPRODUCIBILITY.md)
- [Completed work and current improvement plan](docs/PROJECT_STATUS.md)
- [Current V2 comparison](results/learning_v2_coverage_2026-10-25_v1/analysis/clean_comparison.csv)
- [Latest verified experiment](results/construction_intensity_2026-10-26_v1/final_review.json)

## What is included

This compact upload includes current code, unit tests, six validation-selected V2 checkpoints, compact metrics, figures, configurations, audit receipts, and existing PDF/presentation/demo materials. Datasets, bulk prediction/feature arrays, redundant checkpoints, duplicate source snapshots and ZIP bundles are excluded. A 2 MiB file cap applies except to selected checkpoints and existing document/demo dependencies. The complete research archive remains intact locally.

The generated `docs/publication_manifest.json` records included-file SHA-256 values. `docs/publication_exclusions.json` lists omitted files and why they were excluded. Dataset-derived outputs retain applicable attribution and terms: [third-party notices](THIRD_PARTY_NOTICES.md).

## Current construction results

Same-site spatial comparisons across six BikeParkingConstruction pairs and five buffered folds; identical sampled original cores, complete clouds as geometric support:

| Method | Mean pair/direction binary macro-F1 |
|---|---:|
| Validation-calibrated C2C | 0.5486 |
| Fixed 7.5 cm C2C | 0.5659 |
| M3C2-style implementation | 0.4142 |
| Previous validation-selected learning model | 0.6212 |
| **V2 validation-selected workflow** | **0.6537** |

V2 also has **0.7405 globally pooled unchanged/removed/added macro-F1**, with different weighting. These are F1 scores, not ordinary accuracy. Unchanged-only false alarms fell from 12.45% for calibrated C2C to 5.64% for V2. M3C2-style has an 18.67% unknown rate on full requested support; reference-software equivalence is not claimed.

The separate user-supplied Open_space diagnostic scored V2 0.7857 and C2C 0.8596. It was not used to fit the retained model and is not another construction-site benchmark. See the dataset guide for a catalog-versus-local-size discrepancy.

## Quick start without datasets

Python 3.10 was used. For Windows/Linux CPU, install the matching PyTorch wheel separately from numerical dependencies:

```powershell
git clone https://github.com/Gopal1909/Visioners_2.0.git
cd Visioners_2.0
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tools/verify_publication.py
python -m unittest discover -s code -p "test_*.py" -v
```

The latest research suite passed **83 tests**, using synthetic arrays/temp files rather than original datasets. CPU suffices for these checks. For another platform or CUDA, use [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). The research GPU environment used PyTorch 2.5.1+cu121.

The existing **historical** offline viewer can be exported separately:

```powershell
python final_submission/code/demo.py --output results/demo_local_new
```

Open `results/demo_local_new/change_demo.html`. It uses earlier saved measured predictions and a synthetic geometry check. Its displayed results predate V2; it is not live full-cloud inference.

## What to download

Required: [MultiChange3D](https://3dom-fbk.github.io/multichange3d/) -> **MLS -> Bike Parking Construction**, all six archives `pair_01`, `pair_02`, `pair_03`, `pair_12`, `pair_13`, `pair_23`. Keep ZIPs and extract both labelled PLYs per pair.

Open_space is optional for the external diagnostic. RGBD Office was downloaded locally but unused by the retained workflow. ConSLAM Sequence 3 is optional for qualitative inspection; all sequences/ROS bags are unnecessary for this project. See [DATASETS.md](docs/DATASETS.md).

## Training and interpretation

**This is an evidence-and-code snapshot, not a fully cached resumable trainer.** Completion receipts remain while large cache payloads are excluded. Historical manifests contain original `D:\CV_Project` paths. Do not run archived drivers against their default folders or rewrite provenance to make old audits appear portable. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the dependency chain and limitations.

Six pairs share four epochs from one construction site. Earlier test findings informed later architectural work, and validation has been repeatedly reused. Training/scoring use sampled cores, not every raw point. Added/removed labels come from earlier/later scan direction after binary detection.

Region exports contain 191 sampled candidate components; 83 have no changed-labelled sampled points. This is a point-label audit, not object-level precision. Work-item identities, quantities, and completion percentages remain unverified.

## Next improvements

1. Independently review construction regions: structural work, temporary activity, coverage ambiguity, and uncertainty.
2. Run a controlled, replicated V2-versus-hierarchy comparison with matching training settings.
3. Investigate acquisition metadata and coverage uncertainty.
4. Evaluate dense predictions in a predefined validation area.
5. Add temporal persistence and work-item/BIM association after reliable region detection.

Compact summaries of historical/rejected candidates are included; their complete binaries and source snapshots remain in the local archive. No reports or presentations were regenerated for this publication.
