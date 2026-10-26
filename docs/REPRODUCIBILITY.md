# Reproducibility and safe execution

## Inspect results and verify the uploaded files

```powershell
python tools/verify_publication.py
```

This standard-library-only command resolves paths relative to the clone and checks the publication file manifest. It does not recreate excluded caches or re-execute historical research audits. `.gitattributes` preserves exact file bytes for historical SHA-256 values.

## Environment

Python 3.10.20 was used. For a new Windows/Linux CPU environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s code -p "test_*.py" -v
```

On Linux/macOS, activation is `source .venv/bin/activate`; select a suitable platform wheel from [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). For compatible Windows/Linux NVIDIA hardware, the research environment used this CUDA alternative:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Choose CPU or CUDA, not both commands sequentially. Torchvision/torchaudio are not needed. The 83 research tests use synthetic arrays/temp files and can run without datasets. No reinstall of the original working `dl_env` is necessary.

Optional ConSLAM/interactive viewers need `python -m pip install -r requirements-qualitative.txt`. Report/slide authoring dependencies are not required for tests or result inspection.

## Historical offline viewer

```powershell
python final_submission/code/demo.py --output results/demo_local_new
```

Open `results/demo_local_new/change_demo.html`. Bundled checksums and a real synthetic PCA/cylinder offset are checked. Real-scene content is saved earlier predictions, not current V2 running live on new data.

## Data

Follow [DATASETS.md](DATASETS.md); retain both ZIPs and extracted labelled PLYs. Root-relative `dataset/` is the supported common layout. Preserve original bytes and labels. Historical `D:\CV_Project` paths record provenance; they are not a portable directory requirement.

## Full training is an advanced archival workflow

The compact upload retains current source, protocols, choices, summary measurements and six selected V2 checkpoints. It omits datasets, bulk .npy/.npz arrays, most checkpoints, duplicate source snapshots, ZIP bundles, and oversized tables/images. The small historical demo dependency bundle is retained. Every omission is recorded in docs/publication_exclusions.json.

- Completion markers remain even when their cache arrays are omitted. Do not run drivers against archived defaults and assume caches exist.
- A fresh `--out` alone is insufficient: modules still reference canonical earlier study folders via `BASE`, and manifests contain original absolute paths.
- Original V2 training/inference contains explicit CUDA calls. Later CPU follow-ups do not make archived V2 automatically CPU-capable.
- Preserve archived sources/manifests. A versioned reconstruction adapter and explicit dependency/path reconciliation are needed before a full new-machine rerun.

Dependency chain:

```text
six original labelled pairs + ZIPs
 -> original-point audit and outer spatial protocols
 -> nested inner splits and frozen geometry baselines
 -> V2 feature caches / assembled banks
 -> neural fits and validation-frozen thresholds
 -> identical-core clean/stress features and predictions
 -> analysis and independent verification
 -> optional frozen external diagnostic

V3 fresh banks -> hierarchy/focal pilot -> calibration/refinement
               -> match distributions -> fusion/intensity checks
```

Relevant archival drivers: `run_multipair_study.py`, `run_nested_multipair.py`, `run_learning_v2.py`, `run_learning_v3.py`. Inspect their `BASE`, `DEFAULT` and dependencies first. The initial publication does not pretend this is a turnkey full trainer.

Run large geometry and numerical jobs sequentially on memory-limited hosts. Later GPU attempts hit Windows commit/backing-memory limits; successful follow-up fits used CPU.

## Rebuild this GitHub snapshot from the full original workspace

```powershell
python tools/prepare_github_snapshot.py --destination .publish/Visioners_2.0_compact
```

This publication utility copies a compact allowed file set into a separate existing Git checkout, scans for credential patterns, and writes checksums/exclusions. It leaves source research outputs and datasets intact and refuses conflicting destination overwrites. Its default profile excludes bulk binary artifacts, keeps six validation-selected V2 weights and the small demo dependencies, and caps ordinary files at 2 MiB. Git commit/push are separate explicit actions.

Included summaries of failed experiments and earlier reports are historical evidence, not the current best model. Historical full-archive SHA audits will reference omitted binaries/source copies; use publication verification for the compact clone.
