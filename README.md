# Robust 3D Change Detection for Construction Progress Monitoring

**CV5570 Semester Research Project  •  Indian Institute of Technology Madras**  
**Course:** Computer Vision for Construction Engineering (November 2026)  
**Authors:** Vishwanath Venugopal (CE23B124) and Shaik Koduru Asif (CE23B114)  
**Status:** **COMPLETE & FULLY VERIFIED (V2 Retained Workflow & Complete Deliverables)**  

---

## Executive Summary & Core Results

This project addresses automated construction progress monitoring across multi-temporal 3D point cloud scans. In real-world civil engineering environments, LiDAR scans suffer from sensor range noise, registration misalignment (1–10 cm), steep point density gradients, and view occlusions. Naive proximity thresholding (such as Cloud-to-Cloud distance) frequently confounds these sensor artifacts with structural construction progress.

We formulate a dual-dataset benchmark protocol:
1. **MultiChange3D Bike Parking Construction Scene** (MLS, 4 epochs, 6 pairs, ~5M points per scan) with verified per-point ground truth ($cd\_type \in \{0, 1\}$) for quantitative evaluation across 30 pair-fold studies.
2. **ConSLAM Sequence 3** handheld mobile LiDAR sweeps and 104.5M-point TLS reference for real-world mobile robotics qualitative field validation.

### Clean Construction Site Benchmark (30 Pair-Fold Studies)
Evaluated with **5-fold buffered spatial cross-validation** (2.22 m buffer gaps) across all 6 construction pairs:

| Method | Binary Macro F1 | Directional Changed F1 | Detector Unknown Rate | Coverage |
|---|---:|---:|---:|---:|
| **V2 Multiscale Validation-Selected (Ours)** | **0.6537** | **0.3900** | **0.0%** | **100.0%** |
| Multiscale Robust (seed 42) | 0.6555 | 0.3970 | 0.0% | 100.0% |
| Previous Geometry Boosted | 0.6184 | 0.3432 | 0.0% | 100.0% |
| Cloud-to-Cloud (C2C Legacy 0.075m) | 0.5659 | 0.3000 | 0.0% | 100.0% |
| Cloud-to-Cloud (C2C Calibrated) | 0.5486 | 0.2411 | 0.0% | 100.0% |
| M3C2-Style (Calibrated Cylinders) | 0.4142 | 0.0348 | 18.67% | 81.33% |
| M3C2-Style (Legacy Parameters) | 0.2627 | 0.1632 | 26.61% | 73.39% |

*Key Takeaway:* Our **V2 Multiscale Cross-Scan Network** outperforms classical C2C by **+8.78 percentage points** in Macro F1 and **+9.00 points** in Changed F1, while M3C2-style suffers from massive abstentions (26.6% unknowns) due to non-planar structural edges (rebar, railings, scaffolding).

### Field Stress Robustness (318 Degradation Conditions)
Under **Combined Large Field Stress** (20 cm translation drift, range noise, and decimation):
- **C2C collapses catastrophically to 0.4181 Macro F1** (-14.78 point drop) because shifting static walls exceeds 7.5 cm everywhere, falsely flagging the entire site as changed.
- **V2 Multiscale Robust Network retains 0.6176 Macro F1** (only -3.79 point drop), establishing a massive **+20.0 percentage point lead** over C2C.

### Physical Construction Work-Item Takeoff
Grouped positive predicted change points into **191 discrete 3D construction change regions** (80 candidate additions, 111 candidate removals) with 3D oriented bounding boxes and volumes, providing the computational backbone for 4D BIM Earned Value Management (EVM) and safety compliance.

---

## Primary Project Deliverables

All final deliverables are fully authored, compiled, and verified in `final_submission/`:

- **Final Research Report (PDF):** [`final_submission/report/project_report.pdf`](final_submission/report/project_report.pdf) — 6-page conference-style two-column research paper with embedded high-resolution figures, tables, and citations.
- **Final Research Report (Markdown):** [`final_submission/report/project_report.md`](final_submission/report/project_report.md)
- **Master Presentation Deck (.pptx):** [`final_submission/presentation/project_presentation.pptx`](final_submission/presentation/project_presentation.pptx) — Comprehensive 14-slide 16:9 presentation deck formatted with native PowerPoint tables, figures, callouts, and structured hierarchy.
- **Presentation Speaker Notes:** [`final_submission/presentation/speaker_notes.md`](final_submission/presentation/speaker_notes.md) — 8-minute balanced presentation script split equally between Vishwanath Venugopal and Shaik Koduru Asif (4 minutes each).
- **Interactive 3D WebGL Viewer:** [`final_submission/demo/change_demo.html`](final_submission/demo/change_demo.html) — Standalone browser viewer with 3D orbit controls, epoch selection, and method comparison.
- **Comprehensive Results Summary:** [`final_submission/results_summary.md`](final_submission/results_summary.md)
- **Final Project Status:** [`final_submission/final_status.md`](final_submission/final_status.md)

---

## Repository Structure

```text
Visioners_2.0/
├── code/                                # Complete Python research & benchmark suite
│   ├── build_full_presentation.py       # Compiles 14-slide PPTX deck with python-pptx
│   ├── build_full_report.py             # Compiles 6-page PDF & Markdown research paper
│   ├── c2c_baseline.py                  # Cloud-to-Cloud distance implementation
│   ├── m3c2_baseline.py                 # Audited multiscale cylinder M3C2 implementation
│   ├── learning_v2_model.py             # V2 Multiscale Cross-Scan Siamese PointNet
│   ├── study_pipeline.py                # 318-condition perturbation harness
│   └── test_*.py                        # 83 automated unit tests
├── docs/                                # Documentation, dataset layout & provenance
│   ├── DATASETS.md                      # Download & extraction instructions
│   ├── PROJECT_STATUS.md                # Evolution & audit history
│   └── REPRODUCIBILITY.md               # Environment setup & exact versions
├── final_submission/                    # Official final deliverables
│   ├── demo/change_demo.html            # Standalone interactive 3D WebGL viewer
│   ├── presentation/                    # 14-slide PPTX deck and speaker notes
│   ├── report/                          # 6-page PDF paper and Markdown manuscript
│   ├── results_summary.md               # Complete V2 metric tables across all 6 pairs
│   └── final_status.md                  # Verification audit & delivery checklist
├── results/                             # Saved checkpoints, metrics, and audit logs
│   ├── learning_v2_coverage_2026-10-25_v1/ # Authoritative V2 6-pair study results
│   ├── construction_change_regions_2026-10-26_v1/ # 191 extracted change regions
│   ├── construction_intensity_2026-10-26_v1/      # LiDAR intensity negative control
│   └── conslam_qualitative_2026-10-25_v2/         # ConSLAM Sequence 3 qualitative data
└── requirements.txt                     # Pinned Python package dependencies
```

---

## Quick Start & Verification

### 1. Environment Setup
```powershell
git clone -b complete-project-v2-final https://github.com/ShaikKoduruAsif114/Visioners_2.0.git
cd Visioners_2.0

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt python-pptx reportlab
```

### 2. Run Automated Unit Tests (83/83 Passing)
```powershell
python -m unittest discover -s code -p "test_*.py" -v
```
*Result:* **83 passing unit tests** verifying geometric invariance, normal orientation, cylinder boundaries, focal loss, and optimizer restarts.

### 3. Rebuild Presentation & Report Artifacts
```powershell
python code/build_full_presentation.py
python code/build_full_report.py
```

### 4. Launch Interactive 3D Demo
Open `final_submission/demo/change_demo.html` in any modern web browser.

---

## Team Attribution (CV5570)

- **Vishwanath Venugopal (Roll No. CE23B124):** Geometric baseline auditing (C2C, M3C2), spatial fold partitioning, degradation test harness, ConSLAM inspection.
- **Shaik Koduru Asif (Roll No. CE23B114):** Deep multiscale learning architecture, candidate change region clustering, multi-modal intensity analysis, BIM progress integration.

**Department of Civil Engineering, Indian Institute of Technology Madras**
