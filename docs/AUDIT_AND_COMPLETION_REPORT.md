# Detailed Project Audit, Evolution, and Completion Report

**Course:** CV5570 Computer Vision for Construction Engineering (November 2026)  
**Institution:** Department of Civil Engineering, Indian Institute of Technology Madras  
**Authors:** Vishwanath Venugopal (Roll No. CE23B124) & Shaik Koduru Asif (Roll No. CE23B114)  
**Branch:** `complete-project-v2-final`  
**Repository:** https://github.com/Gopal1909/Visioners_2.0 (Fork: https://github.com/ShaikKoduruAsif114/Visioners_2.0)

---

## 1. Project Background & Problem Evolution

### 1.1 The Initial Proposal
In the original project proposal titled *"Robust 3D Change Detection for Construction Progress Monitoring"*, the objective was formulated around tracking civil construction site progress using multi-sensor point clouds from the **ConSLAM** dataset. The planned methodology was to benchmark conventional geometric approaches (Cloud-to-Cloud and M3C2) against learning-based 3D models (Siamese PointNet) under controlled real-world sensor imperfections (sensor noise, registration error, point sparsity, and occlusion), measuring outcomes via Precision, Recall, F1-score, and IoU.

### 1.2 The Midterm Review Pivot: Resolving the Ground Truth Dilemma
During the midterm review, our team encountered a fundamental, non-negotiable scientific blocker with ConSLAM:
- **ConSLAM lacks per-point ground truth change labels.** ConSLAM was curated as an autonomous robot trajectory and SLAM benchmark. Its consecutive LiDAR sweeps were collected seconds apart (e.g. 4.84 seconds), not across monthly construction cycles.
- Without ground-truth point-level labels ($0 = \text{unchanged}, 1 = \text{changed}$), it is mathematically impossible to compute Precision, Recall, F1-score, or Intersection over Union (IoU). Any reported accuracy would be fabricated conjecture.

To preserve the scientific integrity of the research while fulfilling the course expectations in Computer Vision for Construction Engineering, we executed a strategic pivot during the midterm review:
1. **MultiChange3D Bike Parking Construction Scene (ETH Zurich / FBK, ISPRS 2026):** Adopted as our primary quantitative benchmark engine. It comprises 4 distinct construction epochs, yielding 6 directional pairs, with approximately 5 million Mobile Laser Scanning (MLS) points per cloud, each carrying certified per-point ground truth change labels (`cd_type`: 0 = unchanged, 1 = changed).
2. **ConSLAM Sequence 3:** Retained as a qualitative real-site inspection benchmark, evaluating mobile robot trajectory alignment and LiDAR density characteristics against a 104.5-million-point Terrestrial Laser Scanner (TLS) reference scan.

---

## 2. In-Depth Audit of the Existing GitHub Repository (`Visioners_2.0`)

Our audit examined every commit, script, metric file, and report artifact in `Gopal1909/Visioners_2.0`:

### 2.1 What Was Done in the Repository
1. **V1 Historical Pilot (Frozen in `final_submission/` as of 25 October 2026):**
   - Evaluated only a single pair (`pair_01`) under 11 conditions.
   - On that isolated pair, C2C achieved 0.4118 Macro F1, while the initial Siamese PointNet scored 0.3817 Macro F1 due to spatial distribution shifts and single-scale radius coverage weaknesses.
2. **V2 Research Upgrade (Located in `results/learning_v2_coverage_2026-10-25_v1/` and `code/`):**
   - **Full 6-Pair Expansion:** Processed all temporal transitions (`pair_01`, `pair_02`, `pair_03`, `pair_12`, `pair_13`, `pair_23`).
   - **5-Fold Buffered Spatial Cross-Validation:** Partitioned the site into 5 geographic zones separated by 2.22 m buffer gaps to eliminate spatial data leakage.
   - **V2 Multiscale Cross-Scan Architecture:** Upgraded the learning model to aggregate multiscale neighborhood geometry (up to $r=0.60$ m) with spatial proximity priors and focal loss.
   - **Outstanding Clean Benchmark:** V2 achieved **0.6537 Macro F1 / 0.3900 Changed F1**, decisively surpassing C2C (**0.5659 / 0.3000**) and M3C2 (**0.2627 / 0.1632**).
   - **Exhaustive 318-Condition Robustness Suite:** Benchmarked translation, rotation, Gaussian noise, point sparsity, and cylindrical occlusions. Under combined severe stress, V2 maintained **0.6176 Macro F1** while C2C collapsed to **0.4181** (a +20.0 percentage point advantage).
   - **Construction Change Region Extraction:** Clustered 75,000 positive cores into **191 physical change regions** (80 additions, 111 removals) with bounding dimensions.
   - **LiDAR Intensity Negative Control:** Proved via a shuffled-intensity control that 3D spatial geometry drives >98% of change detection discriminative power.
   - **Unit Testing:** 83 passing automated unit tests.

### 2.2 The Critical Gap in the Repository
As explicitly documented in `docs/PROJECT_STATUS.md` and `results/active_experiment.json`:
- `"report_and_presentation_authoring": false`
- `"No reports or presentations were regenerated for this publication."`

The deliverable files in `final_submission/` were **completely stale V1 artifacts** frozen on October 25:
- The PDF report (`project_report.pdf`) and PPTX presentation (`project_presentation.pptx`) only described the single-pair experiment (`pair_01`), claiming that C2C beat PointNet, and completely omitted the entire 6-pair benchmark, 5-fold cross-validation, 318 stress conditions, V2 multiscale model success, 191 change regions, and intensity findings!

---

## 3. Work Completed in This Branch (`complete-project-v2-final`)

We have resolved all gaps, synthesized the entire research evidence, and produced complete, publication-grade deliverables:

1. **Master Presentation Deck (`final_submission/presentation/project_presentation.pptx`):**
   - Authored a comprehensive 14-slide 16:9 presentation using `python-pptx` (via `code/build_full_presentation.py`).
   - Formatted with custom color-coded callouts, native comparison tables, high-resolution figures, and clear structural hierarchy.
   - Slide structure:
     1. Title & Course Context (CV5570, IIT Madras)
     2. Construction Site Reality & 4 Physical Imperfections
     3. Resolving the Ground Truth Dilemma (ConSLAM vs MultiChange3D)
     4. Directional Multi-Epoch Change Formulation (6 Pairs)
     5. Three Detection Paradigms (C2C, M3C2, V2 Deep Learning)
     6. 5-Fold Buffered Spatial Cross-Validation Protocol
     7. Clean Site Benchmark Results Across All 30 Pair-Fold Studies
     8. Robustness Under 318 Controlled Degradation Conditions
     9. Field Stress Analysis: Deep Learning's +20.0 pt Lead Over Collapsing C2C
     10. Construction Change Region Clustering (191 Components)
     11. Multi-Modal Evidence & LiDAR Intensity Negative Control
     12. Real-World Field Inspection on ConSLAM Mobile Scans
     13. Civil Engineering Impact: 4D BIM & Automated Progress Takeoff
     14. Conclusions, Deliverables & Equal Team Attribution
2. **Synchronized 8-Minute Speaker Notes (`final_submission/presentation/speaker_notes.md`):**
   - Cleanly split between Vishwanath Venugopal and Shaik Koduru Asif (4 minutes / 240 seconds each).
   - Details speaking scripts, evidence citations, and slide transitions.
3. **Comprehensive Research Paper (`final_submission/report/project_report.pdf` & `project_report.md`):**
   - Compiled a 6-page conference-style two-column research paper using ReportLab (via `code/build_full_report.py`).
   - Contains high-resolution figures, four comprehensive result tables, complete mathematical formulations, and formal citations.
4. **Updated Deliverable Summaries (`results_summary.md`, `final_status.md`, `README.md`):**
   - Fully synchronized all documentation to reflect the authoritative V2 results.
5. **Git Operations & Verification:**
   - All 83 automated unit tests executed and passed (`Ran 83 tests ... OK`).
   - Branch `complete-project-v2-final` created, committed, and pushed to GitHub fork (`ShaikKoduruAsif114/Visioners_2.0`).
   - Pull Request opened against `Gopal1909/Visioners_2.0`.
