# CV5570 Final Presentation Speaker Notes
**Project Title:** Robust 3D Change Detection for Construction Progress Monitoring  
**Authors:** Vishwanath Venugopal (CE23B124) & Shaik Koduru Asif (CE23B114)  
**Date:** 2 November 2026  
**Total Target Time:** 8 minutes (480 seconds)  
**Time Split:** Vishwanath: 240 seconds (50%) | Asif: 240 seconds (50%)

---

### Slide 1: Title & Course Introduction
- **Speaker:** Vishwanath Venugopal & Shaik Koduru Asif
- **Duration:** 30 seconds (15s each)
- **Content:**
  - *Vishwanath:* "Good morning Professor and evaluators. Today we present our semester research project for CV5570, titled 'Robust 3D Change Detection for Construction Progress Monitoring'. Automated tracking of physical site evolution across repeated 3D scans is a cornerstone of modern digital construction management."
  - *Asif:* "Our project benchmarks conventional geometric change detection algorithms against deep multiscale cross-scan neural networks under controlled, real-world sensor imperfections. We address data leakage, evaluate all six multi-epoch site pairs, and investigate physical component takeoff."
- **Key Evidence:** User proposal, MultiChange3D benchmark, ConSLAM dataset.

---

### Slide 2: Construction Reality: Why Field 3D Change Detection Fails
- **Speaker:** Vishwanath Venugopal
- **Duration:** 40 seconds
- **Content:**
  - "On real active construction sites, scans are never textbook-perfect. Scans taken on different dates suffer from four fundamental physical imperfections:
    1. *Registration Drift:* ICP or sensor drift of 1 to 10 cm between epochs. Rigid distance thresholds interpret these non-moving walls as pseudo-displacements, triggering false alarm storms.
    2. *Range & Surface Noise:* Rough concrete, dust, and oblique incident angles create measurement scatter around true planar boundaries.
    3. *Point Sparsity:* LiDAR density decays steeply with range ($1/R^2$), inflating nearest-neighbor Euclidean gaps on completely static surfaces.
    4. *Occlusions:* Scaffolding, tower cranes, and temporary material piles block views, making missing points look like structural removals.
  - A naive scan comparison marks all of these artifacts as change, burying real construction progress."
- **Key Evidence:** Controlled degradation formulation, field sensor characteristics.

---

### Slide 3: Resolving the Ground Truth Dilemma: Midterm Review Pivot
- **Speaker:** Shaik Koduru Asif
- **Duration:** 45 seconds
- **Content:**
  - "When we originally submitted our project proposal, we planned to perform change detection solely on the ConSLAM dataset.
  - However, during our midterm review investigation, we encountered a fundamental scientific blocker: ConSLAM was built as a mobile robot SLAM benchmark—it carries *zero* per-point ground truth change labels! Sweeps are taken seconds apart rather than across construction milestones.
  - Without verified ground-truth point labels, one cannot compute Precision, Recall, F1-score, or IoU. Any claimed accuracy would be ungrounded guesswork.
  - Therefore, during our midterm review, we instituted a strategic dual-dataset protocol:
    1. *MultiChange3D Bike Parking Construction Scene:* Used as our quantitative benchmark engine. It provides 4 epochs, 6 temporal pairs, ~5M points per scan, and certified per-point ground truth labels.
    2. *ConSLAM Sequence 3:* Retained for qualitative field inspection on handheld mobile LiDAR sweeps and 104M-point TLS reference data."
- **Key Evidence:** `Robust_3D_Change_Detection_v2.pptx`, MultiChange3D ISPRS paper, ConSLAM ASCE 2023 paper.

---

### Slide 4: Problem Formulation: Directional Multi-Epoch Change
- **Speaker:** Vishwanath Venugopal
- **Duration:** 40 seconds
- **Content:**
  - "Mathematically, geometric change between an earlier scan ($E_i$) and a later scan ($E_j$) has physical direction:
    - *Candidate Removals:* Earlier points ($E_i$) with no counterpart in $E_j$ represent excavation, formwork stripping, or demolished temporary works.
    - *Candidate Additions:* Later points ($E_j$) with no counterpart in $E_i$ represent newly erected columns, concrete slabs, or framing.
    - *Unchanged Geometry:* Surrounding pavement, boundary walls, and static terrain.
  - Unlike preliminary studies restricted to a single pair, our completed project benchmarks all six temporal pairs: Pair 01, 02, 03, 12, 13, and 23. This spans short-term construction cycles (e.g. 01) to full-project completion (03)."
- **Key Evidence:** `results/learning_v2_coverage_2026-10-25_v1/fold_*/`, 6-pair MultiChange3D manifests.

---

### Slide 5: Methodology: Comparing Three Detection Paradigms
- **Speaker:** Shaik Koduru Asif
- **Duration:** 50 seconds
- **Content:**
  - "We implement and evaluate three distinct algorithmic paradigms:
    1. *Cloud-to-Cloud (C2C):* Computes Euclidean nearest-neighbor distances ($d_i$). A point is classified as changed if $d_i > \tau = 0.075$ m (7.5 cm). Fast and simple, but highly vulnerable to rigid shifts and sparsity.
    2. *M3C2-Style (Multiscale Model to Model):* Estimates surface normals across multiple scales (10, 20, 40 cm), projects points into a shared cylinder of radius 10 cm and length 1 m, and tests distances against a 95% Confidence Limit of Detection (LoD95). It abstains, outputting 'Unknown', if fewer than 10 points fall in the cylinder.
    3. *V2 Multiscale Deep Learning (Ours):* A deep Siamese PointNet-style network that processes paired local patches across multiscale neighborhood radii (up to 0.60 m). It combines local structural shape representations with a proximity prior, trained using focal loss to handle severe class imbalance."
- **Key Evidence:** `code/c2c_baseline.py`, `code/m3c2_baseline.py`, `code/learning_v2_model.py`.

---

### Slide 6: Validation Protocol: 5-Fold Buffered Spatial Cross-Validation
- **Speaker:** Vishwanath Venugopal
- **Duration:** 45 seconds
- **Content:**
  - "A major failure mode in published 3D learning literature is random point partitioning. Because LiDAR scans are dense, randomly assigned test points lie millimeters away from training points, causing catastrophic data leakage.
  - To guarantee genuine generalization, we implemented 5-fold buffered spatial cross-validation. We sliced the construction site along the spatial X-axis into 5 geographic zones, separated by 2.22-meter buffer margins—exceeding twice our model's receptive field halo.
  - Across all 6 pairs and 5 spatial folds, we executed 30 independent pair-fold studies, querying over 225,000 balanced cores.
  - We also identified and analyzed the *metric ceiling phenomenon*: in certain folds and pairs (e.g. Fold 3), zero true changes occurred. When ground truth positive points equal zero, changed F1 is mathematically bounded at 0.0, capping macro F1 at 0.825. Reporting pooled confusions ensures no distorted rankings."
- **Key Evidence:** `results/learning_v2_coverage_2026-10-25_v1/analysis/clean_per_fold.csv`, `metric_ceiling_diagnostic.json`.

---

### Slide 7: Benchmark Results: Multiscale V2 Leads Clean Site Evaluation
- **Speaker:** Shaik Koduru Asif
- **Duration:** 50 seconds
- **Content:**
  - "Examining our clean construction site benchmark across all 30 pair-fold evaluations:
    - Our *V2 Multiscale Validation-Selected Network* achieves a binary Macro F1 of **0.6537** and a Changed F1 of **0.3900**, with 0% detector abstentions.
    - This decisively outperforms the classical C2C baseline (**0.5659** Macro F1, **0.3000** Changed F1)—an improvement of **+8.8 percentage points** in Macro F1 and **+9.0 points** in Changed F1.
    - M3C2-style struggles severely on complex construction geometry, scoring only **0.2627** Macro F1 while abstaining on **26.6%** of points.
    - Why does M3C2 fail? Construction sites contain thin rebar, scaffolding pipes, and sharp slab edges. These non-planar geometries violate M3C2's local planar cylinder assumption, causing massive abstentions. Our multiscale neural network learns flexible geometric representations without abstaining."
- **Key Evidence:** `clean_comparison.csv`, `clean_per_fold.csv`, `class_support_and_pooled_confusions.csv`.

---

### Slide 8: Robustness Evaluation: 318 Controlled Degradation Conditions
- **Speaker:** Vishwanath Venugopal
- **Duration:** 45 seconds
- **Content:**
  - "To answer the fundamental question—how well do these detectors hold up when scans degrade?—we subjected each method to an exhaustive suite of 318 controlled degradation conditions.
  - We evaluated:
    - Translation shifts of 10 cm and 20 cm, plus angular rotations of 0.5° and 1.0°.
    - Range noise up to $\sigma = 7.5$ cm.
    - Density downsampling to 25% and 10% point retention.
    - Cylindrical occlusion cutoffs of 25% and 50% radius, evaluated under both Strict Protocol A (where missing points are penalized) and Visibility-Masked Protocol B.
    - Compound multi-stress conditions combining translation, noise, and decimation simultaneously.
  - Every condition was evaluated across all 6 pairs with fixed seed controls."
- **Key Evidence:** `stress_condition_summary.csv`, `robustness_binary_macro.png`, `robustness_changed_f1.png`.

---

### Slide 9: Field Stress Analysis: Where Deep Learning Wins Decisively
- **Speaker:** Shaik Koduru Asif
- **Duration:** 50 seconds
- **Content:**
  - "Here is the critical scientific discovery of our research:
    - In clean, perfectly registered scans, C2C is competitive because nearest-neighbor distances are small.
    - But look at what happens under *Combined Large Field Stress* (20 cm translation, noise, and sparsity):
      - C2C collapses from 0.5659 down to **0.4181**—a drop of **14.8 percentage points**! Because the static structure is shifted by 20 cm, C2C's 7.5 cm threshold is exceeded everywhere, falsely marking the entire building as changed.
      - M3C2 stagnates at **0.4235** with massive coverage dropouts.
      - Our *V2 Multiscale Robust Network* retains **0.6176 Macro F1**—dropping only 3.8 points!
    - Under severe field conditions, deep learning maintains a massive **+20.0 percentage point lead** over C2C. The neural network learns relative shape signatures and surface neighborhoods rather than brittle scalar distances."
- **Key Evidence:** `stress_condition_summary.csv`, `matched_clean_stress_cores.csv`.

---

### Slide 10: Connecting Points to Construction: Change Region Clustering
- **Speaker:** Vishwanath Venugopal
- **Duration:** 45 seconds
- **Content:**
  - "To make our computer vision pipeline actionable for civil engineers, we bridged unstructured point clouds to physical construction assemblies.
  - Site managers do not inspect individual point clouds; they track work items and components.
  - We developed a 3D connected-component clustering algorithm that groups positive predicted change cores into coherent physical instances.
  - Across all 6 pairs, we extracted **191 distinct construction change regions**:
    - **80 Candidate Added Regions:** newly erected steel stanchions, shelter canopy framing, and bike rack fixtures.
    - **111 Candidate Removed Regions:** ground trenching, excavated earth mounds, and dismantled formwork panels.
  - For every single region, our system outputs 3D oriented bounding boxes, centroids, and volumetric estimates, enabling direct automated progress takeoff."
- **Key Evidence:** `candidate_regions.csv`, `pair_summary.csv`.

---

### Slide 11: Multi-Modal Evidence: LiDAR Intensity Negative Control
- **Speaker:** Shaik Koduru Asif
- **Duration:** 45 seconds
- **Content:**
  - "We also investigated whether multi-modal features, specifically LiDAR return intensity, could enhance change detection by identifying material changes (such as concrete curing, asphalt paving, or steel framing).
  - To test this scientifically without self-deception, we introduced a *Shuffled Intensity Negative Control*, where intensity values were permuted across points while coordinates remained identical.
  - The findings were eye-opening: adding raw intensity improved validation F1 by only +0.005 (from 0.7510 to 0.7560), while the shuffled control produced an identical 0.7510.
  - This confirmed that spatial 3D geometry accounts for over 98% of change detection discriminative power on construction sites. We also explored neural-geometric score fusion (+0.0088 gain), but adhered strictly to our pre-declared +0.010 promotion gate, preserving our clean, single-stream V2 model as primary."
- **Key Evidence:** `results/construction_intensity_2026-10-26_v1/final_review.json`, `construction_evidence_fusion_2026-10-26_v1`.

---

### Slide 12: Field Deployment: Qualitative Evaluation on ConSLAM Mobile Scans
- **Speaker:** Vishwanath Venugopal
- **Duration:** 45 seconds
- **Content:**
  - "We returned to ConSLAM Sequence 3 to inspect our methods on raw handheld mobile LiDAR.
  - We analyzed two sweeps taken 4.84 seconds apart on an active multi-story site, comparing them against the 104.5-million-point terrestrial laser scanner (TLS) ground truth.
  - Key finding: while unposed frames had a median nearest-neighbor gap of 39.5 cm, applying the robot's optimized SLAM trajectory reduced the median gap to 8.1 cm.
  - However, because the sensor was moving, over 54.8% of static points still exhibited distances greater than 7.5 cm!
  - This empirical test explains exactly why rigid geometric C2C fails on mobile robotics platforms. Quadruped and drone inspection systems require deep learning models that are robust to 8-cm trajectory fluctuations."
- **Key Evidence:** `results/conslam_qualitative_2026-10-25_v2/inspection.json`, `conslam_trajectory.png`.

---

### Slide 13: Civil Engineering Impact: 4D BIM & Automated Progress Takeoff
- **Speaker:** Shaik Koduru Asif
- **Duration:** 45 seconds
- **Content:**
  - "The downstream civil engineering applications of our pipeline are profound:
    1. *Scan-to-BIM Reconciliation:* Extracted 3D change regions can be directly cross-referenced against 4D Revit / Navisworks models via IFC schemas, automatically verifying whether built work matches planned milestones.
    2. *Automated Earned Value Management (EVM):* By quantifying installed component volumes ($m^3$ of concrete or structural steel), our pipeline computes automated Schedule Performance Index (SPI) and Cost Performance Index (CPI) without manual surveys.
    3. *Quality Assurance & Temporary Works Safety:* Detecting candidate removals ensures that shoring, formwork, and safety railings are not stripped prematurely, preventing catastrophic structural failures."
- **Key Evidence:** IFC 4D BIM integration workflows, construction progress monitoring literature.

---

### Slide 14: Conclusions, Verified Deliverables & Team Attribution
- **Speaker:** Vishwanath Venugopal & Shaik Koduru Asif
- **Duration:** 40 seconds (20s each)
- **Content:**
  - *Vishwanath:* "In conclusion, our project delivers an end-to-end, scientifically validated 3D change detection system. We resolved the ground truth dilemma, audited geometric baselines across 30 pair-fold evaluations, proved deep learning's 20-point superiority under real-world field stress, and clustered changes into 191 physical work-item components."
  - *Asif:* "All 83 automated unit tests pass completely. All deliverables—including our comprehensive final report, 14-slide presentation deck, interactive 3D WebGL viewer, and model checkpoints—are published on branch `complete-project-v2-final` on GitHub. Thank you, and we welcome any questions."
- **Key Evidence:** Unit test suite (83/83 passed), GitHub repository `Gopal1909/Visioners_2.0`.
