# CV5570 — viva preparation

Both presenters should rehearse every answer. Spoken answers are deliberately short; the technical follow-up is for questions. Numerical details must be read with `evidence_reconciliation.json` and the final slide notes. Paths below are relative to `D:\CV_Project`.

## 1. Problem formulation and construction relevance

### 1. What problem did you actually solve?
**Say:** “We detect candidate geometric changes between two scans and compare how reliable several detectors are under controlled scanning imperfections. We do not directly measure completed construction.”
**Deeper:** Binary changed/unchanged decisions become candidate removal or addition according to the earlier/later scan direction. An apparent change may represent permanent work, temporary activity, coverage differences, or error.
**Evidence:** `docs/DATASETS.md`; `results/construction_change_regions_2026-10-26_v1/design.json`.

### 2. Why is this relevant to civil engineering?
**Say:** “It can focus a site engineer’s review on places that might have changed, reducing the area requiring manual inspection.”
**Deeper:** Progress estimation additionally needs event verification, component or work-item association, dates, quantities and schedule information. Localization is an enabling step, not a validated takeoff system.
**Evidence:** `docs/PROJECT_STATUS.md`; `results/construction_temporal_2026-10-26_v1/summary.json`.

### 3. What is the strongest contribution?
**Say:** “A reproducible same-site comparison showing that the retained multiscale learning workflow improves sampled-core change detection over our geometric baselines, accompanied by robustness and failure analysis.”
**Deeper:** The contribution includes common evaluation support, validation-selected settings, explicit abstention handling, and preservation of unsuccessful follow-ups. It is not a new state-of-the-art multi-site benchmark claim.
**Evidence:** `results/learning_v2_coverage_2026-10-25_v1/analysis/clean_comparison.csv`; `docs/PROJECT_STATUS.md`.

## 2. Dataset choice and labels

### 4. Why change from ConSLAM to MultiChange3D?
**Say:** “The available ConSLAM inputs do not have the temporal point-change labels needed for our supervised quantitative evaluation. BikeParkingConstruction provides those labels.”
**Deeper:** ConSLAM remains useful for qualitative acquisition/registration inspection. Two LiDAR sweeps a few seconds apart are not an across-month construction progress ground truth.
**Evidence:** `docs/DATASETS.md`; `results/conslam_qualitative_2026-10-25_v2/inspection.json`.

### 5. Are these six independent construction sites?
**Say:** “No. Six pairwise comparisons come from four acquisition epochs at one site.”
**Deeper:** Four epochs yield six unordered temporal pairs. Shared scans and overlapping temporal events create dependence. Pair-fold rows must not be treated as independent-site replications.
**Evidence:** `docs/DATASETS.md`; `docs/PROJECT_STATUS.md`.

### 6. What do the labels mean?
**Say:** “The source label says changed or unchanged. Changed points in the earlier cloud are candidate removals; those in the later cloud are candidate additions.”
**Deeper:** This directional interpretation does not identify a semantic object or the reason for change. The PLY field is `cd_type`; original row IDs are preserved in the research pipeline.
**Evidence:** `docs/DATASETS.md`; `code/multipair_data.py`.

### 7. What did Open_space demonstrate?
**Say:** “It is a separate user-supplied indoor cross-scene diagnostic, and C2C performed better there. It is not another construction-site validation.”
**Deeper:** Frozen construction settings were used; the scene was not used to fit the retained model. There is also a recorded catalog-versus-local-file identity caveat. This limits generalization claims rather than supporting universal learning superiority.
**Evidence:** `results/learning_v2_coverage_2026-10-25_v1/external_open_space/`; `docs/DATASETS.md`.

## 3. C2C and M3C2-style

### 8. How does C2C work?
**Say:** “For each query point it measures the distance to the nearest point in the other scan and compares that distance with a threshold.”
**Deeper:** It is simple and effective when scans are well aligned and coverage is comparable. A large distance can also come from missing observations, sampling or misregistration. Fixed 7.5 cm and validation-calibrated C2C are separate baselines.
**Evidence:** `code/c2c_baseline.py`; `results/learning_v2_coverage_2026-10-25_v1/analysis/clean_comparison.csv`.

### 9. Why use M3C2-style surface comparison?
**Say:** “It compares local surfaces along an estimated normal and accounts for local spread and support, rather than only nearest-neighbour distance.”
**Deeper:** Reliable normals and sufficient cylindrical support matter. Edges, complex geometry or low support can result in abstention. We do not claim equivalence to independently validated reference software.
**Evidence:** `code/m3c2_baseline.py`; `code/test_m3c2_synthetic.py`.

### 10. Does unknown mean unchanged?
**Say:** “No. Unknown means the detector could not support a decision; it is not a confident negative.”
**Deeper:** Unknown rate and evaluation coverage must accompany scores. Comparing methods only where one abstaining method succeeds can hide difficult points; full-requested-support reporting is important.
**Evidence:** `code/multipair_metrics.py`; `results/learning_v2_coverage_2026-10-25_v1/analysis/clean_comparison.csv`.

## 4. V2 architecture and training

### 11. What does V2 learn beyond distance?
**Say:** “It combines multiscale local geometric evidence across the two scans, so it can interpret a distance together with its surrounding structure.”
**Deeper:** The retained implementation is a compact paired-patch, PointNet-style metric-fusion model, not a reproduction of published KPConv or PointNet++. Describe the actual source and selection configuration, not an invented architecture.
**Evidence:** `code/learning_v2_model.py`; `code/learning_v2_data.py`; final presentation architecture appendix.

### 12. Did the retained V2 use focal loss?
**Say:** “No. The retained V2 implementation uses weighted binary cross-entropy. Focal loss was explored in later follow-ups.”
**Deeper:** Pair/present-class weighting addresses imbalance. Later hierarchy/focal experiments changed more than architecture in some comparisons; the matched controlled pilot is the appropriate narrower comparison.
**Evidence:** `code/learning_v2_model.py`; `code/learning_v3_model.py`; `results/controlled_architecture_2026-10-26_v1/frozen_selection.json`.

### 13. Why not use the newer hierarchical model?
**Say:** “Newer did not mean better. The matched hierarchical pilot failed the frozen validation gates, so we retained V2.”
**Deeper:** Model promotion should follow declared gain and quality guards. Repeatedly relaxing gates or using held-out results to pick a preferred model would weaken the evidence.
**Evidence:** `results/controlled_architecture_2026-10-26_v1/frozen_selection.json`; `docs/PROJECT_STATUS.md`.

## 5. Validation and data leakage

### 14. Why not randomly split individual points?
**Say:** “Neighbouring points are highly correlated. Random point splitting can place almost identical local geometry in both training and testing.”
**Deeper:** Buffered spatial folds separate query regions and account for local neighbourhood support. They reduce a specific spatial-overlap risk but do not prove complete independence or unseen-site performance.
**Evidence:** `code/multipair_crossvalidate.py`; `code/multipair_nested.py`; final evidence reconciliation.

### 15. Is your test set completely untouched?
**Say:** “Not throughout the full project history. Earlier findings informed later development, and validation was reused. We explicitly report that limitation.”
**Deeper:** The same-site spatial comparison is useful evidence, but a fresh construction site or prospectively frozen evaluation is needed for a stronger generalization claim.
**Evidence:** `docs/PROJECT_STATUS.md`; `README.md`.

### 16. Why not call the 30 pair-fold rows independent experiments?
**Say:** “They share a site, epochs and related training/evaluation protocols, so that would exaggerate independence.”
**Deeper:** Repeated-seed variability, perturbation variability and fold variability answer different questions. None should be substituted for independent-site uncertainty estimates.
**Evidence:** `results/learning_v2_coverage_2026-10-25_v1/analysis/between_fold_variation.csv`; `analysis/training_seed_variation.csv` within the same run.

## 6. Metrics and class imbalance

### 17. What is macro-F1, and why not just accuracy?
**Say:** “F1 balances precision and recall. Binary macro-F1 averages the changed and unchanged class F1 scores, so a dominant unchanged class cannot completely hide poor change detection.”
**Deeper:** F1 = 2TP/(2TP+FP+FN) for a class. The implementation’s absent-class convention matters. Always distinguish class averaging, pair/direction averaging and pooling raw confusion counts.
**Evidence:** `code/multipair_metrics.py`; `results/learning_v2_coverage_2026-10-25_v1/analysis/metric_ceiling_diagnostic.json`.

### 18. Why are 0.6537, 0.3900 and 0.7405 all reported for V2?
**Say:** “They answer different questions: mean binary macro-F1, mean changed-class F1, and globally pooled directional three-class macro-F1.”
**Deeper:** Pooling counts changes observation weighting, and directional three-class labels differ from binary detection. These numbers are not interchangeable and none is ordinary accuracy.
**Evidence:** `analysis/clean_comparison.csv` and `analysis/class_support_and_pooled_confusions.csv` under the retained V2 run.

### 19. Why is changed-class F1 much lower?
**Say:** “Changes are difficult and unevenly distributed. Some test directions contain no changed points, and our absent-class-zero convention affects the mean.”
**Deeper:** This is why pooled class metrics, class support, precision/recall and unchanged-only false alarms should accompany averaged F1. It is not a justification to omit difficult directions.
**Evidence:** `results/learning_v2_coverage_2026-10-25_v1/analysis/metric_ceiling_diagnostic.json`; `analysis/class_support_and_pooled_confusions.csv` within the same run.

## 7. Robustness experiments

### 20. Are the stress experiments measured field errors?
**Say:** “They are controlled synthetic perturbations applied to real scan data, not independently measured field-disturbance ground truth.”
**Deeper:** Translation, rotation, noise, sparsity, occlusion and compound stresses probe sensitivity. A perturbation’s units and matched support must be shown. A severe-stress row for a fixed robust model must not silently be relabelled as the selected workflow across all folds.
**Evidence:** `results/learning_v2_coverage_2026-10-25_v1/analysis/stress_condition_summary.csv`; final reconciliation stress protocol.

### 21. Does V2 always beat C2C?
**Say:** “No. V2 improves our retained construction comparison and is more resilient in the highlighted controlled stress case, but C2C wins the separate Open_space diagnostic.”
**Deeper:** The severe-stress advantage is approximately 0.20 absolute macro-F1, not a universal performance guarantee or a 20% accuracy increase.
**Evidence:** retained V2 `analysis/clean_comparison.csv`, `analysis/stress_condition_summary.csv`, and `external_open_space/`.

## 8. False positives and uncertainty

### 22. Are the 191 regions construction objects?
**Say:** “No. They are sampled candidate geometric components. They have not been independently verified as construction work items.”
**Deeper:** Eighty are candidate additions and 111 removals. Eighty-three contain only unchanged-labelled sampled points. That is a sampled point-label audit, not verified object-level precision.
**Evidence:** `results/construction_change_regions_2026-10-26_v1/candidate_regions.csv`; `docs/PROJECT_STATUS.md`.

### 23. Did intensity prove geometry explains more than 98% of performance?
**Say:** “No. The controlled intensity addition gave a small validation gain, but failed the promotion criteria. It does not support that attribution claim.”
**Deeper:** Comparing 0.7560 with 0.7510 does not decompose causal contributions to a nonlinear metric. Shuffling is a useful negative control, not a percentage-explained estimator.
**Evidence:** `results/construction_intensity_2026-10-26_v1/final_review.json`.

### 24. Does a large nearest-neighbour gap prove occlusion?
**Say:** “No. It is geometric evidence, not an explanation of the cause.”
**Deeper:** Pose/ray-origin metadata, coverage checks or human review would be needed to distinguish occlusion, acquisition differences and actual changes. The pair files lack the necessary acquisition metadata for verified visibility attribution.
**Evidence:** `docs/DATASETS.md`; `results/construction_intensity_2026-10-26_v1/final_review.json`.

## 9. ConSLAM and real-world limitations

### 25. What does the ConSLAM figure establish?
**Say:** “Applying the supplied poses improves the alignment of the selected sweeps. It illustrates acquisition and coverage challenges; it does not establish change-detection accuracy.”
**Deeper:** The selected sweeps are about 4.84 seconds apart. The recorded nearest-neighbour statistics are not verified static-point false alarms, and the TLS coordinate relation is not independently confirmed.
**Evidence:** `results/conslam_qualitative_2026-10-25_v2/inspection.json`.

### 26. Is this ready for deployment on any site?
**Say:** “No. We need fresh construction-site evaluation, acquisition metadata and independent event review before deployment claims.”
**Deeper:** The Open_space result illustrates domain dependence. Dense validation patches are deliberately limited and their high pooled score must not replace the main spatial benchmark.
**Evidence:** `docs/PROJECT_STATUS.md`; `results/construction_dense_2026-10-26_v1/summary.csv`.

## 10. BIM and practical progress monitoring

### 27. Can it calculate quantities, SPI or CPI?
**Say:** “Not in the current validated system. We localize candidate geometric changes; verified quantities and earned-value measures need additional information.”
**Deeper:** A component bounding box is not installed material volume. BIM/work-item matching, observation completeness, schedule baselines and cost records would be required.
**Evidence:** `results/construction_temporal_2026-10-26_v1/summary.json`; `docs/PROJECT_STATUS.md`.

### 28. What would you do next?
**Say:** “Independently review candidate events, freeze a small method set, and evaluate on new construction evidence before adding BIM and progress tracking.”
**Deeper:** Preserve uncertainty, count abstentions on full requested support, use matched architecture comparisons, and associate only verified persistent events with work items. Do not lower acceptance gates simply to promote a newer candidate.
**Evidence:** `docs/PROJECT_STATUS.md`; `results/construction_temporal_2026-10-26_v1/generalization_protocol_v2.json`.

## Demo questions

### 29. Is the model running live in the browser?
**Say:** “No. The interaction is live, but the predictions are saved research outputs. This avoids training or inference surprises during the presentation.”
**Deeper:** The viewer exposes labelled data, method decisions and candidate regions with provenance. Rendering subsets are for responsiveness and are not a new benchmark.
**Evidence:** `demo/` provenance and `demo/demo_validation.json` in this delivery; `LIVE_DEMO_GUIDE.md`.

### 30. Did you record or upload the demonstration video?
**Say:** “The package includes a four-minute storyboard and narration. Recording and upload are separate user actions.”
**Deeper:** Never represent a storyboard, a rendered slide sequence or a browser screenshot as a completed screen recording.
**Evidence:** `DEMO_VIDEO_SCRIPT.md`.
