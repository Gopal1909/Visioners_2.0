## 1 Title

Vishwanath Venugopal, 20 seconds

Our project compares geometric and learned ways to locate additions and removals in repeated construction scans. We focus on controlled robustness, fair evaluation and explicit unsupported areas. The picture shows actual labelled original cores from Bike Parking Construction. We make no claim that a geometric discrepancy proves progress or structural completion.

Sources: User proposal Robust_3D_Change_Detection_v2.pdf; artifacts/historical_pilot/primary_comparison.csv; https://3dom-fbk.github.io/multichange3d/

## 2 Data and directional labels

Vishwanath Venugopal, 35 seconds

All quantitative results use MultiChange3D Bike Parking Construction pair 01, about 4.44 and 5.09 million points. The binary field is unchanged zero or changed one in each epoch. A changed earlier point means removed, and a changed later point means added. We verified the official label preparation and classical binary metric code. ConSLAM has no verified per-point change labels here. Its scans and poses provide qualitative context only. Additional official download links redirected to sign-in, so this submission contains no cross-pair or cross-scene experiment.

Sources: artifacts/dataset_manifest.json; artifacts/data_sources_and_access.md; https://github.com/3DOM-FBK/multichange3d/blob/main/scripts/prepare_gt_labels.py

## 3 Three detectors with different assumptions

Vishwanath Venugopal, 65 seconds

C2C uses Euclidean nearest-neighbour distance above 7.5 centimetres, fixed from the historical same-pair sweep. M3C2-style estimates source normals at 10, 20 and 40 centimetre radii, then compares two projected neighbourhood means in the same 10 centimetre radius cylinder of half-length one metre. At least ten projected points per epoch and stable normals are required; otherwise it abstains. We checked the population spread and registration-term placement against CloudCompare v2.13.2 source, but did not validate output equivalence. Our learned baseline shares a point MLP between paired 128-point patches, pools geometry and combines it with an NN prior. Ground-truth labels enter supervision only.

Sources: artifacts/geometry_frozen.json; artifacts/learned/model_config.json; artifacts/historical_pilot/reference_conventions.json; https://arxiv.org/abs/1302.1183; https://arxiv.org/abs/1612.00593

## 4 Buffered spatial evaluation

Vishwanath Venugopal, 55 seconds

We define these X regions using geometry, with 2.22 metre gaps. The gap exceeds twice the largest clean geometric support halo, so clean train, calibration and test patches do not share neighbourhood support. Both acquisition epochs occur in every split, which means this is a same-scene spatial holdout for the learned model, not cross-scene generalization. We sample original indices across occupied two-metre tiles and use complete clouds for geometric support. Sparse tiles therefore get relatively more weight. Calibration diagnostics freeze geometry before new test and stress scoring. Historical full-pair C2C threshold selection and the earlier exploratory pilot remain disclosed sources of test exposure.

Sources: artifacts/split_config.json; artifacts/geometry_frozen.json; artifacts/spatial_points.json; 220 displayed original Scan 0 cores per split

## 5 C2C leads on this clean spatial region

Vishwanath Venugopal, 65 seconds

All three methods use the exact same valid original indices here: 5,350 of the requested 8,000 directional observations. C2C macro F1 is 0.4118, the learned baseline 0.3817, and M3C2-style 0.2342. M3C2 abstains on 33.125 percent. C2C and learned predictions have no detector abstentions; they are masked here for fairness, and their all-core scores are retained separately. These are lower than the earlier whole-scene pilot because the spatial region and sampled class mix differ. Only 27 removal and 86 addition labels remain after validity filtering, so recall is strongly conditional. We preserve the negative finding rather than promote a more favorable sweep row.

Sources: artifacts/primary_comparison.csv; artifacts/all_spatial_test_metrics.csv; historical full-pair values are in a separate scope in artifacts/all_evaluated_results.csv

## 6 Registration errors change each detector differently

Shaik Koduru Asif, 50 seconds

These are one-sided simulated shifts and rotations of Scan 1, with exact Stage 3 seed and directions. All method settings stay fixed. The smaller perturbation can improve scores for C2C or the learned network in this particular direction, so there is no monotonic or universal benefit from misalignment. At 20 centimetres, the longitudinal comparison on identical clean/current valid cores gives macro-F1 changes of about minus 0.105 for C2C, minus 0.139 for M3C2 and minus 0.045 for the learned network. Each severity uses a saved original-index mapping. The report gives both condition-specific scores and matched clean deltas.

Sources: artifacts/presentation_metrics.json; artifacts/matched_clean_deltas.csv; artifacts/perturbation_geometry.json

## 7 Density loss and noise also change valid support

Shaik Koduru Asif, 50 seconds

Sparsity retains exactly the same seeded subset for every method, and evaluation keeps only retained Scan 1 cores. Noise is independent Gaussian displacement per coordinate. The model was trained on clean data only. M3C2 coverage falls sharply under severe noise, so a valid-only score can hide difficult abstentions. Our matched-clean table intersects clean and current validity to separate some support effects from actual prediction changes. On 10 percent retention, the learned model loses roughly 0.096 macro F1 on that longitudinal support, while the corresponding C2C change is around minus 0.005. Those observations are single realizations with changing class counts, not a universal robustness ordering.

Sources: artifacts/matched_condition_metrics.csv; artifacts/matched_clean_deltas.csv; artifacts/coverage.png

## 8 Occlusion requires a visibility-aware interpretation

Shaik Koduru Asif, 40 seconds

The main spatial-test stripe happens to be outside both planned deletion cylinders. Its unchanged scores cannot establish occlusion robustness. We therefore add a separately labelled geometric supplement on the original whole-scene pilot indices. Strict protocol A retains all requested earlier-scan cores. Protocol B excludes cores inside the known synthetic blind cylinder; it changes evaluation support, not the detector. At radius 7.6 metres, M3C2 coverage is about 52 percent in A and 67 percent in B. Missing counterparts remain unknown, not confirmed physical removals. We do not evaluate the learned model on training-region supplement cores and call it a held-out result.

Sources: artifacts/occlusion_supplement/geometry_metrics.csv; artifacts/occlusion_supplement/geometry_frozen.json; all-scene geometric supplement scope

## 9 The learned baseline shows a large spatial gap

Shaik Koduru Asif, 35 seconds

We trained one small architecture for 30 predetermined epochs. Calibration loss, not test performance, selected the first epoch, and calibration F1 selected the 0.7 threshold. Training loss declines while calibration loss worsens. Calibration macro F1 is about 0.765 versus 0.394 on spatial test when both use all requested cores. The three-method common-support test score is separately 0.382. The model never receives change labels as input or absolute coordinates, but a single-scene split still shifts geometry and class balance. This compact baseline does not reproduce Siamese KPConv or characterize all learning methods.

Sources: artifacts/learned/training_history.csv; artifacts/learned/model_frozen.json; artifacts/learned/model_config.json; artifacts/primary_comparison.csv

## 10 ConSLAM provides qualitative site context

Shaik Koduru Asif, 35 seconds

We reuse local Sequence 3 LiDAR, normalized poses and a bounded TLS sample. Only 520 available pose timestamps match the 6,014 LiDAR files exactly. We show the first two matched frames, 4.84 seconds apart, using the supplied poses. Median sampled proximity falls from 0.395 metres before pose application to 0.081 afterwards. That is neither a registration RMS nor change-detection accuracy. The frames belong to one acquisition sequence, and we did not verify a common frame against TLS. Therefore this supports qualitative discussion of alignment and coverage, not a monthly construction-change or progress claim.

Sources: artifacts/conslam/inspection.json; https://www.repository.cam.ac.uk/items/693c260b-e145-4f01-ad88-d2031e4c0590; https://github.com/mac137/ConSLAM

## 11 Findings and limits

Shaik Koduru Asif, 30 seconds

C2C remains the strongest clean method on the matched spatial region. M3C2-style passes geometry tests but has many false positives and unknowns. The compact learned model is feasible on the laptop but exhibits a large spatial validation gap. Our strongest contribution is fair support-aware robustness analysis and an auditable negative result. Limitations include one scene and pair, one seed, historical threshold exposure, uncalibrated registration uncertainty and no independent M3C2 reference-output comparison. The offline demo runs live synthetic geometry and lets you rotate saved measured predictions. Geometry localizes candidate changes; semantic and BIM evidence would be needed to infer completed work or schedule status.

Sources: artifacts/submission_data.json; report/project_report.pdf; README.md; demo uses saved real measurements plus live synthetic check
