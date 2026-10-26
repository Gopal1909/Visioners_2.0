# Current project state

Snapshot: 26 October 2026. Retained workflow: `results/learning_v2_coverage_2026-10-25_v1`. Latest activity: [intensity follow-up audit](../results/construction_intensity_2026-10-26_v1/final_review.json).

## Completed

- All six labelled construction pairs audited, with original indices and archives preserved.
- C2C, M3C2-style, geometry classifiers, and earlier Siamese PointNet-style controls.
- Five buffered spatial folds and validation-selected settings; misalignment, noise, sparsity, occlusion and combined stresses.
- V2 multiscale cross-scan learning, including a corrected radius-coverage sampling weakness.
- 318 clean/stress V2 cases plus a separate user-supplied external-scene diagnostic.
- Fresh training cores, hierarchy/attention, focal loss, hard sampling, checkpoint calibration, and fixed-budget refinement.
- Geometry vetoes, model agreement, match distributions, score fusion, and intensity with a shuffled negative control.
- Sampled candidate change regions, validation error atlases, independently checked outputs, and 83 passing tests.

## Results and interpretation

Retained V2 construction mean binary macro-F1: **0.653659**. Mean directional changed F1: **0.389989**. Differently weighted globally pooled unchanged/removed/added macro-F1: **0.740508**.

| Fold0 validation comparison | Mean clean/mild/medium macro-F1 |
|---|---:|
| V2 robust reference | 0.760792 |
| Fresh pooled BCE | 0.669952 |
| Fresh hierarchy BCE | 0.732709 |
| Fresh hierarchy focal | 0.749383 |
| Fresh geometry classifier | 0.727249 |
| Explicit match distributions | 0.751012 |
| Neural/geometry fusion | 0.769639 |
| Match distributions + core intensity | 0.756002 |
| Shuffled intensity | 0.751012 |

The fusion gain (+0.008847) missed the predeclared +0.01 requirement. Other follow-ups failed their respective clean/FPR/gain gates. None established a replacement or received new held-out scoring to search for a favorable result. These validation scores cannot be compared directly with 0.653659 held-out spatial F1.

Hierarchy/focal loss, explicit distributions and intensity showed useful intermediate effects. V2/V3 changed training duration, regularization, sampling and augmentation alongside architecture, so they are not an architecture-only comparison. The custom hierarchy does not reproduce published KPConv/PointNet++.

## Remaining gaps

- One construction site; six pairs share four epochs. Later development used previous test findings and repeatedly reused validation.
- Under absent-class-zero F1, 21/60 unchanged-only test directions cause protocol ceilings of 0.825 for mean binary macro-F1 and 0.65 for mean changed F1. Keep pooled class metrics and false-alarm rates alongside these; do not rename them accuracy.
- User-supplied Open_space favors C2C 0.859584 over V2 0.785656. See its data-identity caveat in [DATASETS.md](DATASETS.md).
- Regions are sparse sampled components: 191 total, 80 added/111 removed; 83 have only unchanged-labelled sampled points. This is not object precision or full-object segmentation.
- About 81% of primary clean validation false-positive observations have NN gaps >=8 cm. Geometric gaps do not establish their cause.
- No construction acquisition poses/visibility metadata. Intensity and horizontal shape are not semantic or occlusion labels.
- No verified construction work-item identities, persistent region association, installed quantities, or completion percentages.

## Next work

1. Independently review a small construction event set; keep uncertainty/event annotations separate from benchmark labels.
2. Freeze a small candidate set and reserve fresh construction development/test evidence.
3. Run a matched V2/hierarchy ablation holding sampling, augmentation, loss, regularization, checkpoint objective and fit budget fixed; confirm on another seed/fold.
4. Investigate acquisition metadata and coverage uncertainty, counting abstentions on full support.
5. Evaluate dense/block inference in a predefined validation area; larger context requires new support guards.
6. Add past-only temporal persistence and work-item/BIM association after reliable region detection.

Existing report/presentation/demo material describes older study stages and does not supersede current result receipts. This compact GitHub snapshot contains summaries and selected weights; complete arrays, redundant checkpoints and duplicate archived sources remain local.
