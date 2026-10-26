# Measured results by evaluation scope

The table below contains condition-specific common-valid scores on the new buffered same-scene spatial-test cores. Parameters stay fixed. C2C and learned exclusions follow the M3C2 validity mask for these rows. Full precision, directional precision/recall/F1/IoU, coverage, runtime and memory are in `artifacts/all_evaluated_results.csv`.

| Condition | C2C macro F1 | M3C2-style macro F1 | Learned + NN macro F1 | Common coverage |
|---|---:|---:|---:|---:|
| clean | 0.4118 | 0.2342 | 0.3817 | 66.88% |
| noise_0.020 | 0.3658 | 0.2166 | 0.3465 | 65.10% |
| noise_0.075 | 0.4315 | 0.2077 | 0.3749 | 32.56% |
| occlusion_4.6 | 0.4118 | 0.2342 | 0.3817 | 66.88% |
| occlusion_7.6 | 0.4118 | 0.2342 | 0.3817 | 66.88% |
| rotation_0.5 | 0.4619 | 0.1222 | 0.4314 | 67.60% |
| rotation_1.0 | 0.3309 | 0.1047 | 0.3968 | 66.76% |
| sparsity_0.10 | 0.4396 | 0.2441 | 0.3101 | 48.65% |
| sparsity_0.25 | 0.4352 | 0.2513 | 0.3679 | 59.38% |
| translation_0.10 | 0.4476 | 0.1652 | 0.4033 | 66.94% |
| translation_0.20 | 0.3432 | 0.1127 | 0.4032 | 66.70% |

The main spatial stripe misses both deletion cylinders. Its identical occlusion scores are **not** occlusion robustness evidence. The separate exploratory all-scene geometric supplement supplies strict A and visibility-masked B results; do not combine that scope into the table's ranking.

At 7.6 m deletion radius, matched C2C/M3C2 macro F1 is 0.5925/0.3066 under A and 0.5950/0.3074 under B. Common coverage is 51.95% / 67.04%. The learned model is not scored on supplement training-region points as a held-out result.

Historical full-pair C2C macro F1 0.7359/mIoU 0.6046 and the earlier all-scene pilot C2C/M3C2 macro F1 0.6274/0.3529 are separate exploratory references. The historical threshold was selected on that pair. No cross-pair or cross-scene result exists.

For the learned model alone, calibration all-core macro F1 0.7649 versus spatial-test all-core 0.3936 exposes a large spatial gap. Its common-support score 0.3817 belongs to the three-method table above. All-core C2C macro F1 is 0.4457.

Use `artifacts/matched_clean_deltas.csv` for longitudinal changes on identical clean/current valid retained original indices. These isolate support changes more carefully than subtracting differently masked table rows.

Unknowns remain explicit. M3C2 clean unknown rate is 33.125%; C2C and learned classifiers have zero detector abstentions. The 66.875% common evaluation coverage is an evaluation restriction for those two classifiers.
