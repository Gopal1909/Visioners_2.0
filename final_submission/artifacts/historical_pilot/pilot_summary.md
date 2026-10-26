# Bounded M3C2-style pilot: findings

25 October 2026. The implementation and requested clean pilot are complete. The geometry passes **19 synthetic tests**, but this pilot does **not** support an accuracy advantage over C2C.

Exactly 2,000 original cores per scan were selected across all occupied 2 m 3D tiles using seeds 42/43. Full clouds supplied all neighbourhoods. The primary configuration was fixed before scoring: PCA radii 0.10/0.20/0.40 m, lowest-curvature scale with adequate eigengap and cross-scale agreement, projection radius 0.10 m, half-length 1.0 m, effect floor 0.02 m, and at least 10 projected neighbours per epoch.

| Quantity | M3C2-style | C2C on identical valid cores |
|---|---:|---:|
| Pooled 3-class macro F1 | 0.3529 | 0.6274 |
| Pooled macro mIoU | 0.2212 | 0.4978 |
| Directional macro F1 | 0.2673 | 0.4773 |
| Accuracy on valid support | 42.30% | 87.29% |
| Correct valid / all 4,000 cores | 26.80% | 55.30% |

Both score the same 1,246/1,288 directional cores: **63.35% coverage**. M3C2 abstains on **36.65%**. C2C itself has zero abstentions; common-support exclusions are reported separately.

| method | scan | precision | recall | F1 | changed_IoU |
| --- | --- | --- | --- | --- | --- |
| M3C2-style | 0 | 0.1479 | 0.9766 | 0.2569 | 0.1474 |
| M3C2-style | 1 | 0.1612 | 1.0000 | 0.2776 | 0.1612 |
| C2C common support | 0 | 0.3892 | 0.5078 | 0.4407 | 0.2826 |
| C2C common support | 1 | 0.4586 | 0.5845 | 0.5139 | 0.3458 |

The LoD uses population projected spread and `1.96 * (sqrt(s0²/n0 + s1²/n1) + registration_error)`, checked against [CloudCompare v2.13.2 / revision 49dbbb6](https://github.com/CloudCompare/CloudCompare/blob/v2.13.2/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp). This is **M3C2-style**, with no independent reference-application output validation. Registration error is a separate parameter, set to zero here: uncertainty is conditional on no modelled registration error. The 1.64 cm historical NN median is not used as registration RMS. Hypothetical 1/2 cm registration sensitivities are saved separately.

The nine radius/effect configurations, raw signed distances, LoDs, support/normal diagnostics, explicit unknown reason masks and per-scan predictions are saved. Geometry took **16.10 s** for normals/all three radii including shared trees; C2C search took **2.38 s** including the same tree build. Sampled process RSS peaked at **941 MiB** through the main plots; these are shared-process measurements, not isolated algorithm peaks.

Limitations: primary false positives are frequent; changed-class valid coverage is only 39.5%/36.6%. Correlated samples, unmodelled alignment uncertainty, unreliable normals and mixed surfaces can undermine conditional LoD. Halving cylinder length changed some distances by 0.427 m, despite zero median change, supporting concern about distant surfaces in some cylinders. Spatial balancing overweights sparse regions, so sampled scores do not estimate full-pair accuracy. No held-out or cross-scene claim, and no score-based primary retuning.

See [full report](pilot_report.md), [metrics](configuration_metrics.csv), [3D view](pilot_3d.png), [sweep figure](configuration_comparison.png), and [failure examples](diagnostic_examples.png). Historical full-pair C2C stays separately labelled. Broader Stage 4 matched robustness and independent software validation remain outstanding. Original data and Stage 2/3 outputs are preserved; no model training or full-cloud M3C2 sweep was run.
