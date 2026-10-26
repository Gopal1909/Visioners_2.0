# Stage 4: bounded M3C2 implementation and comparison plan

Prepared 25 October 2026. **Plan only: M3C2 has not been implemented or evaluated.** The priority before the 2 November 2026 presentation is a defensible geometric comparison on MultiChange3D BikeParkingConstruction `pair_01`. ConSLAM stays qualitative; the learning model remains optional.

## Deliverable and reuse

Add one focused `code/m3c2_baseline.py` with a smoke-test mode, explicit parameters and separate output directories. Reuse `load_point_cloud_ply`, `compute_binary_metrics` and `compute_combined_3class_metrics` from `code/c2c_baseline.py`. Reuse Stage 3 perturbation helpers, including their exact seeds and evaluation masks, rather than copying them. Importing these modules is protected by their main guards; do not invoke their full experiment entry points to obtain helpers.

Use the existing Windows interpreter `C:\Users\prani\anaconda3\envs\dl_env\python.exe` and its NumPy/SciPy/plyfile stack. Normal estimation and neighbourhood searches can run on CPU; CUDA is not required for this baseline. Do not create another environment. The interpreter file exists, but launch from the audit's WSL shell failed with `UtilBindVsockAnyPort: socket failed`; current GPU/package execution is therefore unverified. Begin the implementation task in a working Windows terminal with `conda activate dl_env`, confirming the interpreter and required imports. No reinstall is justified by that WSL transport failure.

Proposed outputs under a **new**, versioned `results/pair_01/m3c2/` run folder: `run_config.json`, selected original point indices, raw directional signed distances and uncertainty, validity/reason masks, selected scales, neighbour counts, predictions, clean/robustness metrics CSVs, comparison plots and a limitations report. Store package versions, source/dataset identifiers, seeds, parameter-selection support, sample sizes, runtime and peak RAM. These paths describe future outputs, not files produced by this audit.

## Method and decision rule

M3C2 estimates local normals, projects neighbourhoods from both epochs onto the same normal, and compares their mean positions. The original reference is [Lague, Brodu and Leroux (2013)](https://arxiv.org/abs/1302.1183). This is a local surface-separation measurement, so do not substitute point-to-plane nearest-neighbour distance and call it M3C2.

For an original source point `c`, estimate its unit normal `n`. Within each epoch collect points in the **same cylinder** centred at `c`: `t = dot(p-c, n)`, transverse distance `<= r_projection`, and `abs(t) <= L`. Compute `delta = mean(t_comparison) - mean(t_source)`, projected spread `s_source, s_comparison`, and counts `n_source, n_comparison`.

For valid neighbourhoods, use the conventional roughness/count uncertainty rule:

```text
LoD95 = 1.96 * (sqrt(s_source²/n_source + s_comparison²/n_comparison) + reg_error)
changed = abs(delta) > max(LoD95, minimum_effect_m)
```

The [CloudCompare implementation](https://github.com/CloudCompare/CloudCompare/blob/master/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp) uses this placement of the registration term. Before claiming implementation equivalence, pin its revision and confirm the spread estimator and neighbourhood conventions against it. Spatial correlation, mixed surfaces and imperfect normals limit the nominal statistical interpretation.

`minimum_effect_m` is a separately declared application effect-size floor. Zero gives significance-only classification; a positive floor is an explicit classification extension. The fixed **C2C** rule remains Euclidean nearest-neighbour distance `> 0.075 m`; that value is not automatically an M3C2 parameter. No empirical M3C2 threshold has been selected yet.

Perform both directions, using source normals for each epoch. A significant result on an original Scan 0 point is a candidate removal, and on Scan 1 a candidate addition. These class names follow the existing directional label convention. The **sign** of `delta` describes displacement along the chosen normal; it does not by itself mean added or removed. Keep signed output for interpretation and use magnitude for binary change decisions.

## Normals and multiscale projection

1. Estimate normals by PCA of source-epoch spherical neighbourhoods. The smallest covariance eigenvector is the normal. Start with candidate **radii** `0.10, 0.20, 0.40 m`, at least 10 neighbours, finite covariance and a meaningful eigengap; expose quality limits in the configuration. These are initial engineering choices to assess on the subset, not validated dataset scales.
2. Choose a local normal scale using a declared geometric rule, such as the lowest normalized smallest eigenvalue among adequate, stable neighbourhoods. Inspect cross-scale angular stability and edge/mixed-surface failures. Freeze that rule before scoring evaluation labels. Record the selected radius and rejected-normal count. Recompute source normals from the actual perturbed source geometry; do not use clean normals as an undocumented oracle under noise/sparsity.
3. Orient normals deterministically by making their largest-magnitude component positive, with a fixed tie-break. This preserves signed reproducibility without assuming +Z works for walls. It is not a physical outward-normal convention. Test sign flips separately: `abs(delta)` and predictions should remain invariant.
4. Calculate projection distances and LoD at **radii** `0.05, 0.10, 0.20 m`. Report all candidate-scale subset results, then freeze one projection radius for the primary comparison. Do not choose the best scale per point using ground truth. Start with axial **half-length** `L=1.0 m` and at least 10 projected neighbours per epoch; inspect whether the cylinder crosses unrelated surfaces. Inspect `L=0.5 m` only as a subset diagnostic before freezing the length. A finite cylinder defines the measurable displacement range.
5. Query neighbourhoods in small batches and process core points in chunks. Avoid a full pairwise distance matrix and avoid accumulating Python neighbour lists for every point. For an initial implementation, bound cylinder candidates with a sphere of radius `sqrt(r_projection² + L²)` and apply exact axial/transverse filters. Profile this conservative search: if dense scenes make it costly, use spatial tiles with halos or an efficient cylinder index before attempting full-cloud execution. Do not silently truncate neighbours to satisfy memory limits.

[CloudCompare's parameter documentation](https://cloudcompare.org/doc/wiki/index.php?title=M3C2_(plugin)) uses normal/projection **diameters**, while this proposed Python interface uses radii. Explicitly convert these when checking an external result. The [PDAL reference](https://pdal.io/en/stable/stages/filters.m3c2.html) provides a useful radius/half-length interface comparison. No installation of either external package is required for this plan.

## Invalid support and comparison fairness

Missing counterparts, insufficient neighbours or unreliable normals produce an **unknown** result with a reason code, not automatically unchanged or physical change. Missing projection support can indicate removal, occlusion or a search length that is too short. Treating every such point as changed would be an additional hybrid detector and must be named and evaluated separately.

For each direction report valid/unknown counts and valid coverage by ground-truth class. Give both methods' metrics on the exact same valid original indices; a C2C full-pair score cannot be compared directly to an M3C2 valid-subset score. Retain the historical full-pair C2C reference alongside it, clearly labelled. For a full-support deployment summary, report conservative accuracy lower bound `correct_valid / total` (all abstentions counted as errors) and abstention rate; do not hide unknown points inside the three classes or claim a full-point macro F1 for an abstaining output. Full-point classification is a separate policy decision if it becomes necessary.

Initially evaluate M3C2 directly at original source indices, not arbitrary voxel centroids. If all-point processing is impractical, use a reproducible sample of original indices across spatial tiles for **both** methods and label this a sampled comparison. Do not transfer predictions to millions of points by nearest-core interpolation and present them as directly evaluated M3C2 measurements. On sparsity/occlusion, score only retained Scan 1 points, preserving original-index mappings.

## Subset checks and parameter selection

First validate synthetic identical planes, a known normal offset with analytical distance, tangential translation of a plane interior, controlled noise, normal sign reversal, coordinate translation invariance, and empty/degenerate neighbourhoods. Verify both direction/label mappings. These checks validate geometry and invalid-data handling; they do not establish dataset accuracy.

Next choose a spatially distributed, label-independent smoke sample of approximately **2,000 core points per epoch**. Query complete local neighbourhoods from both input clouds, or use spatial windows with a halo covering the largest normal radius and cylinder extent. Reading a memory-mapped PLY with `plyfile` or a bounded window is appropriate. Independently taking the first N points of each scan can destroy overlap and is unsuitable for an accuracy comparison. Save selected original indices, sampling rule and seed. Check signed-distance plots, support coverage, normal quality, runtime and memory before scaling.

For the bounded pilot, fix the normal-scale rule and evaluate three projection radii with effect floors `0, 0.02, 0.05 m` (**nine configurations**). Keep `L`, minimum counts and confidence multiplier fixed after the geometry checks. There is no independently calibrated registration uncertainty in the audited outputs: `0.0164 m` historical median unchanged nearest-neighbour distance is not registration RMS. If `reg_error=0` is used initially, label LoD as conditional on no modelled registration uncertainty. Show `0.01` and `0.02 m` as explicitly hypothetical sensitivity settings on the selected subset configuration only; they are not measured calibration values. Do not feed the known injected transformation magnitude into the primary detector as privileged uncertainty information.

Select configurations on documented calibration spatial tiles; keep separate evaluation tiles with buffers preventing shared neighbourhoods across the split. Report this as within-pair validation, not cross-scene testing. If spatial separation is not feasible in the time available, disclose same-pair exploratory tuning, display the small sweep and avoid held-out claims. The historical C2C threshold was itself selected on the full pair, so any spatial evaluation using that reference still inherits that selection history. Freeze M3C2 configuration before robustness tests; never retune at each stress level.

## Comparison and completion gates

Use existing binary precision, recall, F1, changed IoU, and the pooled directional three-class macro F1/mIoU, together with coverage, runtime and memory. Match masks, point indices and perturbations across methods. Use cached C2C clean distances from `c2c_predictions.npz` when suitable; its keys are `distances_01`, `distances_10`, `gt0`, `gt1`, `best_threshold`, `pred0_best`, `pred1_best` (verified from the writer; archive payload not reloaded in this audit). Perturbed C2C subset distances may require fresh queries; saved aggregate full-cloud CSVs cannot supply metrics for arbitrary subsets.

After the subset passes, progress to approximately 20,000 cores per epoch and estimate full-run cost from measured throughput and memory. For a first bounded robustness comparison, include clean plus existing levels: translation `0.10, 0.20 m`; rotation `0.5, 1.0 degrees`; retention `0.25, 0.10`; per-axis noise `0.02, 0.075 m`; and cylinder deletion radii `4.6, 7.6 m` with both protocols. Add combined registration and the full saved sweeps only if measured cost permits. Reuse the fixed seed/directions and mean-based deletion centre, and disclose that these are single realizations.

For occlusion A retain all requested Scan 0 indices; for B exclude indices inside the known cylinder. Keep Scan 1 queries against the full original Scan 0 support in both protocols, matching Stage 3. Recompute the common valid support within each protocol and publish unknown/coverage counts. No mask should be inferred from the change labels.

Stage 4 is complete when geometry checks pass, exact parameters and support are saved, clean and matched robustness comparisons are produced with coverage, and visual examples show both successes and failures. If an available CloudCompare installation can independently check a small exported subset, use it and save its configuration; otherwise describe the implementation as **M3C2-style, not independently validated against the reference software**. No accuracy improvement is promised in advance.

Suggested order: 25–26 October implementation and smoke checks; 27 October configuration selection; 28–29 October matched comparisons and failure inspection; 30 October figures and report; 31 October–1 November presentation preparation and buffer. These are proposed work allocations, not scheduled automations. Begin the optional Siamese PointNet model only if the geometric comparison and presentation artifacts are already complete; this audit does not authorize training.

**Exact next action:** in the working Windows `dl_env`, implement the bidirectional PCA-normal/cylinder-projection/LoD core with explicit unknown masks in `code/m3c2_baseline.py`, then run synthetic checks and the saved 2,000-core-per-epoch smoke subset. Do not launch a full-cloud parameter sweep first.
