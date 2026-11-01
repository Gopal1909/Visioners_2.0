# CV5570 — exact presentation script

Main: 600 seconds. Appendix is not spoken during the timed talk.

Slide 1: Can AI distinguish real construction
progress from 3D scanning errors?
Speaker: Vishwanath Venugopal
Duration: 30s

Exact spoken script:
These are two real construction scans from the same site, acquired at different epochs. The geometry changes, but so does the way it is observed. Our question is whether AI can distinguish physical construction progress from scanning errors. We present Robust 3D Change Detection for Construction Progress Monitoring for CV5570. I am Vishwanath, presenting the problem and experimental design; Asif will present the results, limitations, and practical workflow.

Pointing cue: Point from earlier to later cloud.
Transition: First, consider what a scanner can make look like progress.
Likely question: Are these genuine project scans?

Source details:
D:\CV_Project\dataset\MultiChange3D\BikeParkingConstruction\pair_03\pair_03\pointCloud0.ply
D:\CV_Project\dataset\MultiChange3D\BikeParkingConstruction\pair_03\pair_03\pointCloud1.ply
Scope and numerical interpretation: Display-only stride samples; no full-cloud prediction. Labels color candidate direction, not semantic components.

---

Slide 2: A difference in points is not always a change in work.
Speaker: Vishwanath Venugopal
Duration: 40s

Exact spoken script:
The grey points represent the original site sample. Cyan shows four deliberately altered observations. A small registration error displaces otherwise static surfaces. Noise scatters measurements around surfaces. Lower density increases gaps between nearest neighbours, while occlusion removes visibility altogether. These examples are explanatory transformations of our genuine scan, not the measured stress benchmark. The important distinction is that a detector sees geometry and sampling, not the construction process itself. Missing evidence can mean a blocked view rather than demolition, and a displaced surface need not mean that anything was installed. This motivates comparing several mechanisms under controlled conditions.

Pointing cue: Sweep across the four panels, then underline absence versus removal.
Transition: We therefore need a bounded, testable research question.
Likely question: How are illustrative and measured perturbations distinguished?

Source details:
D:\CV_Project\dataset\MultiChange3D\BikeParkingConstruction\pair_03\pair_03\pointCloud1.ply
D:\CV_Project\code\multipair_evaluate.py
Scope and numerical interpretation: Illustrative projection offsets/noise are not stress severity values; genuine cloud sampled for display.

---

Slide 3: Learn local evidence. Test what survives.
Speaker: Vishwanath Venugopal
Duration: 40s

Exact spoken script:
We ask a narrower question than whether AI understands construction. Can learned local matching outperform geometric distance on sampled points from a construction scene, and what happens when the observations degrade? Our contributions are a directional, multi-epoch comparison, a controlled robustness analysis, and a review-oriented interpretation of predicted change. The final contribution is deliberately not automatic quantity takeoff. We distinguish a point prediction, a spatial candidate region, and a verified construction component. Those are different levels of evidence. The results support comparative same-site conclusions, with transparent caveats about the development history and the absence of an untouched final benchmark.

Pointing cue: Point to the three contribution cards in order.
Transition: The dataset determines what can actually be measured.
Likely question: What is new beyond a distance threshold?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\design.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\interpretation_scope.json
D:\CV_Project\code\learning_v2_model.py
Scope and numerical interpretation: Prior held-out results informed follow-up design; avoid untouched benchmark or zero-leakage claims.

---

Slide 4: Four epochs. Six pairs. Two directions.
Speaker: Vishwanath Venugopal
Duration: 40s

Exact spoken script:
The quantitative benchmark is BikeParkingConstruction from MultiChange3D: one site with four acquisition epochs. Every earlier–later combination gives six temporal pairs: zero–one, zero–two, zero–three, one–two, one–three, and two–three. We score both directions. Earlier points without corresponding later geometry are candidate removals. Later points without earlier counterparts are candidate additions. These remain geometric categories, not verified demolition or installation events. The six pairs share epochs and the five folds share a site, so the thirty pair-fold evaluations are not thirty independent studies. We use the benchmark labels for evaluation, while retaining ConSLAM only for qualitative context.

Pointing cue: Trace the epoch sequence, then the two directional cards.
Transition: Next is the path from local evidence to review.
Likely question: Are thirty evaluations independent?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\design.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\fold_0\protocol.json
Scope and numerical interpretation: One scene, four epochs, six overlapping pairs, 30 pair-fold evaluations, 60 scan directions; no independent-site claim from these means.

---

Slide 5: From scans to decisions—with evidence attached.
Speaker: Vishwanath Venugopal
Duration: 45s

Exact spoken script:
The workflow starts with paired scans and a spatially restricted set of original point cores. Each core receives local evidence from the two scans. The geometric branches use distances or local surface projection, whereas the learned branch combines multiscale patch information with bounded geometry features. The model produces a score, and a validation-chosen operating threshold turns that score into a point decision. A separate grouping step forms candidate review regions from positive sampled cores. Notice the output contract: we have not run or verified full-cloud inference here. A region is a grouping of sampled predictions, not a replacement for inspecting the underlying site geometry and its visibility.

Pointing cue: Follow the five boxes left to right; point to the output contract.
Transition: The three methods interpret local evidence differently.
Likely question: Which outputs are actually implemented?

Source details:
D:\CV_Project\code\multipair_data.py
D:\CV_Project\code\learning_v2_data.py
D:\CV_Project\code\learning_v2_model.py
D:\CV_Project\code\construction_change_regions.py
Scope and numerical interpretation: Scores and candidate grouping on sampled original held-out cores; no training or full-cloud inference performed by this build.

---

Slide 6: Three ways to interpret a missing neighbour.
Speaker: Vishwanath Venugopal
Duration: 55s

Exact spoken script:
C2C asks how far the closest point is. Its fixed reference threshold is seven point five centimetres, while the calibrated version chooses its threshold on validation data. M3C2-style asks whether separation along an estimated surface normal exceeds a local uncertainty criterion. We call it style because this is our implementation, not a claim of reproducing every official M3C2 feature. It can return unknown when local support is insufficient. V2 instead compares paired patches at radii of fifteen, forty, and eighty centimetres. A shared point encoder pools local information, combines cross-scan interactions, and fuses a geometry branch. Crucially, the implemented objective is equal pair-and-present-class weighted binary cross entropy, not focal loss.

Pointing cue: Point to each mechanism; emphasize the loss correction.
Transition: All comparisons depend on where cores are sampled.
Likely question: Is V2 PointNet++ or a focal-loss model?

Source details:
D:\CV_Project\code\c2c_baseline.py
D:\CV_Project\code\m3c2_baseline.py
D:\CV_Project\code\learning_v2_model.py
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\design.json
Scope and numerical interpretation: Custom shared multiscale PointNet metric fusion; 64 points/patch, radii .15/.4/.8m, weighted BCE. Fixed C2C .075m is separate from calibrated baseline.

---

Slide 7: Spatial separation—not random point splitting.
Speaker: Vishwanath Venugopal
Duration: 50s

Exact spoken script:
This is the actual fold-zero cell assignment, drawn from the protocol manifest, not a five-stripe illustration. The grid uses eight-metre cells. Sampled cores are at least two point five metres from cell boundaries, giving a five-metre minimum cross-split core distance. The requested counts per scan-direction and fold are three thousand training, twelve hundred validation, and fifteen hundred test cores. Six pairs, two directions, and five folds therefore give ninety thousand clean held-out point observations. Evaluation sampling is uniform, not class-balanced. Validation chooses thresholds and variants. Earlier held-out results did influence this follow-up design, so we describe these as same-site comparisons and do not claim zero leakage or an untouched benchmark.

Pointing cue: Point to inset core squares and the 2.5 m guard; then counts.
Transition: Asif will now show what this protocol reveals.
Likely question: Why not random splitting, and is this untouched?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\fold_0\protocol.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\design.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\interpretation_scope.json
Scope and numerical interpretation: Grid diagram uses actual cells, shown schematically. 1500*2*6*5=90000 clean held-out observations, repeated epochs not independent samples; requested training 180000 and validation72000 observations if all counts realized.

---

Slide 8: V2 improves clean comparison. Changed F1 stays modest.
Speaker: Shaik Koduru Asif
Duration: 55s

Exact spoken script:
I am Asif, and these are the clean results across the thirty pair-fold evaluations. The selected V2 workflow reaches a mean binary macro F1 of zero point six five three seven. Calibrated C2C is zero point five four eight six, fixed C2C is zero point five six five nine, calibrated M3C2-style is zero point four one four two, and the earlier selected workflow is zero point six two one two. V2 improves by eight point seven eight percentage points over fixed C2C. But the orange number is equally important: mean directional changed F1 is only zero point three nine zero zero. The model improves comparison; it does not reliably solve every change case.

Pointing cue: Read the teal bars, then explicitly point to orange changed F1.
Transition: Averaging choices explain why another F1 number looks higher.
Likely question: Why is changed F1 much lower?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\clean_comparison.csv
Scope and numerical interpretation: Exact V2 binary .6536591206208738; directional changed .3899887030904223. Calibrated C2C .5485634331059023; fixed .5658993214638721; calibrated M3C2 .414227689717096; earlier selected .6211906039032457. Unweighted same-site pair/fold/direction means, not accuracy.

---

Slide 9: One model. Different questions. Different F1s.
Speaker: Shaik Koduru Asif
Duration: 50s

Exact spoken script:
These three numbers are not interchangeable. The first averages binary macro F1 across the spatial and directional evaluations. The second isolates the changed class under the same averaging rule. The third first pools out-of-fold confusion counts and then scores unchanged, removed, and added classes. That pooled three-class macro F1 is zero point seven four zero five. It weights observations differently, so it should not be presented as a replacement for the first two. Also, twenty-one of sixty evaluated directions contain no labelled change. Under the retained convention, an absent-class F1 is zero. Even perfect predictions therefore yield a mean binary macro F1 of zero point eight two five, not one.

Pointing cue: Point across all three metrics; emphasize the different aggregation labels.
Transition: Next, we deliberately degrade the observations.
Likely question: Is pooled F1 an accuracy claim or a directly comparable mean?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\class_support_and_pooled_confusions.csv
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\metric_ceiling_diagnostic.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\clean_comparison.csv
Scope and numerical interpretation: Pooled three-class .7405075400072249; 39 positive-support and21 unchanged-only directions. Oracle ceiling diagnostic only; binary mean ceiling .825, directional changed mean ceiling .65.

---

Slide 10: Robust training helps under controlled misalignment.
Speaker: Shaik Koduru Asif
Duration: 45s

Exact spoken script:
The graph now changes scope: it uses fold-zero models only, evaluated on six pairs and three perturbation seeds per condition. We plot the actual mild and larger combined rigid conditions. Larger means a twenty-centimetre translation together with a two-degree rotation; it does not include an additional density or noise degradation. Robust V2 seed forty-two reaches zero point six one seven six, compared with zero point four one eight one for fixed C2C. The calibrated C2C result, zero point five seven two zero, is also shown to avoid overstating the gain against a weak operating point. This supports robustness of that trained variant, not an all-fold stress claim for the validation-selected workflow.

Pointing cue: Point to severe group; compare all three bars, not just fixed C2C.
Transition: The mistakes still need to be visible.
Likely question: Does this result cover all folds or a combined noise test?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\stress_condition_summary.csv
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\interpretation_scope.json
D:\CV_Project\code\multipair_evaluate.py
Scope and numerical interpretation: Fold0 only,6 pairs x3 perturbation seeds. combined_large .2m translation+2degrees rotation only. Robust seed42 .6176271501676411 SD .0029596161322436405; fixed .41806009084589607 SD .038035885217122714; calibrated .5719863237438338.

---

Slide 11: Expose the errors—not just the winning score.
Speaker: Shaik Koduru Asif
Duration: 45s

Exact spoken script:
These panels use genuine held-out later-epoch cores from pair zero–three, rather than invented error examples. Grey is the sampled context. Orange highlights false alarms, where the model predicts change on an unchanged-labelled core. Cyan highlights missed change. Purple shows a simple score-boundary diagnostic: scores within zero point one of one half. That purple mask is not a calibrated uncertainty probability, and it is not the fold-specific decision threshold. The panels show why a review workflow must preserve both the predictions and the original evidence. We should not convert every positive point into progress, and we should not interpret a low score as proof that no work occurred.

Pointing cue: Point to FP, then FN, then the explicitly labelled score diagnostic.
Transition: We can group evidence, but grouping does not create objects.
Likely question: Are these actual errors, and are the probabilities calibrated?

Source details:
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\pair_03_added_cores.npz
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\source_predictions_manifest.json
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\design.json
Scope and numerical interpretation: 7500 pair03 later held-out cores across five folds. FP=(prediction1,label0);FN=(prediction0,label1); boundary diagnostic abs(model_score-.5)<.1. No posterior calibration claim.

---

Slide 12: 191 candidate regions—not 191 construction objects.
Speaker: Shaik Koduru Asif
Duration: 35s

Exact spoken script:
We grouped predicted-positive held-out cores into spatial candidates using a fixed forty-centimetre connectivity radius and a minimum of three sampled positive members. Across six pairs, the saved output contains one hundred and ninety-one candidate regions. Eighty-three have only unchanged-labelled sampled members. That is a warning about false alarms, not an object-level accuracy statistic. The map shows the saved centroids for pair zero–three. Sampling gaps affect connectivity, and there is no verified component identity across pairs. These candidates can prioritize review; they cannot support installed-object counts, material quantities, or project completion percentages.

Pointing cue: Point to the centroid map, then the 83 warning.
Transition: A separate field dataset illustrates why visibility matters.
Likely question: What exactly does the count 83 mean?

Source details:
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\candidate_regions.csv
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\pair_summary.csv
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\design.json
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\verification.json
Scope and numerical interpretation: 191 CSV rows;83 rows with evaluation_changed_cores==0.90000 sampled observations. Fixed radius .4m,min3. No semantic object segmentation, object recall, verified quantities or temporal identity.

---

Slide 13: ConSLAM is a reality check—not a labelled test.
Speaker: Shaik Koduru Asif
Duration: 40s

Exact spoken script:
ConSLAM provides qualitative field context, not another labelled construction-change test. The two selected mobile frames are approximately four point eight four seconds apart within one acquisition sequence. The saved inspection compares raw and pose-transformed geometry and shows the TLS reference separately. It explicitly states that the coordinate relationship between that TLS reference and the normalized mobile poses is unverified. There are no official per-point change labels for this comparison, so we do not report accuracy, F1, or IoU. We also do not claim supervised V2 inference on these scans. Its value is to illustrate viewpoint, coverage, and alignment issues that a practical review tool must handle.

Pointing cue: Point to raw versus transformed context and the three no-claim lines.
Transition: We finish with what is implemented and what still needs development.
Likely question: Was V2 quantitatively evaluated on ConSLAM?

Source details:
D:\CV_Project\results\conslam_qualitative_2026-10-25_v2\inspection.json
D:\CV_Project\results\conslam_qualitative_2026-10-25_v2\conslam_qualitative.png
Scope and numerical interpretation: 4.839997s gap;12816/12935 selected-frame points. TLS104548932 points,50000 display sample. Pose/TLS relation unverified. No official labels, no learned inference demonstrated here.

---

Slide 14: Detect evidence. Review context. Verify progress.
Speaker: Shaik Koduru Asif
Duration: 30s

Exact spoken script:
The takeaway is simple: detect evidence, review context, and only then verify progress. V2 improves our same-site sampled-core comparison, and robust training helps in the bounded stress setting. The implemented outputs are scores, inspectable errors, and candidate grouping. Verified BIM association, quantities, and schedule integration remain future work. We will now use the five-minute offline demonstration to inspect the evidence rather than claim automatic progress certification. After the demonstration, we reserve three minutes for questions.

Pointing cue: Point to implemented versus future, then the demo transition.
Transition: Switch to the parent-owned offline demonstration.
Likely question: Can this already produce installed quantities or EVM?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\clean_comparison.csv
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\design.json
D:\CV_Project\results\conslam_qualitative_2026-10-25_v2\inspection.json
Scope and numerical interpretation: No BIM installation verification, quantities, EVM, or full-cloud inference claim. Main600s+demo300s+Q&A180s=1080s.

---

Slide 15: Exact metrics and the absent-class ceiling.
Speaker: Shaik Koduru Asif
Duration: 0s (appendix, not in main timing)

Exact spoken script:
The exact table distinguishes calibrated and fixed baselines. The pooled score uses a different aggregation rule; the oracle ceiling is a diagnostic, not a predictive method.

Pointing cue: Use the row and column labels before quoting a number.
Transition: Return to the relevant main result.
Likely question: Why is the oracle ceiling below one?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\clean_comparison.csv
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\class_support_and_pooled_confusions.csv
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\metric_ceiling_diagnostic.json
Scope and numerical interpretation: All exact values are read from CSV at build time. The upper diagnostic is absent-class dependent, not theoretical model performance.

---

Slide 16: V2 is a custom metric-fusion network.
Speaker: Shaik Koduru Asif
Duration: 0s (appendix, not in main timing)

Exact spoken script:
The custom network shares an eight-channel point encoder, combines max and mean pooling, and forms cross-scan interactions. A one-hundred-and-sixty-five-feature geometry branch is fused with the patch branch. Checkpoints minimize clean-validation BCE; validation selects the threshold and variant. It is not a PointNet++ or KPConv reproduction.

Pointing cue: Trace branch dimensions and selection rules.
Transition: Return to the clean-result slide.
Likely question: How were checkpoints and thresholds selected?

Source details:
D:\CV_Project\code\learning_v2_model.py
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\design.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\interpretation_scope.json
Scope and numerical interpretation: Radii .15/.4/.8m,64 points per patch;8→32→64 encoder;1536→128 patch head;165→64→64 geometry;192→64→1 head;dropout .1.Weighted BCE, not focal.

---

Slide 17: Follow-ups are diagnostics—not extra headline wins.
Speaker: Shaik Koduru Asif
Duration: 0s (appendix, not in main timing)

Exact spoken script:
The local follow-ups explore intensity, temporal evidence, and fusion, but they do not replace the headline selected workflow. Open_space is a separate already-exposed indoor diagnostic, not an untouched construction test. Calibrated C2C scored 0.8596 versus selected V2 at 0.7857. It is not pooled into construction-site means, and its data-identity caveat remains. The key missing evidence remains multi-site construction validation and verified component semantics.

Pointing cue: Keep the three evidence scopes separate.
Transition: Return to conclusion.
Likely question: Why not report a larger follow-up number?

Source details:
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\analysis\interpretation_scope.json
D:\CV_Project\results\learning_v2_coverage_2026-10-25_v1\external_open_space\metrics.csv
D:\CV_Project\results\construction_change_regions_2026-10-26_v1\design.json
Scope and numerical interpretation: Exploratory follow-ups not promoted; external evaluation exists but no numeric headline here without method-by-method context.
