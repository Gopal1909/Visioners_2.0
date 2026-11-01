# Four-minute demonstration video

**Status:** storyboard and narration only. No screen recording or YouTube upload is claimed.

**Target:** 4:00, 1920×1080, 30 fps, landscape. Record the local viewer and a small number of deck figures. Use a clear microphone, modest cursor movement and no background music over speech. Increase browser scaling until labels are readable. Start and preload the viewer before recording. Do not film a terminal installing packages or a model-training run.

Use the actual control names and default examples in `LIVE_DEMO_GUIDE.md`; the actions below describe the required view, not a claim that every condition exists for every pair. Only select combinations offered by the viewer. Predictions are saved measurements, not newly computed inference.

## 0:00–0:25 — Problem and genuine scene
**Screen:** opening slide, then the earlier/later construction view. Slow cut rather than decorative animation.
**Narration:** “Can a computer tell real construction change from a scanning error? We are Vishwanath Venugopal and Shaik Koduru Asif from IIT Madras. Our CV5570 project studies robust 3D change detection. These are genuine construction scans. The challenge is that misalignment, noise, sparse measurements and occlusion can also make geometry look different.”
**Presenter:** Vishwanath.

## 0:25–0:55 — Dataset, labels and scope
**Screen:** dataset timeline figure, then ground-truth mode in the viewer. Show both scan directions and colour legend.
**Narration:** “Our quantitative data contain four acquisition epochs at one construction site, forming six temporal pairs. Changed points in the earlier scan are candidate removals; changed points in the later scan are candidate additions. The labels describe geometry, not completed work items. We use buffered spatial folds, but this is not independent-site validation.”
**Action:** choose a supported pair, ground truth, and both directions. Point to the legend; do not rely on colour alone.

## 0:55–1:30 — V2 and baseline comparison
**Screen:** V2 prediction view, then C2C on the same available points, then the clean-results slide.
**Narration:** “C2C thresholds nearest-neighbour distance. Our retained V2 model combines local geometric evidence across multiple scales and is trained with weighted binary cross-entropy. Here we switch between saved predictions on the same scene. In the main same-site spatial comparison, selected V2 achieved mean binary macro-F1 of 0.6537, compared with 0.5486 for calibrated C2C. These are F1 scores, not accuracy.”
**Action:** switch method without changing the pair or condition. A local example illustrates a decision; it does not reproduce the full benchmark average.
**Transition:** “Asif will now show where the method succeeds—and where it still fails.”

## 1:30–2:00 — Real errors and uncertainty
**Screen:** the viewer's tested false-positive/false-negative focus example; alternate prediction and ground truth.
**Narration:** “A convincing demonstration must also show errors. This example compares a saved decision against the point-change label. False positives can arise where geometric support differs, while a missed change is a labelled changed point predicted unchanged. A large geometric gap does not prove occlusion, and a confidence value is not a guarantee of correctness.”
**Presenter:** Asif.
**Action:** use the guide's focus preset and explain only the statistics actually shown. If no confidence is displayed, do not invent it.

## 2:00–2:35 — Controlled degradation
**Screen:** clean/degraded comparison in the viewer, followed briefly by the verified robustness plot.
**Narration:** “We also perturb real scans under controlled conditions. This saved example is not a measured construction event. In the fold-zero combined translation-and-rotation test, robust V2 scored 0.6176 macro-F1, calibrated C2C 0.5720, and fixed-threshold C2C 0.4181. The matched clean robust score was 0.6263. This is a scoped robustness finding, not proof that learning always wins.”
**Action:** select **Actual combined-large · seed 42**, then optionally **Actual 25% retention · seed 42**. Read the displayed configuration and model. The degraded viewer uses saved selected fold-zero clean-variant predictions (threshold .98), not the fixed robust model behind the plot's 0.6176. Say: “This interactive example uses the selected model; the following plot separately evaluates the robust variant.” Visible subsets and support differ; they do not yield the quoted aggregate scores. Return to Clean for regions.

## 2:35–3:10 — Candidate regions
**Screen:** region overlay, select one component, rotate slowly to show bounding box and available region statistics.
**Narration:** “We aggregate saved detections into candidate spatial regions to make review easier. The export contains 191 candidate components: 80 additions and 111 removals. They are not 191 verified construction objects. Eighty-three contain only unchanged-labelled sampled points. Bounding boxes help locate hypotheses, but their volume is not installed material quantity.”
**Action:** highlight one region, explain whether the viewer shows a rendering subset or all exported sampled cores, and avoid naming a material or work item without independent labels.

## 3:10–3:35 — Practical limits and ConSLAM
**Screen:** genuine ConSLAM visualization and an implemented/future workflow slide.
**Narration:** “ConSLAM provides qualitative mobile-scanning context, not labelled change-accuracy validation. Our next step is independent event review and fresh construction data. BIM association, persistent verified tracks and progress quantities require additional evidence. We have not implemented verified earned-value or completion-percentage measurement.”

## 3:35–4:00 — Honest closing
**Screen:** return to attractive construction prediction view, then conclusion slide.
**Narration:** “We built and evaluated geometric and learning-based change detectors, demonstrated a useful same-site V2 improvement, and examined errors and controlled stress. The browser interaction is live, but all model predictions shown here are precomputed research outputs. The result is an evidence-backed change-localization prototype—not an automatic certification of construction progress. Thank you.”
**Presenter:** Asif, with both names on the closing slide.

## Recording sequence
1. Run the root launcher as documented in `LIVE_DEMO_GUIDE.md` and verify the correct page.
2. Rehearse all views once so local assets are loaded; reset to the start pair/camera.
3. Open the final PPTX or PDF and prepare the dataset/results/robustness/ConSLAM/conclusion views.
4. Record a continuous rehearsal or short segments in the order above. Use simple cuts; no fake loading/inference animation.
5. Check final runtime is 3–5 minutes, preferably about 4:00. Trim pauses, not scientific qualifications.
6. Watch the entire export and check audio, colours, legends and model/condition labels.
7. Save the recording separately. Upload to YouTube only with explicit user approval; retain required dataset attribution in its description.

## Offline fallback
If the viewer fails during recording, use genuine saved figures from `assets/` in the same sequence: scan pair → labelled/predicted scene → error atlas → robustness graph → candidate regions → ConSLAM. Say “These saved visualizations show…” rather than pretending the video demonstrates working interaction. Include a visible “Saved-output walkthrough” label. The final presentation PDF is also an offline fallback.

## Attribution and disclosure for the video description
“CV5570, IIT Madras. Vishwanath Venugopal CE23B124 and Shaik Koduru Asif CE23B114. Quantitative construction data: MultiChange3D, Wang et al., ISPRS Annals XI-2-2026, 639–646; dataset-derived visualizations retain applicable CC BY-NC-SA 4.0 attribution and terms. Qualitative mobile LiDAR context: ConSLAM, Trzeciak et al., 2023; provider terms apply. Interactive visualization uses precomputed predictions and sampled data. No verified BIM matching, installed quantities, or project-completion percentages are claimed.”
