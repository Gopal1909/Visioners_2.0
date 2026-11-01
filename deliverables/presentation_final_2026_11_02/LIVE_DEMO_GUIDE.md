# CV5570 live demonstration

## Launch and stop
Open Windows PowerShell and run:

```powershell
Set-Location D:\CV_Project
.\run_live_demo.ps1
```

If a trusted local script is blocked by execution policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\CV_Project\run_live_demo.ps1
```

Open http://127.0.0.1:5570/ if the browser does not appear. The launcher checks asset SHA-256 values and HTTP readiness, binds only to loopback, and reuses its verified existing process. Python defaults to the installed `dl_env` interpreter. Override with `-Python` only if needed; the server uses the standard library. No training or inference runs.

Stop safely:

```powershell
.\run_live_demo.ps1 -Stop
```

The stop command checks process path, executable, ownership token and creation time before stopping its specific PID. Never stop all Python processes. If another service owns the port, use `-Port 5571` after stopping this demo's previous instance.

## Five-minute walkthrough

### 0:00–0:45 — Actual earlier/later data
Keep **Epoch pair: 01**, **Saved condition: Clean**, **Class / error filter: All sampled cores**. Select **Evidence layer: Ground truth**. Change **Epoch / direction** between Earlier and Later, then Both. Explain: green later changes are candidate additions; orange earlier changes are candidate removals; grey is unchanged. The viewer has 7,500 sampled original cores per epoch pooled across spatial folds, not every raw point.

### 0:45–1:40 — Predictions and baseline
Switch **Evidence layer** from Ground truth to **V2 · fold-selected primary**, then **C2C · calibrated saved baseline**, then V2. Read the method/threshold panel. C2C has saved unavailable entries on some clean displayed cores; inspect **COMMON EVALUATION SUPPORT**, do not score an unavailable prediction as unchanged. This interface does not recompute the benchmark metrics. For pair 01, 15,000 points display and 12,278 have common saved support.

### 1:40–2:25 — Show errors honestly
Press **FP / FN focus** to isolate error points and fit the camera. Alternatively select **False positives vs GT** or **False negatives vs GT** under Class / error filter. Pair 01 clean saved V2 has 448 FP and 583 FN over both epochs. Explain that these are point-label disagreements, not independently reviewed construction events. Restore **All sampled cores** and press **Reset camera**.

### 2:25–3:20 — Saved degradation
Choose **Actual combined-large · seed 42** under Saved condition. Read the method panel: this demo uses the saved selected fold-zero model (clean variant, threshold .98), not the fixed robust seed42 model used for the talk's .6176 aggregate. Combined-large is translation plus rotation. These are actual transformed saved XYZ, not a drawing or new inference.

Try **Actual 25% retention · seed 42** and inspect the support text. Missing observations and unavailable decisions are not verified removals. Degraded exports use a smaller fold-zero sampled support, so do not describe changes in visible counts as model performance. Return to Clean before showing regions.

### 3:20–4:20 — Candidate regions
Enable **Clean candidate bounding boxes**. In **Candidate region**, select `removed_001`. Pair 01 shows 436 sampled positive cores, 414 changed-labelled and 22 unchanged-labelled, with a mean score and XYZ bounding box. Mean scores are not calibrated probabilities; point purity is not object precision. Across the export there are 191 candidate components, not 191 verified work items. Bounding box dimensions are not installed material quantities.

### 4:20–5:00 — Interact and conclude
Drag the scene to rotate, use the mouse wheel to zoom, and Shift-drag or right-drag to pan. **Point size** improves projector visibility; **Reset camera** returns to the scene. Optionally select pair 03 or 23 to show this is not one screenshot. Close with: “The interaction is live; predictions are precomputed. This is a review-oriented geometric change prototype, not verified BIM matching or completion estimation.”

## Recovery
- Wait until status says **Ready · saved evidence** after a selection. If loading fails, refresh and return to pair 01 / Clean.
- For an empty error-filter view, restore All sampled cores and Reset camera; some subsets legitimately have no matching errors.
- If startup reports a missing or altered asset, do not run research drivers. Restore the generated demo asset from this delivery or rerun only its isolated `demo/build_demo.py` with the original arrays available.
- If the server is unresponsive, use `-Stop`, then relaunch. Do not delete state merely to bypass ownership validation.
- If Python is unavailable, pass the explicit installed interpreter with `-Python`; no numerical packages are needed to serve already-generated assets.
- If the viewer fails in the classroom, show the final PDF and genuine figures under `deliverables\presentation_final_2026_11_02\assets`. Identify these as saved visualizations, not functioning interaction.

## Offline and provenance
All scripts, CSS and 18 pair/condition payloads are local. No CDN or model downloads are needed. Source hashes and prediction/label alignment receipts are linked inside the viewer under **Source manifests + SHA-256 receipts**. Clean candidate components use selected out-of-fold predictions; degraded views use fold-zero selected-model predictions. Do not substitute the historical submission checkpoint or claim the degraded viewer is the fixed robust stress model.

For rehearsal use `deliverables\presentation_final_2026_11_02\REHEARSAL_CHECKLIST.md`. The four-minute video script is a plan only; no video has been recorded or uploaded.
