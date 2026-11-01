# CV5570 final presentation and live demonstration

Separate authored delivery for **2 November 2026**. Original research code, datasets, checkpoints and historical results remain in place.

## Open these files
- [Editable presentation](CV5570_Final_Presentation.pptx)
- [Presentation PDF](CV5570_Final_Presentation.pdf)
- [Full speaker script](speaker_notes.md)
- [Session timing](presentation_timing.md)
- [Live demo instructions](LIVE_DEMO_GUIDE.md)
- [Four-minute video storyboard](DEMO_VIDEO_SCRIPT.md) — recording not created or uploaded
- [Viva preparation: 30 questions](Q_AND_A_PREPARATION.md)
- [Rehearsal checklist](REHEARSAL_CHECKLIST.md)

## Source reconciliation
The merged GitHub checkout at commit `5477619721ce967ac1f26fa2c49a1eaa0998ed6e` is the latest intended **communication snapshot**. It is not a new model or experiment. Numerical authority comes from original experiment CSV/JSON, evaluation/selection manifests and preserved model artifacts in the full local archive and compact checkout. See the evidence reconciliation files and per-slide notes for exact provenance. Unsupported claims from the merged report are not carried forward. The report itself is not rewritten.

The existing historical offline viewer is not relabelled as V2. The new demonstration uses saved outputs with explicit model/condition provenance. Browser interaction is live; inference and training are not.

## Timing
Target: 10-minute main presentation + 5-minute demo + 3-minute Q&A = 18 minutes. Follow `presentation_timing.md` and the slide scripts for handoff and alternate timing.

## Launch
From a fresh Windows PowerShell session:

```powershell
Set-Location D:\CV_Project
.\run_live_demo.ps1
```

If local execution policy blocks a trusted script, use a process-only override:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\CV_Project\run_live_demo.ps1
```

Use `LIVE_DEMO_GUIDE.md` for the tested walkthrough and safe shutdown. The local server must listen only on loopback; no internet access, training or dataset download is required for viewing generated demonstration assets.

## Verification
`presentation_validation.json`, `demo/demo_validation.json`, and `delivery_validation.json` record the checks performed. The structural checker can be rerun with the configured project interpreter:

```powershell
& 'C:\Users\prani\anaconda3\envs\dl_env\python.exe' D:\CV_Project\deliverables\presentation_final_2026_11_02\verify_delivery.py
```

Do not interpret historical 83-test receipts as a newly executed test suite. No model retraining or expensive benchmark rerun is needed to present this delivery.

## Attribution
Authors: Vishwanath Venugopal CE23B124 and Shaik Koduru Asif CE23B114; CV5570, IIT Madras. MultiChange3D dataset-derived visualizations retain applicable attribution and CC BY-NC-SA 4.0 terms; ConSLAM retains its provider terms. See the project's `THIRD_PARTY_NOTICES.md`. No GitHub push, YouTube upload or final video recording is performed by this delivery.
