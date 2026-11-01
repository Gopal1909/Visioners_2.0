# Final rehearsal checklist

## Session timing
- [ ] Rehearse the main deck in 10 minutes, the live demo in 5, and reserve 3 for questions: 18 minutes total.
- [ ] Both presenters can explain the main metric, source data, V2 loss, shared-site limitation and why the demo uses saved predictions.
- [ ] Rehearse the speaker handoff using `speaker_notes.md`, not the older eight-minute branch script.
- [ ] If Q&A is outside the 18-minute slot, use the alternative timing in `presentation_timing.md`; do not add unsupported results to fill time.

## Files and provenance
- [ ] Open this delivery's `CV5570_Final_Presentation.pptx` and `.pdf`; do not accidentally open the historical deck.
- [ ] Review `evidence_reconciliation.json` / `.md` and the per-slide source notes.
- [ ] Confirm the three F1 aggregations are labelled separately: binary macro, changed class, pooled directional three class.
- [ ] Describe severe stress with its actual model/fold/support scope, not as a universal benchmark.
- [ ] Call extracted clusters candidate geometric regions, not verified construction objects or quantities.

## Classroom setup — 20 minutes before
- [ ] Plug in laptop power. Disable sleep, notifications and unrelated auto-updates for the session using normal user settings.
- [ ] Check projector resolution, slide aspect ratio, text readability and browser zoom.
- [ ] Keep PPTX, PDF, `LIVE_DEMO_GUIDE.md`, the viewer, and saved fallback figures ready.
- [ ] Close irrelevant windows and ensure no credentials/private pages appear during screen sharing.
- [ ] Run the launch command in the root demo guide from a fresh PowerShell window.
- [ ] Confirm the page identifies V2 saved predictions and displays the expected pair, method and condition.
- [ ] Test the browser after disabling external network access; all demonstration assets must remain local.
- [ ] Launch a second time to confirm it reuses the server rather than opening another process on the port.

## Demo drill
- [ ] Show earlier/later views and explain the direction of additions/removals.
- [ ] Switch ground truth and predictions; identify the meaning of colours.
- [ ] Compare C2C and V2 using the same available support.
- [ ] Focus a real error example; do not imply the surrounding region has semantic human verification.
- [ ] Compare clean and saved controlled degradation, naming the actual perturbation.
- [ ] Select a candidate region, show its bounding box and available statistics.
- [ ] Rotate, zoom and pan; reset the camera before handing over.
- [ ] State that the interface is live but predictions are precomputed; no model training runs.
- [ ] Follow the exact tested control sequence in `LIVE_DEMO_GUIDE.md`.

## Failure recovery
- [ ] Practice refreshing and resetting the viewer.
- [ ] Know the launcher shutdown command from the guide; stop only the server owned by this demo.
- [ ] If startup fails, read the reported missing asset/dependency rather than running research drivers.
- [ ] If interactivity fails during the talk, immediately show the saved visualizations in `assets/` and describe them as saved figures.
- [ ] Keep PDF slides as the fallback if PowerPoint rendering differs.

## Four-minute recording
- [ ] Follow `DEMO_VIDEO_SCRIPT.md`; rehearse to 4:00 before recording.
- [ ] Use 1920×1080, 30 fps, microphone check, and readable browser/slide scaling.
- [ ] Show only the deck/demo, hide notifications and do not expose private paths or accounts unnecessarily.
- [ ] Watch the full recording, verify labels are readable and audio is clear.
- [ ] Recording and YouTube upload require the user's action/approval; this package does not claim a video already exists.

## After rehearsal/presentation
- [ ] Stop the demo server using the documented safe shutdown command when no longer needed.
- [ ] Preserve original research artifacts; no training, benchmark reruns or repository push are needed.
