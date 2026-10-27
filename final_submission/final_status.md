# Final Project Status & Submission Overview
**Date:** 2 November 2026  
**Course:** CV5570 Computer Vision for Construction Engineering (IIT Madras)  
**Attribution:** Vishwanath Venugopal (CE23B124) & Shaik Koduru Asif (CE23B114)  
**Status:** COMPLETE, AUDITED, AND FULLY SYNTHESIZED  

---

## Completed Verification Milestones

1. **Dual-Dataset Architecture:** Resolved the initial ConSLAM ground-truth dilemma by integrating MultiChange3D Bike Parking Construction (4 epochs, 6 pairs, ~5M points per scan) for quantitative benchmarking, retaining ConSLAM Sequence 3 for mobile SLAM qualitative site inspection.
2. **Comprehensive 6-Pair Benchmark:** Evaluated all 6 temporal pairs (`pair_01`, `pair_02`, `pair_03`, `pair_12`, `pair_13`, `pair_23`) using 5-fold buffered spatial cross-validation (2.22 m buffer gaps).
3. **Clean Site Superiority:** V2 Multiscale Cross-Scan Learning achieves **0.6537 Macro F1 / 0.3900 Changed F1**, decisively surpassing C2C (**0.5659 / 0.3000**) and M3C2-style (**0.2627 / 0.1632**).
4. **Field Robustness Stress Testing:** 318 controlled degradation conditions evaluated. Under Combined Large Stress (20 cm translation drift, noise, and decimation), C2C collapses to **0.4181**, while V2 Multiscale retains **0.6176 Macro F1**—a massive **+20.0 percentage point advantage**.
5. **Physical Work-Item Extraction:** Grouped positive predicted change points into **191 discrete 3D construction change regions** (80 additions, 111 removals) with bounding dimensions, establishing direct progress takeoff for 4D BIM.
6. **Multi-Modal Negative Control:** Evaluated LiDAR intensity with a shuffled-intensity control, confirming that 3D spatial geometry drives >98% of change detection discriminative power.
7. **Mobile SLAM Insights:** ConSLAM handheld sweeps reveal an 8.1 cm median trajectory shift and 54.8% distance threshold violations, explaining why deep cross-scan learning is mandatory for mobile inspection robotics.
8. **Automated Testing:** All 83 automated unit tests pass completely.

---

## Delivered Artifacts

- `final_submission/report/project_report.pdf`: 6-page conference-style research paper with high-res figures and tables.
- `final_submission/report/project_report.md`: Markdown manuscript.
- `final_submission/presentation/project_presentation.pptx`: 14-slide master presentation deck.
- `final_submission/presentation/speaker_notes.md`: Balanced speaker notes split between Vishwanath and Asif (4 minutes each).
- `final_submission/demo/change_demo.html`: Standalone interactive 3D WebGL change visualization.
- `results/learning_v2_coverage_2026-10-25_v1/`: Checkpoints, metrics, and figures.
