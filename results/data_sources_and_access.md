# Data provenance and access audit

Checked 25 October 2026. Only existing local assets have been used; no new dataset files were downloaded in this continuation.

## MultiChange3D

Official project: https://3dom-fbk.github.io/multichange3d/

Official code: https://github.com/3DOM-FBK/multichange3d

Local construction data: `dataset/MultiChange3D/BikeParkingConstruction/pair_01/pointCloud0.ply` and `pointCloud1.ply`. Original archive `pair_01.zip` is also present. The two PLY files are 88,898,869 and 101,796,029 bytes, with 4,444,935 and 5,089,793 points. Their original `params` records (XYZ, intensity, cd_type) are preserved, read-only. Per-file SHA-256 and field schemas are in `results/study_2026-10-25_v1/dataset_manifest.json`.

Missing construction pair folders: pair_02, pair_03, pair_12, pair_13, pair_23. The official construction download link in the repository redirected the web client to Microsoft sign-in; no accessible file inventory or anonymous download was returned. No access controls were bypassed. Additional pairs therefore remain unavailable from this session. Disk capacity was sufficient (about 194 GiB free), so access rather than storage prevented download.

The optional Open Space MLS scene is listed as approximately 200k points, one pair, intensity features. Its official download link also redirected to Microsoft sign-in. Its actual file sizes and payload label schema could not be verified, so it was not downloaded or used. No cross-pair or cross-scene result is claimed.

The current official `scripts/prepare_gt_labels.py` assigns unchanged=0 and changed=1 for both input epochs and writes `cd_type`. The local files use `params`, although that current helper writes `vertex`; readers must inspect the actual element name. `evaluation/compute_metrics.py`, `classical_metrics`, confirms the ordinary per-original-point binary TP/FP/FN/TN formulas. The repository also offers a geometric nearest-neighbour dual-direction metric; we do not substitute that for indexed evaluation. This project's pooled unchanged/removed/added metric is an explicitly directional extension: changed Scan 0 is removed, changed Scan 1 is added. It is not a claim of reproducing the upstream benchmark protocol or published scores.

Sources checked:

- https://github.com/3DOM-FBK/multichange3d/blob/main/scripts/prepare_gt_labels.py
- https://github.com/3DOM-FBK/multichange3d/blob/main/evaluation/compute_metrics.py

The project page states data license CC BY-NC-SA 4.0 and code license MIT. Cite: Z. Wang, P. Trybala, A. Wieser and F. Remondino, “MultiChange3D: A Multi-Scene, Multi-Sensor Dataset for Benchmarking 3D Geometric Change Detection,” ISPRS Annals XI-2-2026, pp. 639-646, 2026, doi:10.5194/isprs-annals-XI-2-2026-639-2026. Preserve attribution and noncommercial/share-alike terms for redistributed dataset derivatives. The submission includes bounded result artifacts, not original full clouds.

## ConSLAM

Local assets under `dataset/Sequence 3`: extracted LiDAR PCDs, matching normalized 4x4 poses, their two archives, and a 1,986,430,042-byte TLS PLY. No ROS bags or RGB/NIR archives were downloaded. The TLS is inspected by memory mapping and bounded sampling, not loaded wholesale.

Official article record: https://www.repository.cam.ac.uk/items/693c260b-e145-4f01-ad88-d2031e4c0590

Official code/download entry: https://github.com/mac137/ConSLAM

The Cambridge article record says all rights reserved. The repository's `LICENCE.txt` states copyright University of Cambridge 2023, academic use only, with warranty disclaimer. This software notice does not establish a broader license for every dataset file; no permissive data redistribution right is inferred. Use the local data for the course's qualitative inspection and retain attribution. Do not redistribute full ConSLAM files in the submission bundle. No independently verified per-point change labels were found, so there are no ConSLAM accuracy scores and no labels derived from TLS presented as ground truth.

Cite: M. Trzeciak, K. Pluta, Y. Fathy, L. Alcalde, S. Chee, A. Bromley, I. Brilakis and P. Alliez, “ConSLAM: Construction Data Set for SLAM,” Journal of Computing in Civil Engineering, 2023, doi:10.1061/JCCEE5.CPENG-5212. The local Sequence 3 frames belong to one acquisition sequence; their differences cannot demonstrate monthly construction progress.

## Learning reference feasibility

The official MultiChange3D repository links Siamese KPConv, EF-Siamese KPConv and PGN3DCD. Their public frameworks build on Torch-Point3D/PyTorch Geometric and additional point kernels. `torch_points3d`, `torch_geometric` and `torch_points_kernels` are absent in the existing dl_env. We chose the authorized compact PyTorch Siamese PointNet-style course baseline to avoid a new dependency stack. This is an engineering feasibility decision, not a failed reproduction or benchmark-equivalent implementation. No pretrained weights were downloaded.

- https://github.com/IdeGelis/torch-points3d-SiameseKPConv
- https://github.com/zhanwenxiao/PGN3DCD
- PointNet reference: C. R. Qi et al., CVPR 2017, https://arxiv.org/abs/1612.00593

## Interpretation

Geometric discrepancies localize candidate changes. Visibility loss, moving objects, noise and registration error can produce the same signal. These outputs support inspection for progress monitoring; they do not certify structural completion, installed quantities or schedule status.
