# Dataset downloads and layout

Updated 26 October 2026. Original datasets are excluded from GitHub. SharePoint/provider sign-in may be required; use the official access flow.

## Required: Bike Parking Construction

Start at [MultiChange3D](https://3dom-fbk.github.io/multichange3d/), choose **MLS -> Bike Parking Construction**, and download all six original pair ZIPs.

```text
dataset/
  MultiChange3D/
    BikeParkingConstruction/
      pair_01.zip
      pair_02.zip
      pair_03.zip
      pair_12.zip
      pair_13.zip
      pair_23.zip
      pair_01/
        pointCloud0.ply
        pointCloud1.ply
      pair_02/
        pointCloud0.ply
        pointCloud1.ply
      ...
      pair_23/
        pointCloud0.ply
        pointCloud1.ply
```

An extra extracted directory such as `pair_02/pair_02/pointCloud0.ply` is accepted by recursive discovery, provided each pair contains exactly one PLY per scan. **Keep original ZIPs**: the original audit checks archive CRC and extracted-file integrity.

The locally verified PLY element is `params`, with `x,y,z,intensity,cd_type`. Labels: 0 unchanged, 1 changed. Changed Scan 0 points are candidate removals; changed Scan 1 points are candidate additions. These are geometric labels, not material/work-item annotations. Pair names are epoch indices, not dates.

These construction pair files contain intensity but no RGB, acquisition poses, ray origins, timestamps, or calibration metadata. Six pairs share four acquisition epochs. Full clouds provide geometric support; training and evaluation use sampled original cores.

## Optional: user-supplied Open_space

Use `dataset/MultiChange3D/Open_space/`, with exactly one `pointCloud0.ply` and `pointCloud1.ply` discoverable recursively. The completed diagnostic used frozen Bike settings and 5,000 uniform original cores per scan.

**Provenance caveat:** the original plan named MLS Open Space, catalog-listed at approximately 200k points. Actual local files contain **4,245,825 and 3,111,491 points** with XYZ/intensity/change labels. See [recorded filenames and hashes](../results/learning_v2_coverage_2026-10-25_v1/external_open_space/data_audit.json). Confirm the provider archive before claiming exact catalog-scene reproduction; do not silently substitute a similarly named scene.

This already-evaluated user-supplied scene is a diagnostic, not another construction-site benchmark or a new untouched test.

## Optional: RGBD Office

Local Office pairs are under `dataset/MultiChange3D/RGBD_Office/`. They are unused by the retained construction workflow and unnecessary for primary reproduction or unit tests.

## Optional: ConSLAM Sequence 3

Download entry: [official ConSLAM repository](https://github.com/mac137/ConSLAM). The existing qualitative script needs only Sequence 3 LiDAR PCDs, matching normalized 4x4 poses, and its TLS reference:

```text
dataset/
  Sequence 3/
    Copy of lidar-pcd/
      lidar-pcd/
        <timestamp>.pcd
    Copy of lidar_pose_normalized/
      normalized/
        <timestamp>.txt
    Copy of groundtruth_scan.ply
```

Pose TXT filenames must match PCD timestamps. The script uses two sweeps about 4.84 seconds apart and a bounded TLS sample. This is one acquisition sequence, not a cross-date construction-progress test. No verified change labels or quantitative change F1 are available for these inputs.

ROS bags, RGB/NIR archives, every other sequence, and calibration downloads are unnecessary for this particular Python inspection script. The provider acquisition software is a separate stack.

## Placement and integrity

The root-relative `dataset/` layout is most compatible. Some older loaders support `CV_PROJECT_DATA_ROOT`; newer intensity/disk-support scripts directly use root-relative paths. Never edit original points or labels. Historical absolute Windows paths are provenance. Retained completion JSONs do not imply excluded cache arrays exist; read [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Attribution

MultiChange3D: Z. Wang, P. Trybala, A. Wieser, F. Remondino, ISPRS Annals XI-2-2026, 639-646, [DOI](https://doi.org/10.5194/isprs-annals-XI-2-2026-639-2026). Its [official page](https://3dom-fbk.github.io/multichange3d/) specifies CC BY-NC-SA 4.0 for data and MIT for its own code.

ConSLAM: M. Trzeciak et al., *ConSLAM: Construction Data Set for SLAM*, Journal of Computing in Civil Engineering, 2023, [DOI](https://doi.org/10.1061/JCCEE5.CPENG-5212). Consult provider academic-use and data terms.

Retain applicable attribution/noncommercial/share-alike terms for derived arrays, sampled coordinates, labels and figures. See [third-party notices](../THIRD_PARTY_NOTICES.md).
