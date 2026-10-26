# Attribution and third-party terms

## Original project

CV5570: *Robust 3D Change Detection for Construction Progress Monitoring*. Original documents credit Vishwanath Venugopal (CE23B124) and Shaik Koduru Asif (CE23B114). Their source/documents and historical attribution are preserved. This publication does not invent a permissive license for original project source or transfer third-party rights.

## MultiChange3D and derived outputs

Zhaoyi Wang, Pawel Trybala, Andreas Wieser, Fabio Remondino, *MultiChange3D: A Multi-Scene, Multi-Sensor Dataset for Benchmarking 3D Geometric Change Detection*, ISPRS Annals XI-2-2026, 639-646 (2026), [DOI](https://doi.org/10.5194/isprs-annals-XI-2-2026-639-2026).

The [official project](https://3dom-fbk.github.io/multichange3d/) specifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for its data and MIT for its own code. Original clouds/archives are excluded. Derived sampled coordinates, ground-truth labels, predictions, figures and embedded demo samples remain subject to applicable attribution/noncommercial/share-alike terms.

Our additions include model outputs, sampled evaluation protocols, stress transformations, plots and analysis. They are not unmodified provider data or published benchmark reproductions. The upstream code license is not a blanket license for this project's original source or every included artifact. Retain original notices where upstream material is used.

## ConSLAM

Maciej Trzeciak et al., *ConSLAM: Construction Data Set for SLAM*, Journal of Computing in Civil Engineering (2023), [DOI](https://doi.org/10.1061/JCCEE5.CPENG-5212). [Provider repository and terms](https://github.com/mac137/ConSLAM).

Academic-use terms apply. Original PCDs, poses, calibration files and TLS cloud are excluded; retained qualitative figures/measurements are attributed research outputs, not change ground truth or a grant of broader rights in original data. Consult provider terms before further use.

## Software and model references

Dependencies are separately distributed under their own licenses; environments and vendored packages are excluded. Executed versions are recorded in requirements and historical environment files.

PointNet: C. R. Qi et al., CVPR 2017, [paper](https://arxiv.org/abs/1612.00593). Our compact Siamese/multiscale networks and custom hierarchy are project implementations, not pretrained-weight use or faithful KPConv/PointNet++ reproductions.
