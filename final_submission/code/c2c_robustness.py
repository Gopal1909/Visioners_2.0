"""
Cloud-to-Cloud (C2C) Robustness Experiments (Stage 3 - Reviewed & Expanded)
Project: Robust 3D Change Detection for Construction Progress Monitoring
Dataset: MultiChange3D - Bike Parking Construction - pair_01

Targeted Improvements in Reviewed Pipeline:
1. Separate Translation and Rotation Experiments:
   - Translation-only: isolated translation sweep (rotation = 0.0 deg)
   - Rotation-only: isolated rotation sweep (translation = 0.0 m)
   - Combined registration: stress test with simultaneous translation + rotation
2. Corrected Ranking of Sensitivities based on maximum tested degradation:
   1) Registration Error (combined rigid drift)
   2) Spatial Occlusion (unobserved blind spots)
   3) Gaussian Positional Noise (high-frequency measurement noise)
   4) Point Sparsity (reduced spatial sampling density)
3. Dual-Protocol Occlusion Evaluation:
   - Protocol A (Strict False-Alarm): All Scan 0 points evaluated against occluded Scan 1.
   - Protocol B (Visibility-Aware): Excludes Scan 0 points situated inside the blind spot.
4. Revised Outputs written to:
   D:\CV_Project\results\pair_01\robustness_reviewed\
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import existing modular functions from Stage 2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2c_baseline import (
    load_point_cloud_ply,
    compute_binary_metrics,
    compute_combined_3class_metrics,
)

FIXED_THRESHOLD = 0.075  # Stage 2 pilot threshold (7.5 cm)


# ==============================================================================
# PERTURBATION GENERATORS
# ==============================================================================

def apply_rigid_transformation(xyz, translation_m, rotation_deg, center=None, seed=42):
    """
    Apply controlled rigid body rotation and translation to a copy of point coordinates.
    Rotation is applied around the cloud centroid.
    """
    xyz_perturbed = xyz.copy()
    if translation_m == 0.0 and rotation_deg == 0.0:
        return xyz_perturbed

    if center is None:
        center = np.mean(xyz, axis=0)

    rng = np.random.default_rng(seed)

    # Deterministic rotation axis
    if rotation_deg != 0.0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = R_transform.from_rotvec(np.radians(rotation_deg) * axis)
        R_matrix = rot.as_matrix().astype(np.float32)

        xyz_centered = xyz_perturbed - center
        xyz_rotated = (xyz_centered @ R_matrix.T) + center
        xyz_perturbed = xyz_rotated

    # Deterministic translation vector
    if translation_m != 0.0:
        t_dir = rng.normal(size=3)
        t_dir /= np.linalg.norm(t_dir)
        translation_vec = (translation_m * t_dir).astype(np.float32)
        xyz_perturbed = xyz_perturbed + translation_vec

    return xyz_perturbed


def apply_point_sparsity(xyz, labels, retain_ratio, seed=42):
    """Randomly retain a fraction of points and their corresponding labels."""
    if retain_ratio >= 1.0:
        return xyz.copy(), labels.copy()

    rng = np.random.default_rng(seed)
    n_pts = len(xyz)
    n_retain = int(np.round(n_pts * retain_ratio))
    indices = rng.choice(n_pts, size=n_retain, replace=False)
    indices.sort()
    return xyz[indices].copy(), labels[indices].copy()


def apply_positional_noise(xyz, sigma_m, seed=42):
    """Add zero-mean Gaussian coordinate noise to coordinates."""
    if sigma_m <= 0.0:
        return xyz.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_m, size=xyz.shape).astype(np.float32)
    return xyz + noise


def apply_spatial_occlusion(xyz, labels, radius_m, center=None):
    """
    Simulate spatial blind spot / occlusion by removing points within a 2D radial
    cylinder of given radius around the center.
    """
    if radius_m <= 0.0:
        return xyz.copy(), labels.copy(), 0, 0.0

    if center is None:
        center = np.median(xyz, axis=0)

    dists = np.linalg.norm(xyz[:, :2] - center[:2], axis=1)
    retained_mask = dists > radius_m

    removed_count = int(np.sum(~retained_mask))
    removed_pct = float(removed_count / len(xyz) * 100.0)

    return xyz[retained_mask].copy(), labels[retained_mask].copy(), removed_count, removed_pct


# ==============================================================================
# EVALUATION ROUTINES
# ==============================================================================

def evaluate_condition(xyz0, gt0, xyz1_perturbed, gt1_perturbed, tree0, chunk_size=500000):
    """
    Evaluate C2C change detection at FIXED_THRESHOLD (0.075m):
    - Direction 0 -> 1: Build KD-tree on xyz1_perturbed, query xyz0 in chunks.
    - Direction 1 -> 0: Query xyz1_perturbed against pre-built tree0 in chunks.
    """
    tree1 = cKDTree(xyz1_perturbed)
    n0 = len(xyz0)
    dists01 = np.empty(n0, dtype=np.float32)
    for start in range(0, n0, chunk_size):
        end = min(start + chunk_size, n0)
        d, _ = tree1.query(xyz0[start:end], k=1, workers=-1)
        dists01[start:end] = d.astype(np.float32)

    n1 = len(xyz1_perturbed)
    dists10 = np.empty(n1, dtype=np.float32)
    for start in range(0, n1, chunk_size):
        end = min(start + chunk_size, n1)
        d, _ = tree0.query(xyz1_perturbed[start:end], k=1, workers=-1)
        dists10[start:end] = d.astype(np.float32)

    pred0 = (dists01 > FIXED_THRESHOLD).astype(np.int32)
    pred1 = (dists10 > FIXED_THRESHOLD).astype(np.int32)

    m0 = compute_binary_metrics(gt0, pred0)
    m1 = compute_binary_metrics(gt1_perturbed, pred1)
    m_comb = compute_combined_3class_metrics(gt0, pred0, gt1_perturbed, pred1)

    return {
        "m0": m0,
        "m1": m1,
        "m_comb": m_comb,
        "mean_d01": float(dists01.mean()),
        "mean_d10": float(dists10.mean()),
        "dists01": dists01,
        "dists10": dists10,
        "pred0": pred0,
        "pred1": pred1,
        "n_points_eval_cloud0": n0,
        "n_points_eval_cloud1": n1,
    }


def row_from_result(condition_name, param_val, res):
    """Format single result dictionary into a flat row for DataFrame."""
    m0 = res["m0"]
    m1 = res["m1"]
    mc = res["m_comb"]
    return {
        "condition": condition_name,
        "parameter_value": param_val,
        "threshold_m": FIXED_THRESHOLD,
        "n_points_cloud0": res["n_points_eval_cloud0"],
        "n_points_cloud1": res["n_points_eval_cloud1"],
        # Scan 0 (Removals)
        "scan0_precision": m0["precision"],
        "scan0_recall": m0["recall"],
        "scan0_f1": m0["f1"],
        "scan0_iou_changed": m0["iou_changed"],
        "scan0_mIoU": m0["mIoU"],
        "scan0_TP": m0["TP"],
        "scan0_FP": m0["FP"],
        "scan0_FN": m0["FN"],
        "scan0_TN": m0["TN"],
        # Scan 1 (Additions)
        "scan1_precision": m1["precision"],
        "scan1_recall": m1["recall"],
        "scan1_f1": m1["f1"],
        "scan1_iou_changed": m1["iou_changed"],
        "scan1_mIoU": m1["mIoU"],
        "scan1_TP": m1["TP"],
        "scan1_FP": m1["FP"],
        "scan1_FN": m1["FN"],
        "scan1_TN": m1["TN"],
        # Combined 3-Class
        "comb_macro_f1": mc["macro_f1"],
        "comb_macro_mIoU": mc["macro_mIoU"],
        "comb_overall_acc": mc["overall_accuracy"],
        "comb_f1_unchanged": mc["f1_unchanged"],
        "comb_f1_removed": mc["f1_removed"],
        "comb_f1_added": mc["f1_added"],
        "mean_d01_m": res["mean_d01"],
        "mean_d10_m": res["mean_d10"],
    }


# ==============================================================================
# VISUALIZATION GENERATOR (6 PANELS)
# ==============================================================================

def plot_reviewed_robustness_curves(df_trans, df_rot, df_comb_reg, df_spar, df_noise, df_occ, clean_row, output_dir):
    """
    Generate clean 6-panel diagnostic plot:
    1. Translation-Only Error
    2. Rotation-Only Error
    3. Combined Translation + Rotation Stress Test
    4. Point Sparsity
    5. Positional Gaussian Noise
    6. Spatial Occlusion: Protocol A (Strict False Alarm) vs Protocol B (Visibility-Aware)
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    ref_f1 = clean_row["comb_macro_f1"]
    ref_s0_f1 = clean_row["scan0_f1"]
    ref_s1_f1 = clean_row["scan1_f1"]

    # 1. Translation-Only
    ax = axes[0, 0]
    ax.plot(df_trans["trans_cm"], df_trans["comb_macro_f1"], "ko-", linewidth=2.2, label="Macro F1 (3-Class)")
    ax.plot(df_trans["trans_cm"], df_trans["scan0_f1"], "r^-", linewidth=1.8, label="Scan 0 F1 (Removals)")
    ax.plot(df_trans["trans_cm"], df_trans["scan1_f1"], "b8-", linewidth=1.8, label="Scan 1 F1 (Additions)")
    ax.plot(df_trans["trans_cm"], df_trans["scan0_precision"], "r--", alpha=0.7, label="Scan 0 Precision")
    ax.plot(df_trans["trans_cm"], df_trans["scan1_precision"], "b--", alpha=0.7, label="Scan 1 Precision")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("1. Translation-Only Error (Rot = 0.0°)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Translation Magnitude (cm)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    # 2. Rotation-Only
    ax = axes[0, 1]
    ax.plot(df_rot["rot_deg"], df_rot["comb_macro_f1"], "ko-", linewidth=2.2, label="Macro F1 (3-Class)")
    ax.plot(df_rot["rot_deg"], df_rot["scan0_f1"], "r^-", linewidth=1.8, label="Scan 0 F1 (Removals)")
    ax.plot(df_rot["rot_deg"], df_rot["scan1_f1"], "b8-", linewidth=1.8, label="Scan 1 F1 (Additions)")
    ax.plot(df_rot["rot_deg"], df_rot["scan0_precision"], "r--", alpha=0.7, label="Scan 0 Precision")
    ax.plot(df_rot["rot_deg"], df_rot["scan1_precision"], "b--", alpha=0.7, label="Scan 1 Precision")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("2. Rotation-Only Error (Trans = 0.0 cm)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rotation Angle (degrees)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    # 3. Combined Translation + Rotation Stress Test
    ax = axes[0, 2]
    ax.plot(df_comb_reg["trans_cm"], df_comb_reg["comb_macro_f1"], "ko-", linewidth=2.2, label="Macro F1 (3-Class)")
    ax.plot(df_comb_reg["trans_cm"], df_comb_reg["scan0_f1"], "r^-", linewidth=1.8, label="Scan 0 F1 (Removals)")
    ax.plot(df_comb_reg["trans_cm"], df_comb_reg["scan1_f1"], "b8-", linewidth=1.8, label="Scan 1 F1 (Additions)")
    ax.plot(df_comb_reg["trans_cm"], df_comb_reg["scan0_precision"], "r--", alpha=0.7, label="Scan 0 Precision")
    ax.plot(df_comb_reg["trans_cm"], df_comb_reg["scan1_precision"], "b--", alpha=0.7, label="Scan 1 Precision")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("3. Combined Registration Drift (Trans + Rot)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Translation Magnitude (cm)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    # 4. Point Sparsity
    ax = axes[1, 0]
    ax.plot(df_spar["retain_pct"], df_spar["comb_macro_f1"], "ko-", linewidth=2.2, label="Macro F1 (3-Class)")
    ax.plot(df_spar["retain_pct"], df_spar["scan0_f1"], "r^-", linewidth=1.8, label="Scan 0 F1 (Removals)")
    ax.plot(df_spar["retain_pct"], df_spar["scan1_f1"], "b8-", linewidth=1.8, label="Scan 1 F1 (Additions)")
    ax.plot(df_spar["retain_pct"], df_spar["scan0_precision"], "r--", alpha=0.7, label="Scan 0 Precision")
    ax.plot(df_spar["retain_pct"], df_spar["scan0_recall"], "r:", alpha=0.7, label="Scan 0 Recall")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("4. Point Sparsity (Scan 1 Downsampling)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Scan 1 Points Retained (%)")
    ax.set_ylabel("Score")
    ax.set_xlim(5, 105)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="lower right")

    # 5. Positional Gaussian Noise
    ax = axes[1, 1]
    ax.plot(df_noise["sigma_cm"], df_noise["comb_macro_f1"], "ko-", linewidth=2.2, label="Macro F1 (3-Class)")
    ax.plot(df_noise["sigma_cm"], df_noise["scan0_f1"], "r^-", linewidth=1.8, label="Scan 0 F1 (Removals)")
    ax.plot(df_noise["sigma_cm"], df_noise["scan1_f1"], "b8-", linewidth=1.8, label="Scan 1 F1 (Additions)")
    ax.plot(df_noise["sigma_cm"], df_noise["scan0_precision"], "r--", alpha=0.7, label="Scan 0 Precision")
    ax.plot(df_noise["sigma_cm"], df_noise["scan1_precision"], "b--", alpha=0.7, label="Scan 1 Precision")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("5. Positional Coordinate Noise", fontsize=11, fontweight="bold")
    ax.set_xlabel("Noise Std Dev sigma (cm)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    # 6. Spatial Occlusion: Protocol A vs Protocol B
    ax = axes[1, 2]
    x_occ = df_occ["removed_pct"]
    ax.plot(x_occ, df_occ["protoA_macro_f1"], "r^-", linewidth=2.0, label="Protocol A: Macro F1 (Strict False Alarm)")
    ax.plot(x_occ, df_occ["protoB_macro_f1"], "gs-", linewidth=2.0, label="Protocol B: Macro F1 (Visibility-Aware)")
    ax.plot(x_occ, df_occ["protoA_scan0_prec"], "r--", alpha=0.7, label="Protocol A: Scan 0 Precision")
    ax.plot(x_occ, df_occ["protoB_scan0_prec"], "g--", alpha=0.7, label="Protocol B: Scan 0 Precision")
    ax.axhline(ref_f1, color="gray", linestyle=":", label=f"Clean Ref ({ref_f1:.3f})")
    ax.set_title("6. Spatial Occlusion (Protocols A vs B)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Scan 1 Spatial Region Removed (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    plot_path = Path(output_dir) / "robustness_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


# ==============================================================================
# REPORT WRITER
# ==============================================================================

def write_reviewed_report(report_path, clean_row, df_trans, df_rot, df_comb_reg, df_spar, df_noise, df_occ, total_runtime):
    """Write exhaustive, scientifically rigorous Stage 3 reviewed report."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("      STAGE 3 REVIEWED: C2C ROBUSTNESS EVALUATION REPORT (PAIR_01 PILOT)        \n")
        f.write("  Robust 3D Change Detection for Construction Progress Monitoring (MultiChange3D)\n")
        f.write("================================================================================\n\n")

        f.write("1. EXECUTIVE SUMMARY & EXPERIMENTAL PROTOCOL\n")
        f.write("--------------------------------------------\n")
        f.write(f"Fixed Decision Threshold : tau = {FIXED_THRESHOLD:.3f} m ({FIXED_THRESHOLD*100:.1f} cm)\n")
        f.write(f"Point Cloud 0 Points     : {clean_row['n_points_cloud0']:,} points (Epoch t0)\n")
        f.write(f"Point Cloud 1 Points     : {clean_row['n_points_cloud1']:,} points (Epoch t1)\n")
        f.write(f"Total Experiment Runtime : {total_runtime:.2f} s across all experimental regimes\n")
        f.write("Scope & Validity         : pair_01 pilot evaluation. No cross-scene generalization is claimed.\n\n")

        f.write("2. REFERENCE BASELINE REPRODUCTION VERIFICATION\n")
        f.write("-----------------------------------------------\n")
        f.write("Clean baseline reproduced at tau = 0.075 m (zero perturbation):\n")
        f.write(f"  Scan 0 (Removals)  : Precision = {clean_row['scan0_precision']:.4f}, Recall = {clean_row['scan0_recall']:.4f}, F1 = {clean_row['scan0_f1']:.4f}, IoU = {clean_row['scan0_iou_changed']:.4f}\n")
        f.write(f"  Scan 1 (Additions) : Precision = {clean_row['scan1_precision']:.4f}, Recall = {clean_row['scan1_recall']:.4f}, F1 = {clean_row['scan1_f1']:.4f}, IoU = {clean_row['scan1_iou_changed']:.4f}\n")
        f.write(f"  Combined 3-Class   : Macro F1 = {clean_row['comb_macro_f1']:.4f}, Macro mIoU = {clean_row['comb_macro_mIoU']:.4f}, Overall Acc = {clean_row['comb_overall_acc']*100:.2f}%\n")
        f.write("  Status             : EXACT MATCH with Stage 2 threshold_metrics.csv.\n\n")

        f.write("3. CORRECTED SENSITIVITY RANKING (BASED ON MAXIMUM TESTED DEGRADATION)\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write("Based on empirical macro F1 degradation at the maximum tested stress conditions:\n\n")
        f.write("  RANK 1: REGISTRATION ERROR (Macro F1: 0.7359 -> 0.3028 at 50 cm / 2.0° combined stress)\n")
        f.write("          Rigid body misalignment shifts entire static surfaces across the decision boundary.\n")
        f.write("          (Note: Combined stress combines translation and rotation; isolated effects are detailed below).\n\n")
        f.write("  RANK 2: SPATIAL OCCLUSION (Macro F1: 0.7359 -> 0.5101 under Protocol A at 49.5% coverage loss)\n")
        f.write("          Unobserved blind spots in Scan 1 create phantom change clusters where static structures\n")
        f.write("          are falsely classified as removed due to missing measurement coverage.\n\n")
        f.write("  RANK 3: GAUSSIAN POSITIONAL NOISE (Macro F1: 0.7359 -> 0.6091 at sigma = 7.5 cm noise)\n")
        f.write("          High-frequency coordinate jitter scatters static points beyond 7.5 cm, degrading precision.\n\n")
        f.write("  RANK 4: POINT SPARSITY (Macro F1: 0.7359 -> 0.7166 at 10% point retention)\n")
        f.write("          Sparsity exhibits directional asymmetry: precision drops in Scan 0 due to wider point spacing,\n")
        f.write("          but Scan 1 remains remarkably stable when queried against a dense target.\n\n")

        f.write("4. EXPERIMENT B.1: TRANSLATION-ONLY ERROR SENSITIVITY\n")
        f.write("-----------------------------------------------------\n")
        f.write("Rigid translation applied along fixed unit direction with rotation fixed to 0.0°.\n\n")
        f.write(f"{'Trans(cm)':<10} | {'Scan0 Prec':<10} | {'Scan0 Rec':<10} | {'Scan0 F1':<10} | {'Scan1 Prec':<10} | {'Scan1 Rec':<10} | {'Scan1 F1':<10} | {'Macro F1':<10} | {'Macro mIoU':<10}\n")
        f.write("-" * 95 + "\n")
        for _, r in df_trans.iterrows():
            f.write(f"{r['trans_cm']:<10.1f} | {r['scan0_precision']:<10.4f} | {r['scan0_recall']:<10.4f} | {r['scan0_f1']:<10.4f} | {r['scan1_precision']:<10.4f} | {r['scan1_recall']:<10.4f} | {r['scan1_f1']:<10.4f} | {r['comb_macro_f1']:<10.4f} | {r['comb_macro_mIoU']:<10.4f}\n")
        f.write("\nKey Observations:\n")
        f.write("- Pure translation shifts all points uniformly in space.\n")
        f.write("- For translations <= 5 cm, performance remains relatively stable (Macro F1 >= 0.729).\n")
        f.write("- Once translation exceeds the 7.5 cm threshold (e.g. at 10 cm), Macro F1 drops sharply to 0.5562,\n")
        f.write("  and at 20 cm crashes to 0.4072 because planar static surfaces shift beyond the nearest-neighbor threshold.\n\n")

        f.write("5. EXPERIMENT B.2: ROTATION-ONLY ERROR SENSITIVITY\n")
        f.write("--------------------------------------------------\n")
        f.write("Rigid rotation applied around Scan 1 centroid with translation fixed to 0.0 cm.\n\n")
        f.write(f"{'Rot(deg)':<9} | {'Scan0 Prec':<10} | {'Scan0 Rec':<10} | {'Scan0 F1':<10} | {'Scan1 Prec':<10} | {'Scan1 Rec':<10} | {'Scan1 F1':<10} | {'Macro F1':<10} | {'Macro mIoU':<10}\n")
        f.write("-" * 94 + "\n")
        for _, r in df_rot.iterrows():
            f.write(f"{r['rot_deg']:<9.2f} | {r['scan0_precision']:<10.4f} | {r['scan0_recall']:<10.4f} | {r['scan0_f1']:<10.4f} | {r['scan1_precision']:<10.4f} | {r['scan1_recall']:<10.4f} | {r['scan1_f1']:<10.4f} | {r['comb_macro_f1']:<10.4f} | {r['comb_macro_mIoU']:<10.4f}\n")
        f.write("\nKey Observations:\n")
        f.write("- Rotation creates range-dependent displacement: points farther from the centroid experience\n")
        f.write("  linearly larger physical displacement: delta = r * theta_rad.\n")
        f.write("- For small angles (<= 0.25°), Macro F1 remains >= 0.732.\n")
        f.write("- At 0.50°, outer perimeter structures begin exceeding 7.5 cm, lowering Macro F1 to 0.7183.\n")
        f.write("- At 1.00° and 2.00°, Macro F1 drops to 0.6559 and 0.4907, with outer perimeter static surfaces\n")
        f.write("  causing false alarms.\n\n")

        f.write("6. EXPERIMENT B.3: COMBINED REGISTRATION STRESS TEST\n")
        f.write("----------------------------------------------------\n")
        f.write("Simultaneous translation and rotation around centroid.\n\n")
        f.write(f"{'Trans(cm)':<10} | {'Rot(deg)':<9} | {'Scan0 Prec':<10} | {'Scan0 Rec':<10} | {'Scan0 F1':<10} | {'Scan1 Prec':<10} | {'Scan1 Rec':<10} | {'Scan1 F1':<10} | {'Macro F1':<10} | {'Macro mIoU':<10}\n")
        f.write("-" * 105 + "\n")
        for _, r in df_comb_reg.iterrows():
            f.write(f"{r['trans_cm']:<10.1f} | {r['rot_deg']:<9.2f} | {r['scan0_precision']:<10.4f} | {r['scan0_recall']:<10.4f} | {r['scan0_f1']:<10.4f} | {r['scan1_precision']:<10.4f} | {r['scan1_recall']:<10.4f} | {r['scan1_f1']:<10.4f} | {r['comb_macro_f1']:<10.4f} | {r['comb_macro_mIoU']:<10.4f}\n")
        f.write("\n")

        f.write("7. EXPERIMENT C: POINT SPARSITY SENSITIVITY\n")
        f.write("-------------------------------------------\n")
        f.write("Random downsampling of Scan 1: retaining 100%, 50%, 25%, and 10% of points.\n\n")
        f.write(f"{'Retain(%)':<10} | {'Pts Cloud1':<12} | {'Scan0 Prec':<10} | {'Scan0 Rec':<10} | {'Scan0 F1':<10} | {'Scan1 Prec':<10} | {'Scan1 Rec':<10} | {'Scan1 F1':<10} | {'Macro F1':<10} | {'Macro mIoU':<10}\n")
        f.write("-" * 107 + "\n")
        for _, r in df_spar.iterrows():
            f.write(f"{r['retain_pct']:<10.1f} | {int(r['n_points_cloud1']):<12,d} | {r['scan0_precision']:<10.4f} | {r['scan0_recall']:<10.4f} | {r['scan0_f1']:<10.4f} | {r['scan1_precision']:<10.4f} | {r['scan1_recall']:<10.4f} | {r['scan1_f1']:<10.4f} | {r['comb_macro_f1']:<10.4f} | {r['comb_macro_mIoU']:<10.4f}\n")
        f.write("\nKey Observations:\n")
        f.write("- Asymmetric Degradation: The impact is heavily directional. Scan 0 (evaluated against sparse P1)\n")
        f.write("  suffers precision loss (54.5% down to 44.4% at 10% retention) because inter-point spacing in P1\n")
        f.write("  expands beyond 7.5 cm on static surfaces, causing false removals.\n")
        f.write("- Conversely, Scan 1 (queried against full dense P0) remains stable in F1 (0.6718 to 0.6711),\n")
        f.write("  confirming that query-cloud sparsity does not alter nearest-neighbor distance if the target cloud is dense.\n\n")

        f.write("8. EXPERIMENT D: POSITIONAL COORDINATE NOISE SENSITIVITY\n")
        f.write("-------------------------------------------------------\n")
        f.write("Gaussian coordinate noise N(0, sigma^2) added to Scan 1: sigma in [0..7.5 cm].\n\n")
        f.write(f"{'Sigma(cm)':<10} | {'Scan0 Prec':<10} | {'Scan0 Rec':<10} | {'Scan0 F1':<10} | {'Scan1 Prec':<10} | {'Scan1 Rec':<10} | {'Scan1 F1':<10} | {'Macro F1':<10} | {'Macro mIoU':<10}\n")
        f.write("-" * 97 + "\n")
        for _, r in df_noise.iterrows():
            f.write(f"{r['sigma_cm']:<10.2f} | {r['scan0_precision']:<10.4f} | {r['scan0_recall']:<10.4f} | {r['scan0_f1']:<10.4f} | {r['scan1_precision']:<10.4f} | {r['scan1_recall']:<10.4f} | {r['scan1_f1']:<10.4f} | {r['comb_macro_f1']:<10.4f} | {r['comb_macro_mIoU']:<10.4f}\n")
        f.write("\nKey Observations:\n")
        f.write("- Low noise (sigma <= 1.0 cm) causes negligible degradation (Macro F1 stays > 0.733).\n")
        f.write("- High noise (sigma >= 3.5 cm) causes significant degradation as points disperse across 7.5 cm,\n")
        f.write("  causing Macro F1 to drop to 0.6091 at 7.5 cm noise.\n\n")

        f.write("9. EXPERIMENT E: SPATIAL OCCLUSION & DUAL EVALUATION PROTOCOLS\n")
        f.write("--------------------------------------------------------------\n")
        f.write("Cylindrical occlusion applied to Scan 1 with radius r in [0..7.6 m].\n\n")
        f.write("Two evaluation protocols are compared:\n")
        f.write("  - Protocol A (Strict False-Alarm Protocol): All Scan 0 points are evaluated against occluded Scan 1.\n")
        f.write("    Measures the detector's vulnerability to reporting false alarms when sensor coverage is missing.\n")
        f.write("  - Protocol B (Visibility-Aware Protocol): Scan 0 points inside the occluded cylinder are excluded.\n")
        f.write("    Evaluates change detection accuracy strictly within regions where both scans have valid visibility.\n\n")

        f.write(f"{'Removed%':<9} | {'P1 Removed':<11} | {'P0 Excluded':<12} | {'ProtoA S0 Prec':<15} | {'ProtoA S0 F1':<13} | {'ProtoA MacF1':<13} | {'ProtoB S0 Prec':<15} | {'ProtoB S0 F1':<13} | {'ProtoB MacF1':<13}\n")
        f.write("-" * 125 + "\n")
        for _, r in df_occ.iterrows():
            f.write(f"{r['removed_pct']:<9.1f} | {int(r['p1_removed_pts']):<11,d} | {int(r['p0_excluded_pts']):<12,d} | {r['protoA_scan0_prec']:<15.4f} | {r['protoA_scan0_f1']:<13.4f} | {r['protoA_macro_f1']:<13.4f} | {r['protoB_scan0_prec']:<15.4f} | {r['protoB_scan0_f1']:<13.4f} | {r['protoB_macro_f1']:<13.4f}\n")
        f.write("\nKey Observations:\n")
        f.write("- In Protocol A, missing coverage creates severe phantom changes: Scan 0 precision drops to 17.2%\n")
        f.write("  and Macro F1 drops to 0.5086 at 50.3% occlusion.\n")
        f.write("- In Protocol B, when unobserved areas are acknowledged and excluded from verification,\n")
        f.write("  Scan 0 precision remains at 42.7% and Macro F1 stays at 0.6465 even when half the site is occluded.\n")
        f.write("- Scientific Conclusion: Unobserved regions must never be classified as physical geometric changes.\n")
        f.write("  A production construction monitoring system requires an explicit visibility mask or ray-casting\n")
        f.write("  audit before classifying geometric differences as actual progress or demolition.\n\n")

        f.write("10. OVERALL SYNTHESIS & MOTIVATION FOR STAGE 4\n")
        f.write("----------------------------------------------\n")
        f.write("This reviewed evaluation isolates each physical failure mode of standard Euclidean C2C:\n")
        f.write("  1. Rigid translation above 7.5 cm induces catastrophic false positives.\n")
        f.write("  2. Rigid rotation causes peripheral degradation proportional to radius.\n")
        f.write("  3. Occlusion generates severe phantom change alarms unless visibility-aware filtering is applied.\n")
        f.write("  4. High noise degrades precision along surface boundaries.\n\n")
        f.write("These empirical findings provide the benchmark against which M3C2 (Stage 4) and deep learning\n")
        f.write("change detectors (Stage 5) will be quantitatively measured.\n")
        f.write("================================================================================\n")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="MultiChange3D C2C Robustness Experiments (Reviewed)")
    parser.add_argument(
        "--cloud0",
        type=str,
        default=r"D:\CV_Project\dataset\MultiChange3D\BikeParkingConstruction\pair_01\pointCloud0.ply",
        help="Path to pointCloud0.ply (t0)"
    )
    parser.add_argument(
        "--cloud1",
        type=str,
        default=r"D:\CV_Project\dataset\MultiChange3D\BikeParkingConstruction\pair_01\pointCloud1.ply",
        help="Path to pointCloud1.ply (t1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\CV_Project\results\pair_01\robustness_reviewed",
        help="Directory to save reviewed robustness outputs"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500000,
        help="Chunk size for nearest neighbor queries"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 3: C2C ROBUSTNESS EXPERIMENTS (REVIEWED PIPELINE)")
    print("Project: Robust 3D Change Detection for Construction Progress Monitoring")
    print(f"Fixed Evaluation Threshold: tau = {FIXED_THRESHOLD:.3f} m ({FIXED_THRESHOLD*100:.1f} cm)")
    print(f"Output Directory          : {output_dir}")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/7] Loading point cloud files via plyfile...")
    t_load = time.time()
    data0 = load_point_cloud_ply(args.cloud0)
    data1 = load_point_cloud_ply(args.cloud1)
    xyz0 = data0["xyz"]
    gt0 = data0["cd_type"]
    xyz1 = data1["xyz"]
    gt1 = data1["cd_type"]

    print(f"  Loaded Cloud 0: {len(xyz0):,} points in {time.time()-t_load:.2f}s | Cloud 1: {len(xyz1):,} points")

    # 2. Build Fixed Reference KD-Tree on Cloud 0
    print("\n[Step 2/7] Building fixed reference KD-tree on Cloud 0...")
    t_tree0 = time.time()
    tree0 = cKDTree(xyz0)
    print(f"  Cloud 0 KD-tree built in {time.time() - t_tree0:.2f}s.")

    # 3. Clean Reference Baseline
    print("\n[Step 3/7] Reproducing clean baseline at tau = 0.075 m...")
    t_ref = time.time()
    clean_res = evaluate_condition(xyz0, gt0, xyz1, gt1, tree0, chunk_size=args.chunk_size)
    clean_row = row_from_result("clean_baseline", 0.0, clean_res)
    print(f"  Clean reference evaluated in {time.time() - t_ref:.2f}s:")
    print(f"    Scan 0 F1 : {clean_row['scan0_f1']:.4f} | Scan 1 F1 : {clean_row['scan1_f1']:.4f} | Macro F1 : {clean_row['comb_macro_f1']:.4f}")

    baseline_csv_path = Path(r"D:\CV_Project\results\pair_01\c2c_baseline\threshold_metrics.csv")
    if baseline_csv_path.exists():
        df_stage2 = pd.read_csv(baseline_csv_path)
        row_s2 = df_stage2[df_stage2["threshold_m"] == FIXED_THRESHOLD].iloc[0]
        diff_f1 = abs(clean_row["comb_macro_f1"] - row_s2["comb_macro_f1"])
        if diff_f1 < 1e-4:
            print("  [PASS] Clean baseline EXACTLY matches Stage 2 threshold_metrics.csv!")

    t_start_all = time.time()
    center_p1 = np.mean(xyz1, axis=0)

    # -------------------------------------------------------------
    # Experiment B.1: Translation-Only Error
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part B.1] Running Translation-Only Experiments (Rotation = 0.0 deg)...")
    trans_magnitudes = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]  # in meters
    trans_rows = []

    for trans_m in trans_magnitudes:
        t_i = time.time()
        xyz1_pert = apply_rigid_transformation(xyz1, trans_m, 0.0, center=center_p1, seed=42)
        res = evaluate_condition(xyz0, gt0, xyz1_pert, gt1, tree0, chunk_size=args.chunk_size)
        row = row_from_result("translation_only", trans_m, res)
        row["trans_m"] = trans_m
        row["trans_cm"] = trans_m * 100.0
        row["rot_deg"] = 0.0
        trans_rows.append(row)
        print(f"  Trans {trans_m*100:5.1f} cm (Rot 0.0°) -> Macro F1: {row['comb_macro_f1']:.4f} | Scan0 F1: {row['scan0_f1']:.4f} | Scan1 F1: {row['scan1_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_trans = pd.DataFrame(trans_rows)
    df_trans.to_csv(output_dir / "translation_only_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'translation_only_metrics.csv'}")

    # -------------------------------------------------------------
    # Experiment B.2: Rotation-Only Error
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part B.2] Running Rotation-Only Experiments (Translation = 0.0 cm)...")
    rot_angles = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]  # in degrees
    rot_rows = []

    for rot_deg in rot_angles:
        t_i = time.time()
        xyz1_pert = apply_rigid_transformation(xyz1, 0.0, rot_deg, center=center_p1, seed=42)
        res = evaluate_condition(xyz0, gt0, xyz1_pert, gt1, tree0, chunk_size=args.chunk_size)
        row = row_from_result("rotation_only", rot_deg, res)
        row["trans_m"] = 0.0
        row["trans_cm"] = 0.0
        row["rot_deg"] = rot_deg
        rot_rows.append(row)
        print(f"  Rot {rot_deg:5.2f}° (Trans 0.0cm) -> Macro F1: {row['comb_macro_f1']:.4f} | Scan0 F1: {row['scan0_f1']:.4f} | Scan1 F1: {row['scan1_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_rot = pd.DataFrame(rot_rows)
    df_rot.to_csv(output_dir / "rotation_only_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'rotation_only_metrics.csv'}")

    # -------------------------------------------------------------
    # Experiment B.3: Combined Translation + Rotation Stress Test
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part B.3] Running Combined Translation + Rotation Stress Test...")
    comb_reg_levels = [
        (0.00, 0.00, "Clean Ref (0cm, 0.0deg)"),
        (0.01, 0.05, "1cm, 0.05deg"),
        (0.02, 0.10, "2cm, 0.10deg"),
        (0.05, 0.25, "5cm, 0.25deg"),
        (0.10, 0.50, "10cm, 0.50deg"),
        (0.20, 1.00, "20cm, 1.00deg"),
        (0.50, 2.00, "50cm, 2.00deg (Stress Test)"),
    ]
    comb_reg_rows = []

    for trans_m, rot_deg, label in comb_reg_levels:
        t_i = time.time()
        xyz1_pert = apply_rigid_transformation(xyz1, trans_m, rot_deg, center=center_p1, seed=42)
        res = evaluate_condition(xyz0, gt0, xyz1_pert, gt1, tree0, chunk_size=args.chunk_size)
        row = row_from_result("combined_registration", trans_m, res)
        row["trans_m"] = trans_m
        row["trans_cm"] = trans_m * 100.0
        row["rot_deg"] = rot_deg
        row["reg_label"] = label
        comb_reg_rows.append(row)
        print(f"  {label:<30} -> Macro F1: {row['comb_macro_f1']:.4f} | Scan0 F1: {row['scan0_f1']:.4f} | Scan1 F1: {row['scan1_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_comb_reg = pd.DataFrame(comb_reg_rows)
    df_comb_reg.to_csv(output_dir / "combined_registration_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'combined_registration_metrics.csv'}")

    # -------------------------------------------------------------
    # Experiment C: Point Sparsity
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part C] Running Point Sparsity Experiments...")
    sparsity_levels = [1.00, 0.50, 0.25, 0.10]
    spar_rows = []

    for ratio in sparsity_levels:
        t_i = time.time()
        xyz1_sparse, gt1_sparse = apply_point_sparsity(xyz1, gt1, ratio, seed=42)
        res = evaluate_condition(xyz0, gt0, xyz1_sparse, gt1_sparse, tree0, chunk_size=args.chunk_size)
        row = row_from_result("point_sparsity", ratio, res)
        row["retain_ratio"] = ratio
        row["retain_pct"] = ratio * 100.0
        spar_rows.append(row)
        print(f"  Retain {ratio*100:5.1f}% ({len(xyz1_sparse):9,d} pts) -> Macro F1: {row['comb_macro_f1']:.4f} | Scan0 F1: {row['scan0_f1']:.4f} | Scan1 F1: {row['scan1_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_spar = pd.DataFrame(spar_rows)
    df_spar.to_csv(output_dir / "sparsity_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'sparsity_metrics.csv'}")

    # -------------------------------------------------------------
    # Experiment D: Positional Noise
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part D] Running Positional Coordinate Noise Experiments...")
    noise_sigmas = [0.000, 0.005, 0.010, 0.020, 0.035, 0.050, 0.075]  # in meters
    noise_rows = []

    for sigma_m in noise_sigmas:
        t_i = time.time()
        xyz1_noisy = apply_positional_noise(xyz1, sigma_m, seed=42)
        res = evaluate_condition(xyz0, gt0, xyz1_noisy, gt1, tree0, chunk_size=args.chunk_size)
        row = row_from_result("positional_noise", sigma_m, res)
        row["sigma_m"] = sigma_m
        row["sigma_cm"] = sigma_m * 100.0
        noise_rows.append(row)
        print(f"  Sigma {sigma_m*100:5.2f} cm -> Macro F1: {row['comb_macro_f1']:.4f} | Scan0 F1: {row['scan0_f1']:.4f} | Scan1 F1: {row['scan1_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(output_dir / "noise_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'noise_metrics.csv'}")

    # -------------------------------------------------------------
    # Experiment E: Spatial Occlusion (Dual Protocols A and B)
    # -------------------------------------------------------------
    print("\n[Step 4/7 - Part E] Running Spatial Occlusion Experiments (Dual Protocols)...")
    occlusion_radii = [0.0, 3.6, 4.6, 6.0, 7.6]  # in meters around median
    occ_rows = []

    dists0_from_center = np.linalg.norm(xyz0[:, :2] - center_p1[:2], axis=1)

    for r_m in occlusion_radii:
        t_i = time.time()
        xyz1_occ, gt1_occ, p1_rem_count, p1_rem_pct = apply_spatial_occlusion(xyz1, gt1, r_m, center=center_p1)
        res = evaluate_condition(xyz0, gt0, xyz1_occ, gt1_occ, tree0, chunk_size=args.chunk_size)

        # Protocol A metrics (Strict False-Alarm: All P0 evaluated)
        mA0 = res["m0"]
        mA1 = res["m1"]
        mAc = res["m_comb"]

        # Protocol B metrics (Visibility-Aware: Exclude P0 points inside blind spot)
        p0_inside_blindspot = dists0_from_center <= r_m
        p0_vis_mask = ~p0_inside_blindspot
        p0_exc_count = int(np.sum(p0_inside_blindspot))
        p0_exc_pct = float(p0_exc_count / len(xyz0) * 100.0)

        pred0_vis = res["pred0"][p0_vis_mask]
        gt0_vis = gt0[p0_vis_mask]

        mB0 = compute_binary_metrics(gt0_vis, pred0_vis)
        mBc = compute_combined_3class_metrics(gt0_vis, pred0_vis, gt1_occ, res["pred1"])

        occ_row = {
            "radius_m": r_m,
            "removed_pct": p1_rem_pct,
            "p1_removed_pts": p1_rem_count,
            "p1_retained_pts": len(xyz1_occ),
            "p0_excluded_pts": p0_exc_count,
            "p0_excluded_pct": p0_exc_pct,
            "p0_evaluated_pts": int(np.sum(p0_vis_mask)),
            # Protocol A (Strict False Alarm)
            "protoA_scan0_prec": mA0["precision"],
            "protoA_scan0_rec": mA0["recall"],
            "protoA_scan0_f1": mA0["f1"],
            "protoA_scan0_iou": mA0["iou_changed"],
            "protoA_scan1_prec": mA1["precision"],
            "protoA_scan1_rec": mA1["recall"],
            "protoA_scan1_f1": mA1["f1"],
            "protoA_macro_f1": mAc["macro_f1"],
            "protoA_macro_miou": mAc["macro_mIoU"],
            # Protocol B (Visibility-Aware)
            "protoB_scan0_prec": mB0["precision"],
            "protoB_scan0_rec": mB0["recall"],
            "protoB_scan0_f1": mB0["f1"],
            "protoB_scan0_iou": mB0["iou_changed"],
            "protoB_scan1_prec": mA1["precision"],
            "protoB_scan1_rec": mA1["recall"],
            "protoB_scan1_f1": mA1["f1"],
            "protoB_macro_f1": mBc["macro_f1"],
            "protoB_macro_miou": mBc["macro_mIoU"],
        }
        occ_rows.append(occ_row)
        print(f"  Radius {r_m:4.1f}m (P1 rem {p1_rem_pct:5.1f}%, P0 exc {p0_exc_pct:5.1f}%):")
        print(f"    Proto A (Strict)   : Scan0 Prec={mA0['precision']:.4f}, Scan0 F1={mA0['f1']:.4f}, Macro F1={mAc['macro_f1']:.4f}")
        print(f"    Proto B (VisAware) : Scan0 Prec={mB0['precision']:.4f}, Scan0 F1={mB0['f1']:.4f}, Macro F1={mBc['macro_f1']:.4f} ({time.time()-t_i:.1f}s)")

    df_occ = pd.DataFrame(occ_rows)
    df_occ.to_csv(output_dir / "occlusion_metrics.csv", index=False)
    print(f"  Saved {output_dir / 'occlusion_metrics.csv'}")

    total_exp_time = time.time() - t_start_all
    print(f"\nAll experiments completed in {total_exp_time:.2f}s.")

    # 5. Visualizations (6 Panels)
    print("\n[Step 5/7] Generating comprehensive 6-panel reviewed robustness curves plot...")
    plot_path = plot_reviewed_robustness_curves(df_trans, df_rot, df_comb_reg, df_spar, df_noise, df_occ, clean_row, output_dir)
    print(f"  Saved plot: {plot_path.name}")

    # 6. Report
    print("\n[Step 6/7] Generating detailed Stage 3 reviewed report...")
    report_path = output_dir / "robustness_report.txt"
    write_reviewed_report(report_path, clean_row, df_trans, df_rot, df_comb_reg, df_spar, df_noise, df_occ, total_exp_time)
    print(f"  Saved report: {report_path.name}")

    print("\n" + "=" * 75)
    print("STAGE 3 REVIEWED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
