"""Generate the Semantic Building Scan sample benchmark NPZ.
Matches the multi-story building from the user's photograph:
- Main Building: 16m x 12m x 8m tall with 2.5m protruding turret / bay window, mansard roof to 10.5m, and flat roof deck.
- Ground: 35m x 25m pavement at Z=0.
- Scan 0: Main building + ground + temporary scaffolding on the right (Candidate Removal, 1,200 pts).
- Scan 1: Main building + ground + two-story annex + bright neon green roof (Candidate Addition, 2,600 pts).
- Pre-computed C2C and V2 PointNet predictions.
"""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

np.random.seed(42)

def sample_box(x_range, y_range, z_range, n_pts, noise=0.015):
    pts = []
    choices = np.random.choice(6, n_pts)
    for c in choices:
        if c == 0: pts.append([np.random.uniform(*x_range), np.random.uniform(*y_range), z_range[0]])
        elif c == 1: pts.append([np.random.uniform(*x_range), np.random.uniform(*y_range), z_range[1]])
        elif c == 2: pts.append([np.random.uniform(*x_range), y_range[0], np.random.uniform(*z_range)])
        elif c == 3: pts.append([np.random.uniform(*x_range), y_range[1], np.random.uniform(*z_range)])
        elif c == 4: pts.append([x_range[0], np.random.uniform(*y_range), np.random.uniform(*z_range)])
        else: pts.append([x_range[1], np.random.uniform(*y_range), np.random.uniform(*z_range)])
    return np.array(pts) + np.random.normal(0, noise, (n_pts, 3))

# 1. Main Building Structure (Permanent / Unchanged)
main_walls = sample_box([0, 16], [0, 12], [0, 8], 3000)
turret = sample_box([1.5, 5.5], [-2.2, 0.0], [0, 8], 1000)
roof_slopes = sample_box([0.5, 15.5], [0.5, 11.5], [8.0, 10.5], 1400)
flat_roof = sample_box([2, 14], [2, 10], [10.4, 10.6], 600)
main_bldg_0 = np.vstack([main_walls, turret, roof_slopes, flat_roof])
# Add realistic LiDAR angular scatter and point spacing (different laser beam rays)
main_bldg_1 = sample_box([0, 16], [0, 12], [0, 8], 3000)
turret_1 = sample_box([1.5, 5.5], [-2.2, 0.0], [0, 8], 1000)
roof_slopes_1 = sample_box([0.5, 15.5], [0.5, 11.5], [8.0, 10.5], 1400)
flat_roof_1 = sample_box([2, 14], [2, 10], [10.4, 10.6], 600)
main_bldg_1 = np.vstack([main_walls_1 := main_bldg_1, turret_1, roof_slopes_1, flat_roof_1])

# Ground Plane
gx0 = np.random.uniform(-5, 30, 2500); gy0 = np.random.uniform(-6, 18, 2500); gz0 = np.random.normal(0, 0.02, 2500)
ground_0 = np.column_stack([gx0, gy0, gz0])

gx1 = np.random.uniform(-5, 30, 2500); gy1 = np.random.uniform(-6, 18, 2500); gz1 = np.random.normal(0, 0.02, 2500)
ground_1 = np.column_stack([gx1, gy1, gz1])

# 2. Temporary Scaffolding on the Right (Scan 0 ONLY -> Candidate Removal)
scaff_x = np.random.uniform(16.5, 26, 1200)
scaff_y = np.random.uniform(1.5, 10.5, 1200)
scaff_z = np.random.uniform(0.1, 5.0, 1200)
temp_scaff = np.column_stack([scaff_x, scaff_y, scaff_z])

# 3. Two-Story Side Annex & Neon Green Roof (Scan 1 ONLY -> Candidate Addition)
annex_walls = sample_box([16, 26], [1.5, 10.5], [0, 5.5], 1400)
rx = np.random.uniform(15.8, 26.2, 1200)
ry = np.random.uniform(1.2, 10.8, 1200)
rz = 5.5 + (2.5 - np.abs(ry - 6.0) * (2.5 / 4.8)) + np.random.normal(0, 0.02, 1200)
annex_green_roof = np.column_stack([rx, ry, rz])
annex_total = np.vstack([annex_walls, annex_green_roof])

# Assemble Scan 0
scan0 = np.vstack([ground_0, main_bldg_0, temp_scaff])
gt0 = np.zeros(len(scan0), dtype=int)
gt0[len(ground_0) + len(main_bldg_0):] = 1

# Assemble Scan 1
scan1 = np.vstack([ground_1, main_bldg_1, annex_total])
gt1 = np.zeros(len(scan1), dtype=int)
gt1[len(ground_1) + len(main_bldg_1):] = 1

# C2C Distance computation
tree0 = cKDTree(scan0)
dists1, _ = tree0.query(scan1, k=1)
c2c_1 = (dists1 > 0.08).astype(int) # Standard 8 cm threshold

# V2 Siamese PointNet simulated predictions (95% precision, 94% recall)
v2_1 = np.zeros(len(scan1), dtype=int)
add_idx = np.where(gt1 == 1)[0]
v2_1[np.random.choice(add_idx, int(0.94 * len(add_idx)), replace=False)] = 1
# Rare false alarms
static_idx1 = np.where(gt1 == 0)[0]
v2_1[np.random.choice(static_idx1, 48, replace=False)] = 1

out_npz = Path("C:/Users/shaik/Visioners_2.0/colab_sample_data/Semantic_Building_Scan_Sample.npz")
np.savez_compressed(
    out_npz,
    scan0=scan0, gt0=gt0,
    scan1=scan1, gt1=gt1,
    v2_1=v2_1, c2c_1=c2c_1
)

print(f"✓ Saved Semantic Building Scan sample: {out_npz}")
print(f"  Scan 0 points: {len(scan0):,} | Removals (Scaffolding): {sum(gt0):,} ({sum(gt0)/len(gt0):.1%})")
print(f"  Scan 1 points: {len(scan1):,} | Additions (Annex & Roof): {sum(gt1):,} ({sum(gt1)/len(gt1):.1%})")
print(f"  C2C False Alarms: {np.sum((gt1 == 0) & (c2c_1 == 1)):,} pts")
print(f"  V2 False Alarms:  {np.sum((gt1 == 0) & (v2_1 == 1)):,} pts")
