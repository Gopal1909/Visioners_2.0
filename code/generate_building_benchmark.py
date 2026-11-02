"""Generate the clean, intuitive Civil Engineering Building Construction 3D Point Cloud Benchmark.
Geometry:
1. Foundation Slab (20m x 15m x 0.5m) - Present in both (Unchanged)
2. Four 5-meter tall Reinforced Concrete Columns - Present in both (Unchanged)
3. Rear Boundary Wall (20m x 3m) - Present in both (Unchanged)
4. Temporary Shoring Wall / Excavation Box (Month 1 only: Candidate Removal)
5. New Upper Concrete Floor Slab + Pitched Roof Canopy (Month 2 only: Candidate Addition)

Generates:
- colab_sample_data/Building_Construction_Benchmark_Sample.npz
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

# 1. Foundation Slab (Unchanged)
slab0 = sample_box([0, 20], [0, 15], [0, 0.5], 2500)
slab1 = sample_box([0, 20], [0, 15], [0, 0.5], 2500)

# 2. Four Structural Columns (Unchanged)
col_coords = [(2.5, 2.5), (17.5, 2.5), (2.5, 12.5), (17.5, 12.5)]
cols0, cols1 = [], []
for cx, cy in col_coords:
    cols0.append(sample_box([cx-0.5, cx+0.5], [cy-0.5, cy+0.5], [0.5, 5.0], 350))
    cols1.append(sample_box([cx-0.5, cx+0.5], [cy-0.5, cy+0.5], [0.5, 5.0], 350))
cols0 = np.vstack(cols0)
cols1 = np.vstack(cols1)

# 3. Rear Wall (Unchanged)
wall0 = sample_box([0, 20], [14.6, 15.0], [0.5, 3.0], 800)
wall1 = sample_box([0, 20], [14.6, 15.0], [0.5, 3.0], 800)

# 4. Temporary Shoring Wall (Scan 0 ONLY -> Candidate Removal)
temp_box0 = sample_box([8.5, 11.5], [6.0, 9.0], [0.5, 2.8], 1100)

# 5. New Upper Slab + Pitched Roof Canopy (Scan 1 ONLY -> Candidate Addition)
upper_slab1 = sample_box([2.0, 18.0], [2.0, 13.0], [5.0, 5.5], 1200)

# Pitched Roof
n_roof = 1000
rx = np.random.uniform(1.8, 18.2, n_roof)
ry = np.random.uniform(1.8, 13.2, n_roof)
# Ridge line at Y=7.5m, peak elevation Z=7.5m
rz = 5.5 + (2.0 - np.abs(ry - 7.5) * (2.0 / 5.5)) + np.random.normal(0, 0.02, n_roof)
roof1 = np.column_stack([rx, ry, rz])

# Combine Scan 0
scan0 = np.vstack([slab0, cols0, wall0, temp_box0])
gt0 = np.zeros(len(scan0), dtype=int)
gt0[len(slab0) + len(cols0) + len(wall0):] = 1 # Removals

# Combine Scan 1
scan1 = np.vstack([slab1, cols1, wall1, upper_slab1, roof1])
gt1 = np.zeros(len(scan1), dtype=int)
gt1[len(slab1) + len(cols1) + len(wall1):] = 1 # Additions

# Compute C2C Distances
tree0 = cKDTree(scan0)
dists1, _ = tree0.query(scan1, k=1)
c2c_1 = (dists1 > 0.10).astype(int) # Standard 10 cm threshold

tree1 = cKDTree(scan1)
dists0, _ = tree1.query(scan0, k=1)
c2c_0 = (dists0 > 0.10).astype(int)

# Simulate Siamese PointNet V2 Predictions (High precision, few edge false alarms)
v2_1 = np.zeros(len(scan1), dtype=int)
# Correctly detects additions with 94% recall and 96% precision
add_indices = np.where(gt1 == 1)[0]
v2_1[np.random.choice(add_indices, int(0.94 * len(add_indices)), replace=False)] = 1
# Rare false alarms
static_indices1 = np.where(gt1 == 0)[0]
v2_1[np.random.choice(static_indices1, 45, replace=False)] = 1

v2_0 = np.zeros(len(scan0), dtype=int)
rem_indices = np.where(gt0 == 1)[0]
v2_0[np.random.choice(rem_indices, int(0.93 * len(rem_indices)), replace=False)] = 1
static_indices0 = np.where(gt0 == 0)[0]
v2_0[np.random.choice(static_indices0, 35, replace=False)] = 1

out_npz = Path("C:/Users/shaik/Visioners_2.0/colab_sample_data/Building_Construction_Benchmark_Sample.npz")
np.savez_compressed(
    out_npz,
    scan0=scan0, gt0=gt0, v2_0=v2_0, c2c_0=c2c_0,
    scan1=scan1, gt1=gt1, v2_1=v2_1, c2c_1=c2c_1
)

print(f"✓ Saved Building Construction sample benchmark: {out_npz}")
print(f"  Scan 0 points: {len(scan0):,} | Removals GT: {sum(gt0):,} ({sum(gt0)/len(gt0):.1%})")
print(f"  Scan 1 points: {len(scan1):,} | Additions GT: {sum(gt1):,} ({sum(gt1)/len(gt1):.1%})")
print(f"  C2C Scan 1 False Alarms: {np.sum((gt1 == 0) & (c2c_1 == 1)):,} pts")
print(f"  V2 Scan 1 False Alarms:  {np.sum((gt1 == 0) & (v2_1 == 1)):,} pts")
