
import open3d as o3d
import numpy as np
from pathlib import Path

# Folder containing extracted PCD files
pcd_folder = Path(
    r"D:\CV_Project\dataset\Sequence 3\Copy of lidar-pcd\lidar-pcd"
)

pcd_files = sorted(pcd_folder.glob("*.pcd"))

print("Total PCD files:", len(pcd_files))

# Inspect the first 5 scans
for file in pcd_files[:5]:
    pcd = o3d.io.read_point_cloud(str(file))
    points = np.asarray(pcd.points)

    print(f"\nFile: {file.name}")
    print("Number of points:", len(points))
    print("First point:", points[0])
    print("Coordinate range:")
    print("  Min:", points.min(axis=0))
    print("  Max:", points.max(axis=0))