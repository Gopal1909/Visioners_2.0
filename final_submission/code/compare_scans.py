
import open3d as o3d
from pathlib import Path

pcd_folder = Path(
    r"D:\CV_Project\dataset\Sequence 3\Copy of lidar-pcd\lidar-pcd"
)

pcd_files = sorted(pcd_folder.glob("*.pcd"))

# Select two scans
pcd1 = o3d.io.read_point_cloud(str(pcd_files[0]))
pcd2 = o3d.io.read_point_cloud(str(pcd_files[1]))

# Color the scans differently
pcd1.paint_uniform_color([0, 0, 1])  # Blue
pcd2.paint_uniform_color([1, 0, 0])  # Red

print("Scan 1:", pcd_files[0].name)
print("Scan 2:", pcd_files[1].name)
print("Points in scan 1:", len(pcd1.points))
print("Points in scan 2:", len(pcd2.points))

o3d.visualization.draw_geometries(
    [pcd1, pcd2],
    window_name="Two LiDAR Scans"
)