
import open3d as o3d
import numpy as np

pcd_path = r"D:\CV_Project\dataset\Sequence 3\Copy of lidar-pcd\lidar-pcd\1654763770307803.pcd"

pcd = o3d.io.read_point_cloud(pcd_path)

points = np.asarray(pcd.points)

print("Number of points:", len(points))
print("First 5 points:")
print(points[:5])

print("\nMinimum coordinates:", points.min(axis=0))
print("Maximum coordinates:", points.max(axis=0))

print("\nHas colors:", pcd.has_colors())
print("Has normals:", pcd.has_normals())

o3d.visualization.draw_geometries([pcd])