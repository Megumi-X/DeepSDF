import open3d as o3d
import numpy as np
from tqdm import tqdm

res = 1
point_cloud = o3d.io.read_point_cloud("./point_cloud/iteration_30000/point_cloud.ply")
# point_cloud = point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)[0]
point_cloud = point_cloud.voxel_down_sample(voxel_size=0.06 / res)
# point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.10, max_nn=200))
o3d.visualization.draw_geometries([point_cloud])
o3d.io.write_point_cloud("./point_cloud_downsampled.ply", point_cloud)

# center = np.average(np.asarray(point_cloud.points), axis=0)
# point_cloud.translate(-center)
# o3d.io.write_point_cloud("D:/IIIS_THU/Research/Seeing-through-surface/tumbler/tumbler_outer.ply", point_cloud)

voxel_size = 0.06 / res
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

filled_points = []
min_bound = voxel_grid.get_min_bound()
max_bound = voxel_grid.get_max_bound()
nx = int((max_bound[0] - min_bound[0]) / voxel_size) + 1
ny = int((max_bound[2] - min_bound[2]) / voxel_size) + 1
voxel_grid_centers = []
for voxel in voxel_grid.get_voxels():
    voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
    voxel_grid_centers.append(voxel_center)
voxel_grid_centers = np.array(voxel_grid_centers)
for i in tqdm(range(nx)):
    for j in range(ny):
        # line = np.where(np.isclose(voxel_grid_centers[:, 0], min_bound[0] + i * voxel_size, 1e-3, voxel_size / 10) & np.isclose(voxel_grid_centers[:, 1], min_bound[1] + j * voxel_size, 1e-3, voxel_size / 10))[0]
        indices = []
        for voxel in voxel_grid.get_voxels():
            if voxel.grid_index[0] == i and voxel.grid_index[2] == j:
                indices.append(voxel.grid_index)
        if len(indices) == 0:
            continue
        indices.sort(key=lambda x: voxel_grid.get_voxel_center_coordinate(x)[1])
        smallest_z = voxel_grid.get_voxel_center_coordinate(indices[0])[1]
        largest_z = voxel_grid.get_voxel_center_coordinate(indices[-1])[1]
        n = int((largest_z - smallest_z) / voxel_size) + 1
        for k in range(n):
            center = voxel_grid.get_voxel_center_coordinate(indices[0])
            center[1] += k * voxel_size
            for _ in range(1):
                rand_delta = np.random.rand(3) * voxel_size * 0.9
                filled_points.append(center + rand_delta)
filled_points = np.array(filled_points)

# for voxel in voxel_grid.get_voxels():
#     voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
#     for _ in range(5):
#         rand_delta = np.random.rand(3) * voxel_size / 4
#         filled_points.append(voxel_center + rand_delta)

# center_2 = np.average(filled_points, axis=0)
new_points = filled_points
filter_indices_down = np.where(new_points[:, 0] < 0)[0]
filter_indices_up = np.where(new_points[:, 0] >= 0)[0]
new_points_down = new_points[filter_indices_down, :]
new_points_up = new_points[filter_indices_up, :]
new_point_cloud_down = o3d.geometry.PointCloud()
new_point_cloud_up = o3d.geometry.PointCloud()
new_point_cloud = o3d.geometry.PointCloud()
for point in new_points_down:
    new_point_cloud_down.points.append(point)
    point_cloud.points.append(point)
    new_point_cloud.points.append(point)
for point in new_points_up:
    new_point_cloud_up.points.append(point)
    point_cloud.points.append(point)
    new_point_cloud.points.append(point)

# 可视化结果，检查填充效果
o3d.visualization.draw_geometries([point_cloud])
# o3d.visualization.draw_geometries([surface_point_cloud])
o3d.visualization.draw_geometries([new_point_cloud])
o3d.io.write_point_cloud("./suzanne_inner.ply", new_point_cloud)