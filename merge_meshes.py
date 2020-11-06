import os
import numpy as np
import open3d as o3d


def point_in_hull(pt, hull_pts, hull_idxs):
    # Make a new mesh with the new point
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points = o3d.utility.Vector3dVector(np.append(hull_pts, pt.reshape((1, 3)), axis=0))

    # Get the indices of the new hull
    _, idxs_n = pcd_n.compute_convex_hull()

    # If the sets of indices are the same, then the point is inside the hull
    if set(hull_idxs) == set(idxs_n):
        return True
    else:
        return False


def get_outliers(points, outlier_rm_nb_neighbors, outlier_rm_std_ratio):
    # Statistical outlier removal
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.copy(points))
    _, inliers = cloud.remove_statistical_outlier(nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)

    all_indices = set(range(points.shape[0]))
    outliers = all_indices - set(inliers)

    return outliers


# Path and file names
data_dir = '/home/tpatten/Code/if-net/shapenet/data/ho3d/ho3d_AnnoPoses'
v4r_dir = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train'
scene_id = 'SS2'
mesh_name_1 = 'mesh.ply'
mesh_name_2 = 'ifnet_recon_bop_clean.off'
mesh_name_3 = '_AnnoPoses_tsdf_aligned_clean_poisson.ply'
visualize = True
downsample_size = 5
mesh1to2_dist_threshold = 5.0
distance_thresholds = [5.0, 20.0]
outlier_rm_std_ratio = 0.5
outlier_rm_nb_neighbors = 100

# Load meshes
mesh1 = o3d.io.read_triangle_mesh(os.path.join(data_dir, scene_id, mesh_name_1))
mesh2 = o3d.io.read_triangle_mesh(os.path.join(data_dir, scene_id, mesh_name_2))
mesh3 = o3d.io.read_triangle_mesh(os.path.join(v4r_dir, scene_id + mesh_name_3))

# Get the points
points1 = np.asarray(mesh1.vertices)
points2 = np.asarray(mesh2.vertices)
points3 = np.asarray(mesh3.vertices)

# KD tree of mesh2
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(np.copy(points2))
kd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

# Find points in mesh1 that should be removed because far from mesh2
remove_points = []
for i in range(points1.shape[0]):
    pt = np.copy(points1[i, :])
    [_, idx, _] = kd_tree2.search_knn_vector_3d(pt, 1)
    obj_pt = np.asarray(pcd2.points)[idx[0], :]
    dd = np.linalg.norm(pt - obj_pt)
    if dd > mesh1to2_dist_threshold:
        remove_points.append(pt)
        points1[i, :] = [np.nan, np.nan, np.nan]

# Remove all outliers from the points
points1 = points1[~np.isnan(points1).any(axis=1)]

# Make KD trees for the search through mesh1 and mesh3
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(np.copy(points1))
kd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(np.copy(points3))
kd_tree3 = o3d.geometry.KDTreeFlann(pcd3)

# Get a convex hull of mesh1
ch1, ch_idxs1 = pcd1.compute_convex_hull()

# Find points in mesh2 that should be added to mesh1
points2 = np.asarray(pcd2.voxel_down_sample(voxel_size=downsample_size).points)
new_points = []
for pt in points2:
    '''
    [_, idx, _] = kd_tree1.search_knn_vector_3d(pt, 1)
    obj_pt = np.asarray(pcd1.points)[idx[0], :]
    dd = np.linalg.norm(pt - obj_pt)
    if 5.0 < dd < 20.0:
        # Check if it is in the convex hull
        if not point_in_hull(pt, points1, ch_idxs1):
            # Add point to the cloud
            new_points.append(pt)
    '''
    # If the point is outside the convex hull, then add it immediately
    if not point_in_hull(pt, points1, ch_idxs1):
        new_points.append(pt)
    # Otherwise, the point can be inside but must be far from the original mesh, yet close to the poisson reconstruction
    else:
        [_, idx1, _] = kd_tree1.search_knn_vector_3d(pt, 1)
        obj_pt1 = np.asarray(pcd1.points)[idx1[0], :]
        d1 = np.linalg.norm(pt - obj_pt1)
        [_, idx3, _] = kd_tree3.search_knn_vector_3d(pt, 1)
        obj_pt3 = np.asarray(pcd3.points)[idx3[0], :]
        d3 = np.linalg.norm(pt - obj_pt3)
        if d1 > 10.0 and d3 < 10.0:
            new_points.append(pt)

# Display the points
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(np.asarray(new_points))
new_pcd.paint_uniform_color([0.2, 0.2, 0.7])
if len(remove_points) > 0:
    rem_pcd = o3d.geometry.PointCloud()
    rem_pcd.points = o3d.utility.Vector3dVector(np.asarray(remove_points))
    rem_pcd.paint_uniform_color([0.7, 0.2, 0.2])
pcd1.paint_uniform_color([0.5, 0.5, 0.5])

# Visualize
if visualize:
    if len(remove_points) > 0:
        o3d.visualization.draw_geometries([pcd1, new_pcd, rem_pcd])
    else:
        o3d.visualization.draw_geometries([pcd1, new_pcd])

# Save
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(np.append(points1, np.asarray(new_points), axis=0))
o3d.io.write_point_cloud(os.path.join(data_dir, scene_id, 'mesh_cloud.ply'), merged_pcd)
