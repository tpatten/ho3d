import os
import numpy as np
import open3d as o3d


# Path and file names
ifnet_dir = '/home/tpatten/Code/if-net/shapenet/data/ho3d/obj_learn_refine'
ho3d_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/reconstructions'
scene_id = 'SS1'
mesh_tsdf = 'mesh.ply'
mesh_ifnet = 'ifnet_recon_bop_clean.off'
# mesh_poisson = '_AnnoPoses_tsdf_aligned_clean_poisson.ply'
mesh_poisson = '_AnnoPoses_inPaintRend_tsdf_aligned_clean_poisson.ply'
visualize = True
downsample_size = 5
tsdf_to_ifnet_dist_threshold = 5.0  # 5.0
distance_thresholds = [5.0, 20.0]  # [5.0, 20.0]
outlier_rm_std_ratio = 0.5
outlier_rm_nb_neighbors = 100


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


# Load meshes
mesh_tsdf = o3d.io.read_triangle_mesh(os.path.join(ifnet_dir, scene_id + '_300', mesh_tsdf))
mesh_ifnet_300 = o3d.io.read_triangle_mesh(os.path.join(ifnet_dir, scene_id + '_300', mesh_ifnet))
mesh_ifnet_3000 = o3d.io.read_triangle_mesh(os.path.join(ifnet_dir, scene_id + '_3000', mesh_ifnet))
mesh_poisson = o3d.io.read_triangle_mesh(os.path.join(ho3d_dir, scene_id + mesh_poisson))

#o3d.visualization.draw_geometries([mesh_tsdf, mesh_ifnet_300, mesh_ifnet_3000])
#import sys
#sys.exit(0)

# Get the points
points_tsdf = np.asarray(mesh_tsdf.vertices)
points_ifnet = np.concatenate((np.asarray(mesh_ifnet_300.vertices), np.asarray(mesh_ifnet_3000.vertices)))
#points_ifnet = np.asarray(mesh_ifnet_300.vertices)
points_poisson = np.asarray(mesh_poisson.vertices)

# Clean up the tsdf points
pcd_temp = o3d.geometry.PointCloud()
pcd_temp.points = o3d.utility.Vector3dVector(np.asarray(mesh_tsdf.vertices))
outlier_rm_nb_neighbors = 50  # Higher is more aggressive 50
outlier_rm_std_ratio = 0.01  # Smaller is more aggressive 0.01
#pcd_temp, _ = pcd_temp.remove_statistical_outlier(
#    nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
#points_tsdf = np.copy(np.asarray(pcd_temp.voxel_down_sample(voxel_size=downsample_size).points))
points_tsdf = np.copy(np.asarray(pcd_temp.points))

# Clean up the if-net points
pcd_temp.points = o3d.utility.Vector3dVector(points_ifnet)
#pcd_temp, _ = pcd_temp.remove_statistical_outlier(
#    nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
#points_ifnet = np.copy(np.asarray(pcd_temp.voxel_down_sample(voxel_size=downsample_size).points))
points_ifnet = np.copy(np.asarray(pcd_temp.points))

# KD tree of ifnet meshes
pcd_ifnet = o3d.geometry.PointCloud()
pcd_ifnet.points = o3d.utility.Vector3dVector(np.copy(points_ifnet))
kd_tree_ifnet = o3d.geometry.KDTreeFlann(pcd_ifnet)

# Find points in the TSDF reconstruction that should be removed because far from the IF-Net reconstructions
remove_points = []
for i in range(points_tsdf.shape[0]):
    pt = np.copy(points_tsdf[i, :])
    [_, idx, _] = kd_tree_ifnet.search_knn_vector_3d(pt, 1)
    obj_pt = np.asarray(pcd_ifnet.points)[idx[0], :]
    dd = np.linalg.norm(pt - obj_pt)
    if dd > tsdf_to_ifnet_dist_threshold:
        remove_points.append(pt)
        points_tsdf[i, :] = [np.nan, np.nan, np.nan]

# Remove all outliers from the points
points_tsdf = points_tsdf[~np.isnan(points_tsdf).any(axis=1)]

# Make KD trees for the search through the TSDF reconstruction and the poisson reconstruction
pcd_tsdf = o3d.geometry.PointCloud()
pcd_tsdf.points = o3d.utility.Vector3dVector(np.copy(points_tsdf))
kd_tree_tsdf = o3d.geometry.KDTreeFlann(pcd_tsdf)
pcd_poisson = o3d.geometry.PointCloud()
pcd_poisson.points = o3d.utility.Vector3dVector(np.copy(points_poisson))
kd_tree_poisson = o3d.geometry.KDTreeFlann(pcd_poisson)

# Get a convex hull of the TSDF reconstruction
ch1, ch_idxs1 = pcd_tsdf.compute_convex_hull()

# Find points in the IF-Net reconstructions that should be added to the mesh
points_ifnet = np.asarray(pcd_ifnet.voxel_down_sample(voxel_size=downsample_size).points)
new_points = []
for pt in points_ifnet:
    # If the point is outside the convex hull, then add it immediately
    if not point_in_hull(pt, points_tsdf, ch_idxs1):
        new_points.append(pt)
    # Otherwise, the point can be inside but must be far from the original mesh, yet close to the poisson reconstruction
    else:
        [_, idx1, _] = kd_tree_tsdf.search_knn_vector_3d(pt, 1)
        obj_pt_tsdf = np.asarray(pcd_tsdf.points)[idx1[0], :]
        d2tsdf = np.linalg.norm(pt - obj_pt_tsdf)
        [_, idx3, _] = kd_tree_poisson.search_knn_vector_3d(pt, 1)
        obj_pt_poisson = np.asarray(pcd_poisson.points)[idx3[0], :]
        d2poisson = np.linalg.norm(pt - obj_pt_poisson)
        if d2tsdf > distance_thresholds[0] and d2poisson < distance_thresholds[1]:
            new_points.append(pt)

# Display the points
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(np.asarray(new_points))
new_pcd.paint_uniform_color([0.2, 0.2, 0.7])
if len(remove_points) > 0:
    rem_pcd = o3d.geometry.PointCloud()
    rem_pcd.points = o3d.utility.Vector3dVector(np.asarray(remove_points))
    rem_pcd.paint_uniform_color([0.7, 0.2, 0.2])
pcd_tsdf.paint_uniform_color([0.5, 0.5, 0.5])

# Visualize
if visualize:
    if len(remove_points) > 0:
        o3d.visualization.draw_geometries([pcd_tsdf, new_pcd, rem_pcd])
    else:
        o3d.visualization.draw_geometries([pcd_tsdf, new_pcd])

# Save
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(np.append(points_tsdf, np.asarray(new_points), axis=0))
o3d.io.write_point_cloud(os.path.join(ho3d_dir, scene_id + '_merged_refine.ply'), merged_pcd)
