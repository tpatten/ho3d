import os
import argparse
import numpy as np
import open3d as o3d
import copy


def remove_outliers(cloud, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                    raduis_rm_min_nb_points=0, raduis_rm_radius=0.):
    # Copy the input cloud
    in_cloud = copy.deepcopy(cloud)

    if o3d.__version__ == "0.7.0.0":
        # Statistical outlier removal
        if outlier_rm_nb_neighbors > 0 and outlier_rm_std_ratio > 0:
            in_cloud, _ = o3d.geometry.statistical_outlier_removal(
                in_cloud, nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
        # Radius outlier removal
        if raduis_rm_min_nb_points > 0 and raduis_rm_radius > 0:
            in_cloud, _ = o3d.geometry.radius_outlier_removal(
                in_cloud, nb_points=raduis_rm_min_nb_points, radius=raduis_rm_radius)
    else:
        # Statistical outlier removal
        if outlier_rm_nb_neighbors > 0 and outlier_rm_std_ratio > 0:
            in_cloud, _ = in_cloud.remove_statistical_outlier(
                nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
        # Radius outlier removal
        if raduis_rm_min_nb_points > 0 and raduis_rm_radius > 0:
            in_cloud, _ = in_cloud.remove_radius_outlier(nb_points=raduis_rm_min_nb_points, radius=raduis_rm_radius)

    return in_cloud


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Convert a cloud to a mesh')
    parser.add_argument("filename", type=str, help="Name of the .ply file to convert to a mesh")
    args = parser.parse_args()

    print('==> Loading file {}'.format(args.filename))
    cloud = o3d.io.read_point_cloud(args.filename)

    # Downsample
    print('==> Downsampling')
    cloud = cloud.voxel_down_sample(voxel_size=0.0025)

    # Remove outliers
    print('==> Removing outliers')
    outlier_rm_nb_neighbors = 250  # Higher is more aggressive
    outlier_rm_std_ratio = 0.00001  # Smaller is more aggressive
    raduis_rm_min_nb_points = 250  # Don't change
    raduis_rm_radius_factor = 10  # Don't change
    voxel_size = 0.001
    print('Input points: {}'.format(np.asarray(cloud.points).shape[0]))
    cloud = remove_outliers(cloud,
                            outlier_rm_nb_neighbors=outlier_rm_nb_neighbors,
                            outlier_rm_std_ratio=outlier_rm_std_ratio,
                            raduis_rm_min_nb_points=0, raduis_rm_radius=0)
                            # raduis_rm_min_nb_points=raduis_rm_min_nb_points,
                            # raduis_rm_radius=voxel_size * raduis_rm_radius_factor)
    print('Output points: {}'.format(np.asarray(cloud.points).shape[0]))
    o3d.io.write_point_cloud('/home/tpatten/Data/cloud.ply', cloud)

    # Compute the normals
    print('==> Computing normals')
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Ball pivoting
    print('==> Generating mesh')
    radii = np.asarray([0.005, 0.01, 0.02, 0.04])
    mesh_recon = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii))
    out_filename = os.path.join(os.path.dirname(args.filename), 'mesh.ply')
    o3d.io.write_triangle_mesh(out_filename, mesh_recon)
