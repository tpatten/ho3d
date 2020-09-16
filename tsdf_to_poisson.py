# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import open3d as o3d
import copy
from enum import IntEnum


class ReconstructionMethod(IntEnum):
    POISSON = 1
    BALL_PIVOT = 2


class PoissonSurfaceReconstructor:
    def __init__(self, args):
        self.args = args

        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(self.args.model_file)

        # First pass removes small clusters and creates intermediate surface reconstruction
        mesh = self.remove_noise(mesh)
        mesh = self.reconstruction(mesh, r_method=ReconstructionMethod.BALL_PIVOT)

        # Second pass applies filter and final Poisson surface reconstruction
        mesh = self.remove_noise(mesh)
        mesh = self.taubin_filer(mesh)
        mesh = self.reconstruction(mesh, r_method=ReconstructionMethod.POISSON)

        # Clean up
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        mesh = mesh.remove_unreferenced_vertices()

        # Rotate to correct coordinate system
        mesh.rotate(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32), center=(0, 0, 0))

        # Change scale and offset if this is the BOP format
        if self.args.bop_format:
            offset_bop = np.asarray([-0.00499, 0.0024, 0.01329])
            mesh.translate(-offset_bop)
            mesh.scale(1000, center=(0, 0, 0))

        # Save
        if self.args.save:
            filename = self.args.model_file.replace('.ply', '_poisson.ply')
            o3d.io.write_triangle_mesh(filename, mesh)

        # Visualize
        if self.args.visualize:
            # Load ycbv model
            if self.args.bop_format:
                mesh_ycb = o3d.io.read_triangle_mesh('/home/tpatten/Data/bop/ycbv/models_eval/obj_000012.ply')
            else:
                mesh_ycb = o3d.io.read_triangle_mesh(
                    '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/models/021_bleach_cleanser/textured_simple.obj')
            o3d.visualization.draw_geometries([mesh, mesh_ycb])

    @staticmethod
    def remove_noise(mesh_in):
        # Remove small (noisy) parts
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_in.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        mesh_out = copy.deepcopy(mesh_in)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_out.remove_triangles_by_mask(triangles_to_remove)

        return mesh_out

    @staticmethod
    def taubin_filer(mesh_in):
        mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=3)
        mesh_out.compute_vertex_normals()

        #mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=2)
        #mesh_out.compute_vertex_normals()

        return mesh_out

    def reconstruction(self, mesh_in, r_method=ReconstructionMethod.POISSON):
        # Sample points from mesh
        pcd = mesh_in.sample_points_poisson_disk(number_of_points=2500, init_factor=5)
        # o3d.visualization.draw_geometries([pcd])

        if r_method == ReconstructionMethod.POISSON:
            # Try to remove outliers
            if self.args.clean_up_outlier_removal:
                pcd = self.remove_outliers(pcd,
                                           outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                           outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
                                           raduis_rm_min_nb_points=0,
                                           raduis_rm_radius=0)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # Run Poisson reconstruction
            mesh_recon, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif r_method == ReconstructionMethod.BALL_PIVOT:
            # Ball pivoting
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh_recon = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        else:
            print('Unknown reconstruction method selected')
            mesh_recon = mesh_in

        return mesh_recon

    @staticmethod
    def remove_outliers(cloud, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                        raduis_rm_min_nb_points=0, raduis_rm_radius=0.):
        # Copy the input cloud
        in_cloud = copy.deepcopy(cloud)

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
    parser = argparse.ArgumentParser(description='HO-3D Object model reconstruction')
    args = parser.parse_args()
    args.model_file = '/home/tpatten/Data/Hands/HO3D/train/ABF10/GT_start0_max-1_skip1_tsdf.ply'
    args.visualize = True
    args.save = True
    args.outlier_rm_nb_neighbors = 50  # Higher is more aggressive
    args.outlier_rm_std_ratio = 0.01  # Smaller is more aggressive
    args.clean_up_outlier_removal = True
    args.bop_format = True
    psr = PoissonSurfaceReconstructor(args)
