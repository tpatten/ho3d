# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import os
import argparse
import numpy as np
import json
#from utils.grasp_utils import *
import open3d as o3d
import copy
from enum import IntEnum
from scipy.spatial.distance import directed_hausdorff


class ReconstructionMethod(IntEnum):
    POISSON = 1
    BALL_PIVOT = 2
    NONE = 3


class PoissonSurfaceReconstructor:
    def __init__(self, args):
        self.args = args

        # Load the json file containing the mapping of the scene to YCB model
        model_name_data = None
        with open(os.path.join(self.args.ho3d_to_ycb_map_path)) as f:
            model_name_data = json.load(f)

        # Get the scene key
        if self.args.scene == '':
            scene_key = self.args.model_file
            # Remove the path if it is in the filename
            scene_key = os.path.split(scene_key)[1]
            scene_key = os.path.splitext(scene_key)[0]
            # Get the characters in the model file before the first underscore
            scene_key = scene_key.split('_')[0]
        else:
            scene_key = self.args.scene
        scene_key = ''.join([i for i in scene_key if not i.isdigit()])

        # Load the reference model
        if self.args.bop_format:
            ycb_model_filename = os.path.join(
                self.args.bop_model_path, 'obj_' + model_name_data[scene_key]['bop'].zfill(6) + '.ply')
        else:
            ycb_model_filename = os.path.join(
                self.args.ycb_model_path, model_name_data[scene_key]['ycbv'], 'textured_simple.obj')
        mesh_ycb = o3d.io.read_triangle_mesh(ycb_model_filename)

        mesh_ycb_points = copy.deepcopy(np.asarray(mesh_ycb.vertices))
        mesh_ycb_centroid = np.mean(mesh_ycb_points, axis=0)

        # Load the mesh
        # If the first character is an underscore, then this is stored outside the scene files
        '''
        if self.args.model_file[0] == '_':
            tsdf_filename = os.path.join(self.args.ho3d_path, self.args.scene + self.args.model_file)
        else:
            if self.args.scene == '':
                tsdf_filename = os.path.join(self.args.ho3d_path, self.args.model_file)
            else:
                tsdf_filename = os.path.join(self.args.ho3d_path, self.args.scene, self.args.model_file)
        '''
        tsdf_filename = self.args.model_file
        print('Processing {}'.format(tsdf_filename))
        mesh = o3d.io.read_triangle_mesh(tsdf_filename)

        # Rotate to correct coordinate system
        if self.args.flip_from_openGL:
            mesh.rotate(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32), center=(0, 0, 0))

        # Change scale and offset if this is the BOP format
        if self.args.bop_format:
            # If scene contains numbers, remove them
            offset_bop = np.asarray(model_name_data[scene_key]['offset_bop'])
            mesh.translate(-offset_bop)
            mesh.scale(1000, center=(0, 0, 0))




        # o3d.visualization.draw_geometries([mesh])

        # First pass removes small clusters and creates intermediate surface reconstruction
        # if self.args.model_file[0] == '_':
        # mesh = self.remove_noise(mesh, mesh_ycb_centroid)
        # mesh.compute_vertex_normals()
        # mesh = self.reconstruction(mesh, remove_outliers=False, r_method=ReconstructionMethod.BALL_PIVOT)
        # o3d.visualization.draw_geometries([mesh])

        # Second pass applies filter and final Poisson surface reconstruction
        #mesh = self.remove_noise(mesh, mesh_ycb_centroid)
        # o3d.visualization.draw_geometries([mesh])
        #  if self.args.model_file[0] == '_':
        # mesh = self.taubin_filer(mesh)

        if self.args.r_method == ReconstructionMethod.POISSON or self.args.r_method == ReconstructionMethod.BALL_PIVOT:
            mesh = self.reconstruction(mesh, remove_outliers=self.args.clean_up_outlier_removal,
                                       r_method=self.args.r_method)
            # o3d.visualization.draw_geometries([mesh])

        # Clean up
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        mesh = mesh.remove_unreferenced_vertices()

        ''''
        # Rotate to correct coordinate system
        if self.args.flip_from_openGL:
            mesh.rotate(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32), center=(0, 0, 0))

        # Change scale and offset if this is the BOP format
        if self.args.bop_format:
            # If scene contains numbers, remove them
            offset_bop = np.asarray(model_name_data[scene_key]['offset_bop'])
            mesh.translate(-offset_bop)
            mesh.scale(1000, center=(0, 0, 0))
        '''

        # Save
        if self.args.save:
            if self.args.r_method == ReconstructionMethod.POISSON:
                save_filename = tsdf_filename.replace('.ply', '_poisson.ply')
            elif self.args.r_method == ReconstructionMethod.BALL_PIVOT:
                save_filename = tsdf_filename.replace('.ply', '_ball_pivot.ply')
            else:
                save_filename = tsdf_filename.replace('.ply', '_clean.ply')
            print('Saving {}'.format(save_filename))
            o3d.io.write_triangle_mesh(save_filename, mesh)
            # import trimesh
            # tmesh = trimesh.load(save_filename)
            # tmesh.export(save_filename.replace('.ply', '_trimesh.ply'))

        # Visualize
        if self.args.visualize:
            hd_mesh_to_model = directed_hausdorff(np.asarray(mesh.vertices), np.asarray(mesh_ycb.vertices))[0]
            hd_model_to_mesh = directed_hausdorff(np.asarray(mesh_ycb.vertices), np.asarray(mesh.vertices))[0]
            print('Hausdorff (mesh to model): {}'.format(hd_mesh_to_model))
            print('Hausdorff (model to mesh): {}'.format(hd_model_to_mesh))
            print('Hausdorff (max):           {}'.format(max(hd_mesh_to_model, hd_model_to_mesh)))
            o3d.visualization.draw_geometries([mesh, mesh_ycb])
            # o3d.visualization.draw_geometries([mesh])

    @staticmethod
    def remove_noise(mesh_in, reference_centroid=None):
        # Remove small (noisy) parts
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_in.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if reference_centroid is None:
            largest_cluster_idx = cluster_n_triangles.argmax()
        else:
            cluster_index_pairs = [(cix, i) for i, cix in enumerate(cluster_n_triangles)]
            sorted_clusters = sorted(cluster_index_pairs, reverse=True)

            # Compute the centroid of each cluster
            faces = np.asarray(mesh_in.triangles)
            vertices = np.asarray(mesh_in.vertices)
            # print(triangle_clusters.shape, faces.shape, faces.min(), faces.max(), vertices.shape)
            for s in sorted_clusters:
                # Get the triangle indices
                triangles_indices = np.where(triangle_clusters == s[1])[0]  # Bool array
                cluster_faces = faces[triangles_indices]
                cluster_vertices = vertices[cluster_faces]
                cluster_vertices_set = set()
                for vert in cluster_vertices:
                    for v in vert:
                        cluster_vertices_set.add((v[0], v[1], v[2]))
                #print(s[0], triangles_indices.shape[0], cluster_faces.shape[0], cluster_vertices.shape[0],
                #      len(cluster_vertices_set))

                # Compute the centroid
                x = y = z = 0.0
                for pt_set in cluster_vertices_set:
                    x += pt_set[0]
                    y += pt_set[1]
                    z += pt_set[2]
                x /= len(cluster_vertices_set)
                y /= len(cluster_vertices_set)
                z /= len(cluster_vertices_set)

                dist_to_centroid = np.linalg.norm(reference_centroid -
                                                  np.asarray([x, y, z]).reshape(reference_centroid.shape))
                if dist_to_centroid < 50.0:
                    largest_cluster_idx = s[1]
                    break

            #largest_cluster_idx = sorted(cluster_index_pairs, reverse=True)[0][1]

        mesh_out = copy.deepcopy(mesh_in)
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_out.remove_triangles_by_mask(triangles_to_remove)

        return mesh_out

    @staticmethod
    def taubin_filer(mesh_in):
        mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=3)
        mesh_out.compute_vertex_normals()

        # mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=2)
        # mesh_out.compute_vertex_normals()

        return mesh_out

    def reconstruction(self, mesh_in, remove_outliers=False, r_method=ReconstructionMethod.POISSON):
        # Sample points from mesh
        pcd = mesh_in.sample_points_poisson_disk(number_of_points=2500, init_factor=5)
        # Try to remove outliers
        if remove_outliers:
            pcd = self.remove_outliers(pcd,
                                       outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                       outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
                                       raduis_rm_min_nb_points=0,
                                       raduis_rm_radius=0)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # o3d.visualization.draw_geometries([pcd])

        if r_method == ReconstructionMethod.POISSON:
            # Run Poisson reconstruction
            # mesh_recon, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=16)
            mesh_recon, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=16)
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
    parser = argparse.ArgumentParser(description='HO-3D Clean up TSDF reconstruction with Poisson reconstruction')
    parser.add_argument("model_file", type=str, help="The TSDF model file to be cleaned up")
    parser.add_argument("--scene", type=str, help="Sequence of the dataset", default='')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/Data/bop_test/ho3dbop/reconstructions'
    args.ycb_model_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/models'
    args.bop_model_path = '/home/tpatten/Data/bop/ycbv/models_eval'
    args.ho3d_to_ycb_map_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/ho3d_to_ycb.json'
    args.visualize = True
    args.save = True
    args.outlier_rm_nb_neighbors = 50  # Higher is more aggressive
    args.outlier_rm_std_ratio = 0.01  # Smaller is more aggressive
    args.clean_up_outlier_removal = True
    args.bop_format = True
    args.flip_from_openGL = False
    # [1] ReconstructionMethod.POISSON, [2] ReconstructionMethod.BALL_PIVOT, [3] ReconstructionMethod.NONE
    args.r_method = ReconstructionMethod.NONE
    psr = PoissonSurfaceReconstructor(args)
