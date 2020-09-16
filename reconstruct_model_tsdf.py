# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import cv2
import open3d as o3d
import copy
import transforms3d as tf3d
from enum import IntEnum


HAND_MASK_DIR = 'hand'
OBJECT_MASK_DIR = 'object'
HAND_MASK_VISIBLE_DIR = 'hand_vis'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'
COORD_CHANGE_MAT = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
ICP_THRESH = 0.02
RANSAC_THRESH = 0.015


class ICPMethod(IntEnum):
    Point2Point = 1
    Point2Plane = 2


class RegMethod(IntEnum):
    GT = 1
    GT_ICP = 2
    GT_ICP_FULL = 4


class ModelReconstructor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.rgb = None
        self.depth = None
        self.anno = None
        self.scene_pcd = None
        self.prev_cloud = None
        self.prev_pose = np.eye(4)
        if args.mask_erosion_kernel > 0:
            self.erosion_kernel = np.ones((args.mask_erosion_kernel, args.mask_erosion_kernel), np.uint8)
        else:
            self.erosion_kernel = None
        self.volume = None

        # Reconstruct the model
        loaded_pcd = None
        if self.args.model_file == '':
            loaded_pcd = self.reconstruct_object_model()

        if self.args.visualize:
            if not self.args.model_file == '':
                loaded_pcd = self.load_object_model(self.args.model_file)

                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                #                             outlier_rm_std_ratio=self.args.outlier_rm_std_ratio * 0.0001,
                #                             raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                #                             raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                #                             raduis_rm_min_nb_points=1250,
                #                             raduis_rm_radius=self.args.voxel_size * 10)

                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=0.,
                #                             outlier_rm_std_ratio=0.,
                #                             raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                #                             raduis_rm_radius=self.args.voxel_size * 5)

            loaded_pcd = remove_outliers(loaded_pcd,
                                         outlier_rm_nb_neighbors=250,
                                         outlier_rm_std_ratio=0.01,
                                         raduis_rm_min_nb_points=0, raduis_rm_radius=0)

            self.visualize(mask_pcds=[loaded_pcd])
            # self.visualize(mask_pcds=[self.load_object_model(self.args.model_file)])

            if self.args.save:
                loaded_pcd = o3d.geometry.voxel_down_sample(loaded_pcd, 0.001)
                o3d.geometry.estimate_normals(loaded_pcd,
                                              search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                  radius=0.1, max_nn=30))
                points = np.asarray(loaded_pcd.points)
                normals = np.asarray(loaded_pcd.normals)

                if self.args.model_file == '':
                    filename = os.path.join(self.base_dir, self.data_split, self.args.scene,
                                            str(self.args.reg_method).split('.')[1])
                    if self.args.reg_method == RegMethod.GT_ICP or self.args.reg_method == RegMethod.GT_ICP_FULL:
                        filename += '_' + str(self.args.icp_method).split('.')[1]
                    filename += '_start' + str(self.args.start_frame) + '_max' + str(self.args.max_num) + \
                                '_skip' + str(self.args.skip) + '_clean.xyz'
                else:
                    filename = self.args.model_file.split('.')[0] + '_clean.xyz'
                f = open(filename, "w")
                for i in range(len(points)):
                    f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
                                                         normals[i, 0], normals[i, 1], normals[i, 2]))
                f.close()

                o3d.io.write_point_cloud(filename.replace('.xyz', '.ply'), loaded_pcd)

    def reconstruct_object_model(self):
        # Get the scene ids
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb')))

        # Load the camera parameters
        cam_params = self.read_camera_intrinsics(self.args.scene)

        # Initialize the TSDF volume
        if self.args.construct_tsdf:
            self.volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=self.args.sdf_voxel_length,
                sdf_trunc=self.args.sdf_trunc,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        # For each frame in the directory
        cam_poses = []
        est_cam_poses = []
        mask_pcds = []
        scene_pcds = []
        processed_frames = []
        counter = self.args.start_frame
        num_processed = 0
        while counter < len(frame_ids):
            # Get the id
            frame_id = frame_ids[counter].split('.')[0]
            # Create the filename for the metadata file
            meta_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'meta',
                                         str(frame_id) + '.pkl')
            if self.args.max_num == 0:
                print('[{}/{}] Processing file {}'.format(counter, len(frame_ids), meta_filename))
            else:
                print('[{}/{}] Processing file {}'.format(num_processed, self.args.max_num, meta_filename))

            # Read image, depths maps and annotations
            self.rgb, self.depth, self.anno, scene_pcd = self.load_data(self.args.scene, frame_id)

            # Read the mask
            mask_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'mask',
                                         OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png')
            if not os.path.exists(mask_filename):
                print('No mask available for frame {}'.format(frame_id))
                counter += self.args.skip
                continue

            mask = cv2.imread(mask_filename)[:, :, 0]
            if self.erosion_kernel is not None:
                mask = cv2.erode(mask, self.erosion_kernel, iterations=1)

            # Extract the masked point cloud
            cloud, colors = self.image_to_world(mask, cut_z=np.linalg.norm(self.anno['objTrans'])*1.1)
            if cloud.shape[0] == 0:
                print('Empty cloud for frame {}'.format(frame_id))
                counter += self.args.skip
                continue

            # Transform the cloud and get the camera position
            cloud, cam_pose, est_cloud, est_cam_pose = self.transform_to_object_frame(cloud)

            # Create the point cloud for visualization
            mask_pcd = o3d.geometry.PointCloud()
            if self.args.reg_method == RegMethod.GT:
                mask_pcd.points = o3d.utility.Vector3dVector(cloud)
            else:
                mask_pcd.points = o3d.utility.Vector3dVector(est_cloud)
            mask_pcd.colors = o3d.utility.Vector3dVector(colors)
            # mask_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Remove outliers
            mask_pcd = remove_outliers(mask_pcd,
                                       outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                       outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
                                       raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                                       raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
            #mask_pcd = remove_outliers(mask_pcd,
            #                           outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
            #                           outlier_rm_std_ratio=self.args.outlier_rm_std_ratio)

            # Add to lists
            cam_poses.append(cam_pose)
            est_cam_poses.append(est_cam_pose)
            mask_pcds.append(mask_pcd)
            scene_pcds.append(scene_pcd)
            processed_frames.append(frame_id)

            # Add to SDF volume
            if self.args.construct_tsdf:
                masked_rgb = self.apply_mask(self.rgb, mask)
                masked_depth = self.apply_mask(self.depth, mask)
                masked_rgb = o3d.geometry.Image(masked_rgb)
                masked_depth = o3d.geometry.Image((masked_depth * 1000).astype(np.float32))
                rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(
                    masked_rgb, masked_depth, depth_trunc=self.args.sdf_depth_trunc, convert_rgb_to_intensity=False)
                self.volume.integrate(rgbd, cam_params, np.linalg.inv(cam_pose))

            # Increment counters
            counter += self.args.skip
            num_processed += 1

            # Update the previous cloud and pose
            if self.args.reg_method == RegMethod.GT_ICP_FULL:
                if self.prev_cloud is None:
                    self.prev_cloud = mask_pcd
                else:
                    self.prev_cloud.points.extend(mask_pcd.points)
                    self.prev_cloud.colors.extend(mask_pcd.colors)
            else:
                self.prev_cloud = mask_pcd
            self.prev_pose = est_cam_pose

            # Down sample
            if self.args.voxel_size > 0:
                self.prev_cloud = o3d.geometry.voxel_down_sample(self.prev_cloud, self.args.voxel_size)

            # Exit if reached the limit
            if self.args.max_num > 0 and num_processed == self.args.max_num:
                break

            # Save intermediate results
            if self.args.save and self.args.save_intermediate and num_processed % 20 == 0:
                base_dir = os.path.join(self.base_dir, self.data_split, self.args.scene)
                self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses, mask_pcds,
                                                  frame_id=frame_id)
        # Get the mesh from the TSDF
        tsdf_mesh = None
        if self.args.construct_tsdf:
            tsdf_mesh = self.volume.extract_triangle_mesh()
            tsdf_mesh.compute_vertex_normals()

        # Save
        combined_pcd = None
        if self.args.save:
            base_dir = os.path.join(self.base_dir, self.data_split, self.args.scene)
            combined_pcd = self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses,
                                                             mask_pcds, tsdf_mesh)

        # Visualize
        if self.args.visualize:
            scene_pcds = None
            if self.args.reg_method == RegMethod.GT:
                self.visualize(cam_poses, mask_pcds, scene_pcds=scene_pcds)
            else:
                self.visualize(est_cam_poses, mask_pcds, scene_pcds=scene_pcds, gt_poses=cam_poses)
            self.visualize_volume(tsdf_mesh)

        if combined_pcd is None:
            return self.combine_clouds(mask_pcds)
        else:
            return combined_pcd

    def load_data(self, seq_name, frame_id):
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)

        # Create a cloud from the rgb and depth images
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_o3d, depth_o3d)
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        K = anno['camMat']
        cam_intrinsics.set_intrinsics(640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        scene_pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, cam_intrinsics)
        scene_pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points) * 1000)
        # scene_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return rgb, depth, anno, scene_pcd

    def read_camera_intrinsics(self, seq_name):
        cam_params = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        cam_mat = np.zeros((3, 3))
        cam_mat[2, 2] = 1.

        f_name = os.path.join(self.base_dir, self.data_split, seq_name, 'meta/0000.pkl')
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations pickle file at %s. Aborting.' % f_name)
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)

        cam_mat[0, 0] = pickle_data['camMat'][0, 0]
        cam_mat[1, 1] = pickle_data['camMat'][1, 1]
        cam_mat[0, 2] = pickle_data['camMat'][0, 2]
        cam_mat[1, 2] = pickle_data['camMat'][1, 2]

        cam_params.set_intrinsics(640, 480, cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2])

        return cam_params

    def image_to_world(self, mask=None, cut_z=1000.):
        i_fx = 1. / self.anno['camMat'][0, 0]
        i_fy = 1. / self.anno['camMat'][1, 1]
        cx = self.anno['camMat'][0, 2]
        cy = self.anno['camMat'][1, 2]

        pts = []
        colors = []
        for v in range(self.rgb.shape[0]):
            for u in range(self.rgb.shape[1]):
                if mask is None or mask[v, u] > 0:
                    z = self.depth[v, u]
                    if z > 0.001 and z < cut_z:
                        x = (u - cx) * z * i_fx
                        y = (v - cy) * z * i_fy
                        pts.append([x, y, z])
                        colors.append(self.rgb[v, u])

        pts = np.asarray(pts)
        colors = np.asarray(colors) / 255

        return pts, colors

    def apply_mask(self, image, mask):
        # Get the indices that are not in the mask
        valid_idx = np.where(mask.flatten() == 0)[0]

        # Set the invalid indices to 0
        masked_image = np.copy(image).flatten()
        for v in valid_idx:
            masked_image[v] = 0  # [0, 0, 0]

        # Reshape back to input shape
        masked_image = masked_image.reshape(image.shape)

        #masked_image = copy.deepcopy(image)
        #for u in range(mask.shape[0]):
        #    for v in range(mask.shape[1]):
        #        if mask_to_apply[u, v] == 0:
        #            masked_image[u, v] = 0

        return masked_image

    def transform_to_object_frame(self, cloud):
        # Get the ground truth transformation
        obj_trans = np.copy(self.anno['objTrans'])
        obj_trans = obj_trans.dot(COORD_CHANGE_MAT.T)
        obj_rot = np.copy(self.anno['objRot'])
        obj_rot = obj_rot.flatten().dot(COORD_CHANGE_MAT.T).reshape(self.anno['objRot'].shape)
        rot_max = cv2.Rodrigues(obj_rot)[0].T

        cloud_tfd = np.copy(cloud)
        cloud_tfd -= obj_trans
        cloud_tfd = np.matmul(cloud_tfd, np.linalg.inv(rot_max))

        cam_pose = np.eye(4)
        cam_pose[:3, :3] = rot_max
        cam_pose[:3, 3] = np.matmul(-obj_trans, np.linalg.inv(rot_max))

        if self.args.reg_method == RegMethod.GT or self.prev_cloud is None:
            return cloud_tfd, cam_pose, cloud_tfd, cam_pose
        else:
            # Down sample
            source_cloud = o3d.geometry.PointCloud()
            source_cloud.points = o3d.utility.Vector3dVector(cloud)
            if self.args.voxel_size > 0:
                source_cloud = o3d.geometry.voxel_down_sample(source_cloud, self.args.voxel_size)
            # Perform registration
            est_cam_pose = np.copy(self.register_new_cloud(source_cloud, cam_pose))
            # Transform the cloud
            est_cloud_tfd = np.copy(cloud)
            est_cloud_trans = np.matmul(-est_cam_pose[0:3, 3], est_cam_pose[0:3, 0:3])
            est_cloud_tfd -= est_cloud_trans
            est_cloud_tfd = np.matmul(est_cloud_tfd, np.linalg.inv(est_cam_pose[0:3, 0:3]))
            euler_gt = tf3d.euler.mat2euler(cam_pose[:3, :3])
            trans_gt = cam_pose[:3, 3]
            euler_est = tf3d.euler.mat2euler(est_cam_pose[:3, :3])
            trans_est = est_cam_pose[:3, 3]
            print('X {:.4f}\tY {:.4f}\tZ {:.4f}\tRx {:.4f}\tRy {:.4f}\tRz {:.4f}'.format(
                abs(trans_gt[0] - trans_est[0]) * 100, abs(trans_gt[1] - trans_est[1]) * 100,
                abs(trans_gt[2] - trans_est[2]) * 100,
                np.degrees(abs(euler_gt[0] - euler_est[0])), np.degrees(abs(euler_gt[1] - euler_est[1])),
                np.degrees(abs(euler_gt[2] - euler_est[2]))))

            return cloud_tfd, cam_pose, est_cloud_tfd, est_cam_pose

    def register_new_cloud(self, cloud, initial_transform):
        return self.align_icp(cloud, initial_transform)

    def align_icp(self, cloud, trans_init):
        # Get the threshold
        threshold = ICP_THRESH

        # Perform ICP
        if self.args.icp_method == ICPMethod.Point2Point:
            reg_p2p = o3d.registration.registration_icp(cloud, self.prev_cloud, threshold, trans_init,
                                                        o3d.registration.TransformationEstimationPointToPoint())
        elif self.args.icp_method == ICPMethod.Point2Plane:
            if not cloud.has_normals():
                radius_normal = self.args.voxel_size * 2
                o3d.geometry.estimate_normals(
                    cloud,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            if not self.prev_cloud.has_normals():
                radius_normal = self.args.voxel_size * 2
                o3d.geometry.estimate_normals(
                    self.prev_cloud,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            reg_p2p = o3d.registration.registration_icp(cloud, self.prev_cloud, threshold, trans_init,
                                                        o3d.registration.TransformationEstimationPointToPlane())
        else:
            print('Unrecognized ICP method {}'.format(self.args.icp_method))
            return None

        return reg_p2p.transformation

    @staticmethod
    def visualize(cam_poses=None, mask_pcds=None, scene_pcds=None, gt_poses=None):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        # vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.5))

        # Plot the object cloud
        for m in mask_pcds:
            vis.add_geometry(m)

        # Plot scene
        if scene_pcds is not None:
            for s in scene_pcds:
                vis.add_geometry(s)

        # Plot camera pose
        if cam_poses is not None:
            for m in cam_poses:
                points = m[:3, :].T
                points[0:3, :] *= 0.175
                points[0:3, :] += np.tile(points[3, :], (3, 1))
                lines = [[3, 0], [3, 1], [3, 2]]
                line_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                vis.add_geometry(line_set)

        # Plot the ground truth camera poses
        if gt_poses is not None:
            for m in gt_poses:
                points = m[:3, :].T
                points[0:3, :] *= 0.1
                points[0:3, :] += np.tile(points[3, :], (3, 1))
                lines = [[3, 0], [3, 1], [3, 2]]
                line_colors = [[.6, .4, .4], [.4, .6, .4], [.4, .4, .6]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                vis.add_geometry(line_set)

        # End visualizer
        vis.run()
        vis.destroy_window()

    def visualize_volume(self, mesh):
        # Visualize
        if mesh is not None:
            o3d.visualization.draw_geometries([mesh])

    def save_clouds_and_camera_poses(self, base_dir, frame_ids, cam_poses, est_cam_poses, mask_pcds, mesh, frame_id=0):
        # Combine and downsample the clouds
        down_pcd = self.combine_clouds(mask_pcds)

        # Extract the points and normals
        points = np.asarray(down_pcd.points)
        normals = np.asarray(down_pcd.normals)

        # Create the filename and write the data
        filename = os.path.join(base_dir, str(self.args.reg_method).split('.')[1])
        if self.args.reg_method == RegMethod.GT_ICP or self.args.reg_method == RegMethod.GT_ICP_FULL:
            filename += '_' + str(self.args.icp_method).split('.')[1]
        filename += '_start' + str(self.args.start_frame) + '_max' + str(self.args.max_num) +\
                    '_skip' + str(self.args.skip)
        if int(frame_id) > 0:
            filename += '_frame' + str(frame_id)
        filename += '.xyz'
        f = open(filename, "w")
        for i in range(len(points)):
            f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
                                                 normals[i, 0], normals[i, 1], normals[i, 2]))
        f.close()

        # Write at .ply
        o3d.io.write_point_cloud(filename.replace('.xyz', '.ply'), down_pcd)

        # Write the mesh as .ply
        o3d.io.write_triangle_mesh(filename.replace('.xyz', '_tsdf.ply'), mesh)

        '''
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(down_pcd)
        vis.run()
        vis.destroy_window()
        '''

        return down_pcd

    @staticmethod
    def combine_clouds(mask_pcds):
        all_points = np.asarray(mask_pcds[0].points)
        all_colors = np.asarray(mask_pcds[0].colors)
        for i in range(1, len(mask_pcds)):
            all_points = np.vstack((all_points, np.asarray(mask_pcds[i].points)))
            all_colors = np.vstack((all_colors, np.asarray(mask_pcds[i].colors)))

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        down_pcd = o3d.geometry.voxel_down_sample(combined_pcd, voxel_size=0.001)

        o3d.geometry.estimate_normals(down_pcd,
                                      search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                          radius=0.1, max_nn=30))

        return down_pcd

    @staticmethod
    def load_object_model(filename):
        print('Loading file {}'.format(filename))
        pcd = o3d.geometry.PointCloud()
        pts = np.loadtxt(filename)[:, :3]
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd


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

        # Save
        if self.args.save:
            filename = self.args.model_file.replace('.ply', '_poisson.ply')
            o3d.io.write_triangle_mesh(filename, mesh)

        # Visualize
        if self.args.visualize:
            # Load ycbv model
            BOP_FORMAT = True
            if BOP_FORMAT:
                mesh_ycb = o3d.io.read_triangle_mesh('/home/tpatten/Data/bop/ycbv/models_eval/obj_000012.ply')
                offset_bop = np.asarray([-0.00499, 0.0024, 0.01329])
                mesh.translate(-offset_bop)
                mesh.scale(1000, center=(0, 0, 0))
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
        # First sample points
        pcd = mesh_in.sample_points_poisson_disk(number_of_points=2500, init_factor=5)
        # o3d.visualization.draw_geometries([pcd])

        if r_method == ReconstructionMethod.POISSON:
            # Try to remove outliers
            if self.args.clean_up_outlier_removal:
                pcd = remove_outliers(pcd,
                                      outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                      outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
                                      raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                                      raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
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


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Object model reconstruction')
    parser.add_argument("scene", type=str, help="Sequence of the dataset")
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/Data/Hands/HO3D/'
    args.model_file = ''
    # args.model_file = '/home/tpatten/Data/Hands/HO3D/train/ABF10/GT_start0_max-1_skip1.xyz'
    # args.scene = 'ABF10'
    args.visualize = False
    args.save = False
    args.save_intermediate = False
    args.icp_method = ICPMethod.Point2Plane
    # Point2Point=1, Point2Plane=2
    args.reg_method = RegMethod.GT
    # GT=1, GT_ICP=2, GT_ICP_FULL=3
    args.start_frame = 0
    args.max_num = -1
    args.skip = 1
    args.mask_erosion_kernel = 5
    args.outlier_rm_nb_neighbors = 2500  # Higher is more aggressive
    args.outlier_rm_std_ratio = 0.000001  # Smaller is more aggressive
    args.raduis_rm_min_nb_points = 250  # Don't change
    args.raduis_rm_radius_factor = 10  # Don't change
    args.voxel_size = 0.001
    args.sdf_voxel_length = 2.0 / 512.0
    args.sdf_trunc = 0.02
    args.sdf_depth_trunc = 1.2
    args.construct_tsdf = True

    # Create the reconstruction
    reconstructor = ModelReconstructor(args)

    '''
    # Clean up the mesh
    args.model_file = '/home/tpatten/Data/Hands/HO3D/train/ABF10/GT_start0_max-1_skip1_tsdf.ply'
    args.visualize = True
    args.save = False
    args.outlier_rm_nb_neighbors = 50  # Higher is more aggressive
    args.outlier_rm_std_ratio = 0.01  # Smaller is more aggressive
    args.raduis_rm_min_nb_points = 0
    args.raduis_rm_radius_factor = 0
    args.clean_up_outlier_removal = True
    psr = PoissonSurfaceReconstructor(args)
    '''
