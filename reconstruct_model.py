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
    ICP_PAIR = 3
    ICP_FULL = 4
    FPHF_ICP_PAIR = 5
    FPFH_ICP_FULL = 6
    FASTGLOB_ICP_PAIR = 7
    FASTGLOB_ICP_FULL = 8


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

        # Reconstruct the model
        loaded_pcd = None
        if self.args.model_file == '':
            loaded_pcd = self.reconstruct_object_model()

        if self.args.visualize:
            if not self.args.model_file == '':
                loaded_pcd = self.load_object_model(self.args.model_file)

            #loaded_pcd = self.remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
            #                                  outlier_rm_std_ratio=self.args.outlier_rm_std_ratio * 0.0001,
            #                                  raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
            #                                  raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
            #loaded_pcd = self.remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
            #                                  raduis_rm_min_nb_points=1250,
            #                                  raduis_rm_radius=self.args.voxel_size * 10)
            self.visualize(mask_pcds=[loaded_pcd])
            # self.visualize(mask_pcds=[self.load_object_model(self.args.model_file)])

            if self.args.save:
                loaded_pcd = o3d.geometry.voxel_down_sample(loaded_pcd, 0.001)
                o3d.geometry.estimate_normals(loaded_pcd,
                                              search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                  radius=0.1, max_nn=30))
                points = np.asarray(loaded_pcd.points)
                normals = np.asarray(loaded_pcd.normals)
                filename = self.args.model_file.split('.')[0] + '_clean.xyz'
                f = open(filename, "w")
                for i in range(len(points)):
                    f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
                                                         normals[i, 0], normals[i, 1], normals[i, 2]))
                f.close()

    def reconstruct_object_model(self):
        # Get the scene ids
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb')))

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
            #mask_pcd = self.remove_outliers(mask_pcd,
            #                                outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
            #                                outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
            #                                raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
            #                                raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
            mask_pcd = self.remove_outliers(mask_pcd,
                                            outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                            outlier_rm_std_ratio=self.args.outlier_rm_std_ratio)

            # Add to lists
            cam_poses.append(cam_pose)
            est_cam_poses.append(est_cam_pose)
            mask_pcds.append(mask_pcd)
            scene_pcds.append(scene_pcd)
            processed_frames.append(frame_id)

            # Increment counters
            counter += self.args.skip
            num_processed += 1

            # Update the previous cloud and pose
            if self.args.reg_method == RegMethod.ICP_FULL or self.args.reg_method == RegMethod.FPFH_ICP_FULL or \
                    self.args.reg_method == RegMethod.FASTGLOB_ICP_FULL:
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

        # Save
        combined_pcd = None
        if self.args.save:
            base_dir = os.path.join(self.base_dir, self.data_split, self.args.scene)
            combined_pcd = self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses,
                                                             mask_pcds)

        # Visualize
        if self.args.visualize:
            scene_pcds = None
            if self.args.reg_method == RegMethod.GT:
                self.visualize(cam_poses, mask_pcds, scene_pcds=scene_pcds)
            else:
                self.visualize(est_cam_poses, mask_pcds, scene_pcds=scene_pcds, gt_poses=cam_poses)

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
            est_cam_pose = np.copy(self.register_new_cloud(source_cloud,
                                                           self.get_initial_transform(source_cloud, cam_pose)))
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

            '''
            # Debug: Visualize
            print('--')
            print('{:.4f}\t{:.4f}\t{:.4f}'.format(obj_trans[0], obj_trans[1], obj_trans[2]))
            print('{:.4f}\t{:.4f}\t{:.4f}'.format(est_cloud_trans[0], est_cloud_trans[1], est_cloud_trans[2]))
            print('{:.4f}\t{:.4f}\t{:.4f}'.format(est_cloud_trans[0] - obj_trans[0], est_cloud_trans[1] - obj_trans[1],
                                                  est_cloud_trans[2] - obj_trans[2]))
            print('--')

            vis = o3d.visualization.Visualizer()
            vis.create_window()

            previous_pcd = copy.deepcopy(self.prev_cloud)
            previous_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            vis.add_geometry(previous_pcd)

            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(cloud_tfd)
            gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(gt_pcd)

            est_pcd = o3d.geometry.PointCloud()
            est_pcd.points = o3d.utility.Vector3dVector(est_cloud_tfd)
            est_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            vis.add_geometry(est_pcd)

            vis.run()
            vis.destroy_window()
            # Debug: End
            '''

            return cloud_tfd, cam_pose, est_cloud_tfd, est_cam_pose

    def get_initial_transform(self, cloud, cam_pose):
        initial_transform = np.eye(4)
        if self.args.reg_method == RegMethod.GT_ICP:
            initial_transform = cam_pose
        elif self.args.reg_method == RegMethod.ICP_PAIR or self.args.reg_method == RegMethod.ICP_FULL:
            initial_transform = self.prev_pose
        elif self.args.reg_method == RegMethod.FPHF_ICP_PAIR or self.args.reg_method == RegMethod.FPFH_ICP_FULL:
            initial_transform = self.fpfh_registration(cloud)
        elif self.args.reg_method == RegMethod.FASTGLOB_ICP_PAIR or self.args.reg_method == RegMethod.FASTGLOB_ICP_FULL:
            initial_transform = self.fast_global_registration(cloud)
        else:
            print('Unrecognized registration method {}\nInitial transform is identity matrix'.format(
                self.args.reg_method))
        return initial_transform

    def register_new_cloud(self, cloud, initial_transform):
        return self.align_icp(cloud, initial_transform)

    def align_icp(self, cloud, trans_init):
        # Get the threshold
        threshold = ICP_THRESH
        #if self.args.voxel_size > 0:
        #    threshold = self.args.voxel_size * 0.4

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

    def fpfh_registration(self, cloud):
        # Compute the FPFH features
        fpfh_cloud = self.compute_fpfh(cloud)
        fpfh_prev_cloud = self.compute_fpfh(self.prev_cloud)

        # Get the distance threshold for RANSAC
        distance_threshold = ICP_THRESH  # RANSAC_THRESH
        #if self.args.voxel_size > 0:
        #    distance_threshold = self.args.voxel_size * 1.5

        # Align the clouds with RANSAC
        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            cloud, self.prev_cloud, fpfh_cloud, fpfh_prev_cloud, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))

        return result_ransac.transformation

    def compute_fpfh(self, cloud):
        radius_normal = self.args.voxel_size * 2
        o3d.geometry.estimate_normals(
            cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.args.voxel_size * 5
        fpfh = o3d.registration.compute_fpfh_feature(
            cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return fpfh

    def fast_global_registration(self, cloud):
        # Compute the FPFH features
        fpfh_cloud = self.compute_fpfh(cloud)
        fpfh_prev_cloud = self.compute_fpfh(self.prev_cloud)

        # Get the distance threshold for RANSAC
        distance_threshold = ICP_THRESH  # RANSAC_THRESH
        # if self.args.voxel_size > 0:
        #    distance_threshold = self.args.voxel_size * 0.5

        result_registration = o3d.registration.registration_fast_based_on_feature_matching(
            cloud, self.prev_cloud, fpfh_cloud, fpfh_prev_cloud,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))

        return result_registration.transformation

    @staticmethod
    def remove_outliers(cloud, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                        raduis_rm_min_nb_points=0, raduis_rm_radius=0.):
        # Copy the input cloud
        in_cloud = copy.deepcopy(cloud)

        # Statistical outlier removal
        if outlier_rm_nb_neighbors > 0 and outlier_rm_std_ratio > 0:
            in_cloud, _ = o3d.geometry.statistical_outlier_removal(
                in_cloud, nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)

        # Radius outlier removal
        if raduis_rm_min_nb_points > 0 and raduis_rm_radius > 0:
            in_cloud, _ = o3d.geometry.radius_outlier_removal(
                in_cloud, nb_points=raduis_rm_min_nb_points, radius=raduis_rm_radius)

        return in_cloud

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

    def save_clouds_and_camera_poses(self, base_dir, frame_ids, cam_poses, est_cam_poses, mask_pcds, frame_id=0):
        '''
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
        '''

        down_pcd = self.combine_clouds(mask_pcds)

        points = np.asarray(down_pcd.points)
        normals = np.asarray(down_pcd.normals)
        filename = os.path.join(base_dir, str(self.args.reg_method).split('.')[1] +
                                '_start' + str(self.args.start_frame) +
                                '_max' + str(self.args.max_num) +
                                '_skip' + str(self.args.skip))
        if int(frame_id) > 0:
            filename = filename + '_frame' + str(frame_id)
        filename = filename + '.xyz'
        f = open(filename, "w")
        for i in range(len(points)):
            f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
                                                 normals[i, 0], normals[i, 1], normals[i, 2]))
        f.close()

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


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Object model reconstruction')
    args = parser.parse_args()
    # args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.model_file = ''  # '/home/tpatten/Data/Hands/HO3D/train/BB10/GT_start0_max500_skip2.xyz'
    #args.model_file = '/home/tpatten/Data/Hands/HO3D/train/GPMF10/GT_start0_max500_skip2.xyz'
    args.ho3d_path = '/home/tpatten/Data/Hands/HO3D/'
    args.scene = 'GPMF10'
    args.visualize = True
    args.save = False
    args.save_intermediate = False
    args.icp_method = ICPMethod.Point2Plane
    # Point2Point=1, Point2Plane=2
    args.reg_method = RegMethod.GT_ICP
    # GT=1, GT_ICP=2, ICP_PAIR=3, ICP_FULL=4, FPHF_ICP_PAIR=5, FPFH_ICP_FULL=6, FASTGLOB_ICP_PAIR=7, FASTGLOB_ICP_FULL=8
    args.start_frame = 0
    args.max_num = 15
    args.skip = 16
    args.mask_erosion_kernel = 5
    args.outlier_rm_nb_neighbors = 500
    args.outlier_rm_std_ratio = 0.001
    args.raduis_rm_min_nb_points = 250
    args.raduis_rm_radius_factor = 5
    args.voxel_size = 0.001

    # Visualize the trajectory
    mask_extractor = ModelReconstructor(args)
