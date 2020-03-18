# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import copy

HAND_MASK_DIR = 'hand'
OBJECT_MASK_DIR = 'object'
HAND_MASK_VISIBLE_DIR = 'hand_vis'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'
COORD_CHANGE_MAT = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)


class TrajectoryVisualizer:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.rgb = None
        self.depth = None
        self.anno = None
        self.scene_pcd = None
        if args.mask_erosion_kernel > 0:
            self.erosion_kernel = np.ones((args.mask_erosion_kernel, args.mask_erosion_kernel), np.uint8)
        else:
            self.erosion_kernel = None

        # Compute and visualize
        self.visualize_trajectory()

    def visualize_trajectory(self):
        # Get the scene ids
        frame_ids = os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb'))

        # For each frame in the directory
        cam_poses = []
        mask_pcds = []
        scene_pcds = []
        processed_frames = []
        counter = 0
        num_processed = 0
        while counter < len(frame_ids):
            # Get the id
            frame_id = frame_ids[counter].split('.')[0]
            # Create the filename for the metadata file
            meta_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'meta', str(frame_id) + '.pkl')
            print('Processing file {}'.format(meta_filename))

            # Read image, depths maps and annotations
            self.rgb, self.depth, self.anno, scene_pcd = self.load_data(self.args.scene, frame_id)

            # Read the mask
            mask_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'mask',
                                         OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png')
            mask = cv2.imread(mask_filename)[:, :, 0]
            if self.erosion_kernel is not None:
                mask = cv2.erode(mask, self.erosion_kernel, iterations=1)

            # Extract the masked point cloud
            cloud, colors = self.image_to_world(mask, cut_z=np.linalg.norm(self.anno['objTrans'])*1.1)

            # Transform the cloud and get the camera position
            cloud, cam_pose = self.transform_to_object_frame(cloud)

            # Create the point cloud for visualization
            mask_pcd = o3d.geometry.PointCloud()
            mask_pcd.points = o3d.utility.Vector3dVector(cloud)
            mask_pcd.colors = o3d.utility.Vector3dVector(colors)
            # mask_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Remove outliers
            mask_pcd = self.remove_outliers(mask_pcd)

            # Add to lists
            cam_poses.append(cam_pose)
            mask_pcds.append(mask_pcd)
            scene_pcds.append(scene_pcd)
            processed_frames.append(frame_id)

            # Increment counters
            counter += self.args.skip
            num_processed += 1

            # Exit if reached the limit
            if self.args.max_num > 0 and num_processed == self.args.max_num:
                break

        # Save
        if self.args.save:
            base_dir = os.path.join(self.base_dir, self.data_split, self.args.scene)
            self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, mask_pcds)

        # Visualize
        if self.args.visualize:
            scene_pcds = None
            self.visualize(cam_poses, mask_pcds, scene_pcds)

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
                    if z < cut_z:
                        x = (u - cx) * z * i_fx
                        y = (v - cy) * z * i_fy
                        pts.append([x, y, z])
                        colors.append(self.rgb[v, u])

        pts = np.asarray(pts)
        colors = np.asarray(colors) / 255

        return pts, colors

    def transform_to_object_frame(self, cloud):
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

        return cloud_tfd, cam_pose

    def remove_outliers(self, cloud):
        if self.args.outlier_rm_nb_neighbors > 0 and self.args.outlier_rm_std_ratio > 0:
            _, ind = o3d.geometry.statistical_outlier_removal(cloud,
                                                              nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                                              std_ratio=self.args.outlier_rm_std_ratio)
            in_cloud = o3d.geometry.select_down_sample(cloud, ind)
            return in_cloud
        else:
            return cloud

    @staticmethod
    def visualize(cam_poses, mask_pcds, scene_pcds=None):
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
        for m in cam_poses:
            points = m[:3, :].T
            points[0:3, :] *= 0.25
            points[0:3, :] += np.tile(points[3, :], (3, 1))
            lines = [[3, 0], [3, 1], [3, 2]]
            line_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()

    @staticmethod
    def save_clouds_and_camera_poses(base_dir, frame_ids, cam_poses, mask_pcds):
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

        points = np.asarray(down_pcd.points)
        normals = np.asarray(down_pcd.normals)
        filename = os.path.join(base_dir, 'reconstruction.xyz')
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


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Camera trajectory visualization')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.scene = 'ABF10'
    args.visualize = True
    args.save = True
    args.max_num = -1
    args.skip = 100
    args.mask_erosion_kernel = 5
    args.outlier_rm_nb_neighbors = 500
    args.outlier_rm_std_ratio = 0.001

    # Visualize the trajectory
    mask_extractor = TrajectoryVisualizer(args)
