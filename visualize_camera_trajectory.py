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


class TrajectoryVisualizer:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.rgb = None
        self.depth = None
        self.anno = None
        self.scene_pcd = None

        self.visualize_trajectory()

    def visualize_trajectory(self):
        # Get the scene ids
        frame_ids = os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb'))

        # For each frame in the directory
        cam_poses = []
        mask_pcds = []
        scene_pcds = []
        count = 0
        base_cam_pose = np.eye(4)
        for fid in frame_ids:
            # Get the id
            frame_id = fid.split('.')[0]
            if frame_id == '0001':
                frame_id = '0500'
            # Create the filename for the metadata file
            meta_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'meta', str(frame_id) + '.pkl')
            print('Processing file {}'.format(meta_filename))

            # Read image, depths maps and annotations
            self.rgb, self.depth, self.anno, scene_pcd = self.load_data(self.args.scene, frame_id)

            # Read the mask
            mask_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'mask',
                                         OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png')
            mask = cv2.imread(mask_filename)

            # Extract the masked point cloud
            obj_rot = self.anno['objRot']
            obj_trans = self.anno['objTrans']
            cloud, colors = self.image_to_world(mask, cut_z=np.linalg.norm(obj_trans)*1.1)
            mask_pcd = o3d.geometry.PointCloud()
            mask_pcd.points = o3d.utility.Vector3dVector(cloud)
            mask_pcd.colors = o3d.utility.Vector3dVector(colors)
            # mask_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            obj_trans = obj_trans.dot(coord_change_mat.T)
            obj_rot = obj_rot.flatten().dot(coord_change_mat.T).reshape(self.anno['objRot'].shape)
            rot_max = cv2.Rodrigues(obj_rot)[0].T

            pts = np.asarray(mask_pcd.points)
            pts -= obj_trans
            pts = np.matmul(pts, np.linalg.inv(rot_max))
            mask_pcd.points = o3d.utility.Vector3dVector(pts)

            cam_pose = np.eye(4)
            cam_pose[:3, :3] = rot_max
            cam_pose[:3, 3] = np.matmul(-obj_trans, np.linalg.inv(rot_max))
            print(cam_pose)

            # Visualize
            # self.visualize([cam_pose], [mask_pcd], [scene_pcd])
            # sys.exit(0)

            cam_poses.append(cam_pose)
            mask_pcds.append(mask_pcd)
            scene_pcds.append(scene_pcd)

            break
            count += 1
            if count == 2:
                break

        # Visualize
        # scene_pcds = None
        self.visualize(cam_poses, mask_pcds, scene_pcds)

    def load_data(self, seq_name, frame_id):
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)

        # Create a cloud for visualization
        # rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
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
                if mask is None or mask[v, u][0] > 0:
                    z = self.depth[v, u]
                    if z < cut_z:
                        x = (u - cx) * z * i_fx
                        y = (v - cy) * z * i_fy
                        pts.append([x, y, z])
                        colors.append(self.rgb[v, u])

        pts = np.asarray(pts)
        colors = np.asarray(colors) / 255

        return pts, colors

    def visualize(self, cam_poses, mask_pcds, scene_pcds=None):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.5))

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
            points[0, :] += points[3, :]  # TODO tile this
            points[1, :] += points[3, :]
            points[2, :] += points[3, :]
            lines = [[3, 0], [3, 1], [3, 2]]
            line_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Camera trajectory visualization')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.scene = 'ABF10'
    args.visualize = True
    args.save = False

    mask_extractor = TrajectoryVisualizer(args)
