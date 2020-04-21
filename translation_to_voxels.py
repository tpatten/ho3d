# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import numpy as np
import open3d as o3d
from math import ceil

SUBJECTS = ['ABF', 'BB', 'GPMF', 'GSF', 'MDF', 'ShSu']
DIAMETER = 0.232280153674483
PASSGREEN = lambda x: '\033[92m' + x + '\033[0m'
FAILRED = lambda x: '\033[91m' + x + '\033[0m'


class VoxelVisualizer:
    def __init__(self, args):
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.res = args.resolution
        self.axis_symmetry = args.axis_symmetry
        self.save = args.save
        self.verbose = args.verbose
        self.do_visualize = args.visualize

        gripper_pcd = o3d.io.read_point_cloud(args.gripper_cloud_path)
        self.gripper_xyz = np.asarray(gripper_pcd.points).T

        # Compute and visualize
        self.visualize_voxels()

    def visualize_voxels(self):
        # For each subject
        sub_dirs = os.listdir(os.path.join(self.base_dir, self.data_split))
        valid_sub_dirs = []
        for s in sub_dirs:
            counter = 0
            for c in s:
                if not c.isalpha():
                    break
                else:
                    counter += 1
            subject_name = s[0:counter]
            if subject_name in SUBJECTS:
                valid_sub_dirs.append(s)
        sub_dirs = valid_sub_dirs

        max_lims = np.zeros((3, 1))
        for s in sub_dirs:
            # Get the scene ids
            frame_ids = os.listdir(os.path.join(self.base_dir, self.data_split, s, 'rgb'))

            # For each frame in the directory
            for f in frame_ids:
                # Get the id
                frame_id = f.split('.')[0]
                # Create the filename for the metadata file
                meta_filename = os.path.join(self.base_dir, self.data_split, s, 'meta', str(frame_id) + '.pkl')
                print('Processing file {}'.format(meta_filename))

                # Read image, depths maps and annotations
                anno, grasp = self.load_data(s, frame_id)
                hand_joints = anno['handJoints3D']
                grasp_position = grasp[:3, 3].flatten()
                # grasp_position = None

                voxels = self.compute_voxels(hand_joints, grasp_position=grasp_position)
                x_lim = abs(voxels[0][0])
                y_lim = abs(voxels[1][0])
                z_lim = abs(voxels[2][0])
                if x_lim > max_lims[0]:
                    max_lims[0] = x_lim
                if y_lim > max_lims[1]:
                    max_lims[1] = y_lim
                if z_lim > max_lims[2]:
                    max_lims[2] = z_lim

                # Visualize
                if self.do_visualize:
                    self.visualize(voxels, hand_joints, grasp_pose=grasp)

        # Generate the final grid
        n_steps = np.abs(np.ceil(max_lims / self.res))
        for i in range(n_steps.shape[0]):
            if n_steps[i] == 0:
                n_steps[i] = 1
        x, y, z = np.meshgrid(np.arange(-n_steps[0] * self.res, n_steps[0] * self.res, self.res),
                              np.arange(-n_steps[1] * self.res, n_steps[1] * self.res, self.res),
                              np.arange(-n_steps[2] * self.res, n_steps[2] * self.res, self.res))

        grid3 = np.asarray(list(zip(x.flatten(), y.flatten(), z.flatten())))
        print('Axis limits:\t{:.4f}\t{:.4f}\t{:.4f}'.format(max_lims[0], max_lims[1], max_lims[2]))
        print('Number of elements:\t{}'.format(grid3.shape[0]))
        ## Save
        #if self.save:
        #    # base_dir = os.path.join(self.base_dir, self.data_split, self.args.scene)
        #    # self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, mask_pcds)

    def load_data(self, seq_name, frame_id):
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)

        # Get the gripper pose
        gripper_pose_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'meta',
                                             'grasp_bl_' + frame_id + '.pkl')
        with open(gripper_pose_filename, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)
        grasp_pose = pickle_data.reshape(4, 4)

        return anno, grasp_pose

    def compute_voxels(self, hand_joints, grasp_position=None):
        offset = np.expand_dims(np.mean(hand_joints, axis=0), 0)
        if grasp_position is None:
            hand_joints = hand_joints - offset
            dist = np.max(np.sqrt(np.sum(hand_joints ** 2, axis=1)), 0)
            lims = np.max(hand_joints, axis=0)
        else:
            grasp_position = grasp_position - offset
            dist = np.sqrt(np.sum(grasp_position ** 2))
            lims = grasp_position
        lims = lims.reshape(3, 1)

        if self.axis_symmetry:
            n_steps = abs(ceil(dist / self.res))
            x, y, z = np.meshgrid(np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                                  np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                                  np.arange(-n_steps * self.res, n_steps * self.res, self.res))
        else:
            n_steps = np.abs(np.ceil(lims / self.res))
            for i in range(n_steps.shape[0]):
                if n_steps[i] == 0:
                    n_steps[i] = 1
            x, y, z = np.meshgrid(np.arange(-n_steps[0] * self.res, n_steps[0] * self.res, self.res),
                                  np.arange(-n_steps[1] * self.res, n_steps[1] * self.res, self.res),
                                  np.arange(-n_steps[2] * self.res, n_steps[2] * self.res, self.res))
        grid3 = np.asarray(list(zip(x.flatten(), y.flatten(), z.flatten())))

        if self.verbose and grasp_position is not None:
            dists = np.linalg.norm(grid3 - grasp_position, axis=1)
            min_idx = np.argmin(dists)
            print('Min dist {:.0f}mm'.format(dists[min_idx]*1000))

        if self.verbose:
            print('Max dist {:.0f}mm'.format(dist*1000))
            print('Num voxels {}'.format(grid3.shape[0]))
        return grid3

    def visualize(self, voxels, hand_joints, grasp_pose=None):
        offset = np.expand_dims(np.mean(hand_joints, axis=0), 0)

        nearest_voxel = None
        if grasp_pose is not None:
            dists = np.linalg.norm(voxels - (grasp_pose[:3, 3].flatten() - offset), axis=1)
            min_idx = np.argmin(dists)
            print('Translation error {:.0f}mm'.format(dists[min_idx] * 1000))
            nearest_voxel = voxels[min_idx]

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Add the hand joints
        for i in range(len(hand_joints)):
            mm = o3d.create_mesh_sphere(radius=0.005)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([0, 0, 1])
            tt = np.eye(4)
            tt[0:3, 3] = hand_joints[i] - offset
            mm.transform(tt)
            vis.add_geometry(mm)

        # Add the voxels
        for v in voxels:
            mm = o3d.create_mesh_box(width=0.1*self.res, height=0.1*self.res, depth=0.1*self.res)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([0.5, 0.5, 0.5])
            tt = np.eye(4)
            tt[0:3, 3] = v
            mm.transform(tt)
            vis.add_geometry(mm)
        if nearest_voxel is not None:
            mm = o3d.create_mesh_box(width=0.005, height=0.005, depth=0.005)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([0, 1, 0])
            tt = np.eye(4)
            tt[0:3, 3] = nearest_voxel
            mm.transform(tt)
            vis.add_geometry(mm)

        # Add the grasp position
        if grasp_pose is not None:
            mm = o3d.create_mesh_box(width=0.005, height=0.005, depth=0.005)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([1, 0, 0])
            tt = np.eye(4)
            tt[0:3, 3] = grasp_pose[:3, 3].flatten() - offset
            mm.transform(tt)
            vis.add_geometry(mm)

            gripper_gt = np.copy(self.gripper_xyz)
            gripper_gt = np.matmul(grasp_pose[:3, :3], gripper_gt)
            gripper_gt[0, :] += (grasp_pose[0, 3] - offset[0][0])
            gripper_gt[1, :] += (grasp_pose[1, 3] - offset[0][1])
            gripper_gt[2, :] += (grasp_pose[2, 3] - offset[0][2])
            gripper_pcd = o3d.geometry.PointCloud()
            gripper_pcd.points = o3d.utility.Vector3dVector(gripper_gt.T)
            gripper_pcd.paint_uniform_color([0.9, 0.4, 0.4])
            vis.add_geometry(gripper_pcd)

            discretised_grasp_pose = np.copy(grasp_pose)
            discretised_grasp_pose[:3, 3] = nearest_voxel
            gripper_approx = np.copy(self.gripper_xyz)
            gripper_approx = np.matmul(grasp_pose[:3, :3], gripper_approx)
            gripper_approx[0, :] += discretised_grasp_pose[0, 3]
            gripper_approx[1, :] += discretised_grasp_pose[1, 3]
            gripper_approx[2, :] += discretised_grasp_pose[2, 3]
            gripper_pcd_approx = o3d.geometry.PointCloud()
            gripper_pcd_approx.points = o3d.utility.Vector3dVector(gripper_approx.T)
            gripper_pcd_approx.paint_uniform_color([0.4, 0.9, 0.4])
            vis.add_geometry(gripper_pcd_approx)

            add_error = np.linalg.norm(gripper_gt - gripper_approx, axis=0).mean()
            print('ADD error {:.4f}'.format(add_error))
            if add_error < 0.1*DIAMETER:
                print('%s' % (PASSGREEN('PASS')))
            else:
                print('%s' % (FAILRED('FAIL')))

        # Run the visualizer
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Translation voxel visualization')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.gripper_cloud_path = 'hand_open_new.pcd'
    args.resolution = 0.025
    args.axis_symmetry = True
    args.visualize = False
    args.save = False
    args.verbose = False

    # Visualize the voxels
    voxel_visualizer = VoxelVisualizer(args)

