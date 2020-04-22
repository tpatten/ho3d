# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import numpy as np
import open3d as o3d
from math import ceil
import transforms3d as tf3d
from scipy.spatial.distance import cityblock
import collections

SUBJECTS = ['ABF', 'BB', 'GPMF', 'GSF', 'MDF', 'ShSu']
DIAMETER = 0.232280153674483
PASSGREEN = lambda x: '\033[92m' + x + '\033[0m'
FAILRED = lambda x: '\033[91m' + x + '\033[0m'
SAVE_FILENAME = 'translation_classification_counts'
SAVE_FILENAME_AUG = 'translation_classification_counts_augmented'


class VoxelGenerator:
    def __init__(self, args):
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.res = args.resolution
        self.axis_symmetry = args.axis_symmetry
        self.augmentation = args.augmentation
        self.save = args.save
        self.verbose = args.verbose
        self.do_visualize = args.visualize
        self.hard_limits = args.hard_limits

        gripper_pcd = o3d.io.read_point_cloud(args.gripper_cloud_path)
        self.gripper_xyz = np.asarray(gripper_pcd.points).T

        # Generate and visualize
        self.generate_voxels()

    def generate_voxels(self):
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
        counts = {}
        counts[1] = []
        counts[8] = []
        counts[27] = []
        counts[64] = []
        counts[125] = []
        counts[216] = []
        counts[343] = []
        counts[512] = []
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

                voxels, voxel_center = self.compute_voxels(hand_joints, grasp_position=grasp_position)
                counts[voxels.shape[0]].append((s + '/' + str(frame_id), voxel_center))

                x_lim = abs(voxels[0][0])
                y_lim = abs(voxels[0][1])
                z_lim = abs(voxels[0][2])
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
        max_lims = max_lims.flatten()
        print('Final axis limits: {:.4f}  {:.4f}  {:.4f}'.format(max_lims[0], max_lims[1], max_lims[2]))
        print('Number of elements: {}'.format(grid3.shape[0]))
        for c in counts:
            print('{}:\t{}'.format(c, len(counts[c])))

        # Save
        if self.save:
            if self.augmentation:
                filename = os.path.join(self.base_dir, SAVE_FILENAME_AUG +
                                        '_R' + str(self.res).replace('.', '-') + '.txt')
            else:
                filename = os.path.join(self.base_dir, SAVE_FILENAME + '_R' + str(self.res).replace('.', '-') + '.txt')
            print('Saving to {}'.format(filename))
            f = open(filename, "w")
            for c in counts:
                for s in counts[c]:
                    f.write("{} {:.4f} {:.4f} {:.4f} {}\n".format(s[0], s[1][0], s[1][1], s[1][2], c))
            f.close()

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
        hand_joints = hand_joints - offset
        grasp_position = grasp_position - offset

        if self.augmentation:
            # print('Before: {}'.format(grasp_position))
            hand_joints, grasp_position = self.augment_data(hand_joints, grasp_position)
            # print('After:  {}'.format(grasp_position))

        if grasp_position is None:
            dist = np.max(np.sqrt(np.sum(hand_joints ** 2, axis=1)), 0)
            lims = np.max(hand_joints, axis=0)
        else:
            dist = np.sqrt(np.sum(grasp_position ** 2))
            lims = grasp_position
        lims = lims.reshape(3, 1)
        if self.hard_limits is not None:
            lims = np.asarray(self.hard_limits).reshape(3, 1)

        if self.hard_limits is not None or not self.axis_symmetry:
            n_steps = np.abs(np.ceil(lims / self.res))
            for i in range(n_steps.shape[0]):
                if n_steps[i] == 0:
                    n_steps[i] = 1
            x, y, z = np.meshgrid(np.arange(-n_steps[0] * self.res, n_steps[0] * self.res, self.res),
                                  np.arange(-n_steps[1] * self.res, n_steps[1] * self.res, self.res),
                                  np.arange(-n_steps[2] * self.res, n_steps[2] * self.res, self.res))
        else:
            n_steps = abs(ceil(dist / self.res))
            x, y, z = np.meshgrid(np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                                  np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                                  np.arange(-n_steps * self.res, n_steps * self.res, self.res))

        grid3 = np.asarray(list(zip(x.flatten(), y.flatten(), z.flatten())))

        nearest_voxel = [1000, 1000, 1000]
        if grasp_position is not None:
            # dists = np.linalg.norm(grid3 - (grasp_position - offset), axis=1)
            dists = np.linalg.norm(grid3 - grasp_position, axis=1)
            min_idx = np.argmin(dists)
            nearest_voxel = grid3[min_idx]

        if self.verbose:
            if grasp_position is None:
                print('Max dist {:.0f}mm'.format(dist * 1000))
            else:
                dists = np.linalg.norm(grid3 - grasp_position, axis=1)
                min_idx = np.argmin(dists)
                print('Min dist {:.0f}mm'.format(dists[min_idx]*1000))
                print('Vox dist {:.0f}mm'.format(np.linalg.norm(nearest_voxel - grasp_position.flatten()) * 1000))
                gripper_gt = self.get_transformed_gripper_pcd(grasp_position.flatten(), np.eye(3))
                gripper_approx = self.get_transformed_gripper_pcd(nearest_voxel, np.eye(3))
                add_error = np.linalg.norm(gripper_gt - gripper_approx, axis=0).mean()
                print('ADD error {:.4f} / {:.2f}%'.format(add_error, add_error / DIAMETER * 100))
                if add_error < 0.1 * DIAMETER:
                    print('%s' % (PASSGREEN('PASS')))
                else:
                    print('%s' % (FAILRED('FAIL')))

            print('Num voxels {}'.format(grid3.shape[0]))
            print('Axis lims: {:.4f}  {:.4f}  {:.4f}'.format(abs(grid3[0][0]), abs(grid3[0][1]), abs(grid3[0][2])))

        return grid3, nearest_voxel

    @staticmethod
    def augment_data(point_set, target, disable_global=False):
        # Global rotation
        rotation_matrix = tf3d.euler.euler2mat(np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi),
                                               np.random.uniform(-np.pi, np.pi))
        point_set_augmented = np.copy(point_set)
        if not disable_global:
            point_set_augmented = np.matmul(point_set_augmented, rotation_matrix)

        target_augmented = None
        if target is not None:
            target_augmented = np.copy(target)
            # target_augmented = target_augmented.reshape((3, 3))
            if not disable_global:
                target_augmented = np.matmul(target_augmented, rotation_matrix)

        # target_augmented = target_augmented.reshape((1, 9)).flatten()

        return point_set_augmented, target_augmented

    def get_transformed_gripper_pcd(self, translation, rotation):
        gripper_tf = np.copy(self.gripper_xyz)
        gripper_tf = np.matmul(rotation, gripper_tf)
        gripper_tf[0, :] += translation[0]
        gripper_tf[1, :] += translation[1]
        gripper_tf[2, :] += translation[2]

        return gripper_tf

    def visualize(self, voxels, hand_joints, grasp_pose=None):
        offset = np.expand_dims(np.mean(hand_joints, axis=0), 0)

        nearest_voxel = None
        if grasp_pose is not None:
            dists = np.linalg.norm(voxels - (grasp_pose[:3, 3].flatten() - offset), axis=1)
            min_idx = np.argmin(dists)
            # print('Translation error {:.0f}mm'.format(dists[min_idx] * 1000))
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

            gripper_gt = self.get_transformed_gripper_pcd((grasp_pose[:3, 3] - offset).flatten(), grasp_pose[:3, :3])
            gripper_pcd = o3d.geometry.PointCloud()
            gripper_pcd.points = o3d.utility.Vector3dVector(gripper_gt.T)
            gripper_pcd.paint_uniform_color([0.9, 0.4, 0.4])
            vis.add_geometry(gripper_pcd)

            gripper_approx = self.get_transformed_gripper_pcd(nearest_voxel, grasp_pose[:3, :3])
            gripper_pcd_approx = o3d.geometry.PointCloud()
            gripper_pcd_approx.points = o3d.utility.Vector3dVector(gripper_approx.T)
            gripper_pcd_approx.paint_uniform_color([0.4, 0.9, 0.4])
            vis.add_geometry(gripper_pcd_approx)

        # Run the visualizer
        vis.run()
        vis.destroy_window()


class VoxelAnalyzer:
    def __init__(self, args):
        self.base_dir = args.ho3d_path
        self.res = args.resolution
        self.augmentation = args.augmentation

        counts, grid3, counts_per_grid_cell = self.analyze()

        if args.visualize:
            self.visualize(grid3, counts_per_grid_cell)

    def analyze(self):
        # Set up structures to store data
        counts = {}
        counts[1] = []
        counts[8] = []
        counts[27] = []
        counts[64] = []
        counts[125] = []
        counts[216] = []
        counts[343] = []
        counts[512] = []

        n_steps = 4
        x, y, z = np.meshgrid(np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                              np.arange(-n_steps * self.res, n_steps * self.res, self.res),
                              np.arange(-n_steps * self.res, n_steps * self.res, self.res))
        grid3 = np.asarray(list(zip(x.flatten(), y.flatten(), z.flatten())))
        counts_per_grid_cell = np.zeros((grid3.shape[0], 1))

        # Load the file
        if self.augmentation:
            data_file = os.path.join(self.base_dir, SAVE_FILENAME_AUG + '_R' + str(self.res).replace('.', '-') + '.txt')
        else:
            data_file = os.path.join(self.base_dir, SAVE_FILENAME + '_R' + str(self.res).replace('.', '-') + '.txt')
        f = open(data_file, "r")
        for line in f:
            fname, vx, vy, vz, count_group = line.split(' ')
            vx = float(vx)
            vy = float(vy)
            vz = float(vz)
            count_group = int(count_group.rstrip())
            counts[count_group].append(fname)
            dists = np.linalg.norm(grid3 - np.asarray([vx, vy, vz]), axis=1)
            min_idx = np.argmin(dists)
            counts_per_grid_cell[min_idx] += 1
        f.close()

        counts_per_grid_cell = counts_per_grid_cell.flatten()

        for c in counts:
            print('{}:\t{}'.format(c, len(counts[c])))

        dist_counts = {}
        num_used_cells = 0
        num_empty_cells = 0
        lims = np.zeros((2, 3))
        for i in range(grid3.shape[0]):
            if counts_per_grid_cell[i] > 0:
                for j in range(3):
                    if grid3[i][j] > lims[1][j]:
                        lims[1][j] = grid3[i][j]
                    elif grid3[i][j] < lims[0][j]:
                        lims[0][j] = grid3[i][j]
                mdist = cityblock(np.asarray([0, 0, 0]), grid3[i])
                if dist_counts.get(mdist) is None:
                    dist_counts[mdist] = 0
                dist_counts[mdist] += counts_per_grid_cell[i]
                num_used_cells += 1
            else:
                num_empty_cells += 1
        print('Number of used grid cells: {}'.format(num_used_cells))
        print('Number of empty grid cells: {}'.format(num_empty_cells))

        lims = np.ceil(lims / self.res).astype(int)
        print('Restructured grid {} x {} x {} = {}'.format(
            lims[1][0] - lims[0][0] + 1, lims[1][1] - lims[0][1] + 1, lims[1][2] - lims[0][2] + 1,
            (lims[1][0] - lims[0][0] + 1) * (lims[1][1] - lims[0][1] + 1) * (lims[1][2] - lims[0][2] + 1)))

        od = collections.OrderedDict(sorted(dist_counts.items()))
        for k, v in od.items():
            print('{:.3f} : {}\t\t{}'.format(k, int(v), int(k / self.res)))

        return counts, grid3, counts_per_grid_cell

    def visualize(self, voxels, counts):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Add the voxels as a point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(voxels)
        cloud.paint_uniform_color([1, 0, 0])
        colors = np.asarray(cloud.colors)
        for i in range(voxels.shape[0]):
            if counts[i] == 0:
                colors[i] = [0.7, 0.7, 0.7]
        cloud.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(cloud)

        # Add the voxels
        box_dim = 0.1 * self.res
        for i in range(voxels.shape[0]):
            if counts[i] > 0:
                mm = o3d.create_mesh_box(width=box_dim, height=box_dim, depth=box_dim)
                mm.compute_vertex_normals()
                mm.paint_uniform_color([1, 0, 0])
                tt = np.eye(4)
                tt[0:3, 3] = voxels[i] - 0.5 * box_dim
                mm.transform(tt)
                vis.add_geometry(mm)

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
    args.augmentation = False
    args.axis_symmetry = True
    args.visualize = False
    args.save = True
    args.verbose = False
    args.hard_limits = None
    # args.hard_limits = []
    args.analyze_mode = False
    if args.analyze_mode:
        args.ho3d_path = '/home/tpatten/'

    if not args.analyze_mode:
        # Visualize the voxels
        voxel_gen = VoxelGenerator(args)
    else:
        # Analyze the voxels
        voxel_analyzer = VoxelAnalyzer(args)
