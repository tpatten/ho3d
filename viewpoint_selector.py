# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import cv2
import open3d as o3d
import json
import numpy as np
from enum import IntEnum


OBJECT_MASK_VISIBLE_DIR = 'object_vis'
COORD_CHANGE_MAT = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)


class ViewScore(IntEnum):
    Rendering = 1
    Segmentation = 2


class SelectionType(IntEnum):
    Uniform = 1
    Greedy = 2


class ViewpointSelector:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'

        # Load the scores file
        print('Loading viewpoint scores...')
        if self.args.score_type == ViewScore.Rendering:
            scene_scores = self.load_rendering_scores()
        else:
            scene_scores = self.load_segmentation_mask_counts()

        # Load data
        viewpoints = self.load_data()

        # Cluster the data
        views_to_grid_map = self.cluster_views(viewpoints)

        # Select views
        selected_views = self.select_viewpoints(views_to_grid_map, scene_scores)
        print('Reduced frames from {} to {}'.format(len(viewpoints), len(selected_views)))

        # Visualize views
        if self.args.visualize:
            self.visualize_views(viewpoints, view_subset=selected_views)

        # Save the views
        if self.args.save:
            # Create the filename and write the data
            filename = os.path.join(self.base_dir, self.data_split, self.args.scene,
                                    'views_' + str(self.args.selection_type).split('.')[1] +
                                    '_' + str(self.args.score_type).split('.')[1])
            filename += '_step' + str(self.args.view_step_size).replace('.', '-')
            filename += '.json'
            print('Saving to {}'.format(filename))
            save_data = {'frame_ids': selected_views}
            with open(filename, 'w') as file:
                json.dump(save_data, file, indent=2)

    def load_data(self):
        data = {}
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb')))
        for fid in range(len(frame_ids)):
            frame_id = frame_ids[fid].split('.')[0]
            if fid % 50 == 0:
                print('--- {}'.format(frame_id))
            anno = read_annotation(self.base_dir, self.args.scene, frame_id, self.data_split)
            pose = self.transform_to_object_frame(anno)
            position_sphere = self.point_to_sphere(pose[:3, 3])
            sphere_pose = np.eye(4)
            sphere_pose[:3, :3] = pose[:3, :3]
            sphere_pose[:3, 3] = position_sphere
            data[str(int(frame_id))] = {'pose': pose, 'sphere': sphere_pose}

        return data

    def load_rendering_scores(self):
        if os.path.isfile(self.args.hand_obj_rendering_scores):
            with open(self.args.hand_obj_rendering_scores) as json_file:
                hand_obj_rendering_scores = json.load(json_file)
            sub_dirs = sorted([f for f in os.listdir(os.path.join(self.base_dir, self.data_split))
                               if os.path.isdir(os.path.join(self.base_dir, self.data_split, f))])
            bop_scene_id = str(sub_dirs.index(self.args.scene) + 1)
            print('BOP ID is {} from HO-3D ID {}'.format(bop_scene_id, self.args.scene))
            scene_rendering_scores = hand_obj_rendering_scores[bop_scene_id]
        else:
            scene_rendering_scores = None

        return scene_rendering_scores

    def load_segmentation_mask_counts(self):
        segmentation_mask_counts = {}
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb')))
        for fid in range(len(frame_ids)):
            frame_id = frame_ids[fid].split('.')[0]
            if fid % 50 == 0:
                print('--- {}'.format(frame_id))
            segmentation_mask_counts[str(int(frame_id))] = {'seg_pixel_count': self.count_mask_pixels(frame_id)}

        return segmentation_mask_counts

    def count_mask_pixels(self, frame_id):
        # Load the mask file
        mask_filename = os.path.join(self.args.mask_dir, self.data_split, self.args.scene, 'seg',
                                     str(frame_id) + '.jpg')
        if not os.path.exists(mask_filename):
            return 0
        mask_rgb = cv2.imread(mask_filename)

        # Generate binary object mask
        count = 0
        for u in range(mask_rgb.shape[0]):
            for v in range(mask_rgb.shape[1]):
                if mask_rgb[u, v, 0] > 230 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] < 10:
                    count += 1

        return count

    @staticmethod
    def transform_to_object_frame(viewpoints):
        # Get the ground truth transformation
        obj_trans = np.copy(viewpoints['objTrans'])
        obj_trans = obj_trans.dot(COORD_CHANGE_MAT.T)
        obj_rot = np.copy(viewpoints['objRot'])
        obj_rot = obj_rot.flatten().dot(COORD_CHANGE_MAT.T).reshape(viewpoints['objRot'].shape)
        rot_max = cv2.Rodrigues(obj_rot)[0].T

        cam_pose = np.eye(4)
        cam_pose[:3, :3] = rot_max
        cam_pose[:3, 3] = np.matmul(-obj_trans, np.linalg.inv(rot_max))

        return cam_pose

    @staticmethod
    def point_to_sphere(point, radius=1):
        p_length = np.linalg.norm(point)
        new_point = point * radius / p_length
        return new_point

    def cluster_views(self, viewpoints):
        # Get bounds of the voxel grid
        view_limits = [[1000, 1000, 1000], [-1000, -1000, -1000]]
        frame_keys = sorted(viewpoints.keys())
        for frame_id in frame_keys:
            position = viewpoints[frame_id]['sphere'][:3, 3]
            if position[0] < view_limits[0][0]:
                view_limits[0][0] = position[0]
            if position[1] < view_limits[0][1]:
                view_limits[0][1] = position[1]
            if position[2] < view_limits[0][2]:
                view_limits[0][2] = position[2]
            if position[0] > view_limits[1][0]:
                view_limits[1][0] = position[0]
            if position[1] > view_limits[1][1]:
                view_limits[1][1] = position[1]
            if position[2] > view_limits[1][2]:
                view_limits[1][2] = position[2]

        # Create the grid between the limits
        x = np.linspace(view_limits[0][0], view_limits[1][0],
                        int((view_limits[1][0] - view_limits[0][0]) / self.args.view_step_size))
        y = np.linspace(view_limits[0][1], view_limits[1][1],
                        int((view_limits[1][1] - view_limits[0][1]) / self.args.view_step_size))
        z = np.linspace(view_limits[0][2], view_limits[1][2],
                        int((view_limits[1][2] - view_limits[0][2]) / self.args.view_step_size))
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        grid = np.zeros((x.shape[0], 3))
        views_to_grid_map = {}
        for i in range(x.shape[0]):
            grid[i, :] = [x[i], y[i], z[i]]
            views_to_grid_map[i] = []

        # Associate each view to a grid cell
        for frame_id in frame_keys:
            position = viewpoints[frame_id]['sphere'][:3, 3]
            dist = np.linalg.norm(position - grid, axis=-1)
            grid_key = np.argmin(dist)
            views_to_grid_map[grid_key].append(frame_id)

        return views_to_grid_map

    def select_viewpoints(self, views_to_grid_map, scene_scores):
        # Iterate through each grid cell and select the view in it with the highest rendering score
        best_views = []
        for grid_cell in views_to_grid_map.keys():
            if len(views_to_grid_map[grid_cell]) > 0:
                best_score = 0
                best_frame = 0
                for frame_id in views_to_grid_map[grid_cell]:
                    if self.args.score_type == ViewScore.Rendering:
                        view_score = (scene_scores[frame_id]['objs_scores'][0][3] +
                                      scene_scores[frame_id]['hand_scores'][3] +
                                      scene_scores[frame_id]['scores'][3]) / 3.0
                    else:
                        view_score = scene_scores[frame_id]['seg_pixel_count']
                    if view_score > best_score:
                        best_score = view_score
                        best_frame = frame_id
                # Add this view
                best_views.append(best_frame)
        return best_views

    @staticmethod
    def visualize_views(viewpoints, view_subset=None):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Sphere
        mesh_sphere = o3d.create_mesh_sphere(radius=0.1)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        vis.add_geometry(mesh_sphere)

        # Get the subset of views
        if view_subset is not None:
            frame_keys = view_subset
        else:
            frame_keys = viewpoints.keys()

        # Plot camera poses
        for frame_id in frame_keys:
            m = viewpoints[frame_id]['pose']
            points = m[:3, :].T
            points[0:3, :] *= 0.1
            points[0:3, :] += np.tile(points[3, :], (3, 1))
            lines = [[3, 0], [3, 1], [3, 2]]
            line_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            vis.add_geometry(line_set)

        # End visualizer
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Viewpoint selector')
    parser.add_argument("scene", type=str, help="Sequence of the dataset")
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2'
    args.mask_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2_segmentations_rendered/'
    args.hand_obj_rendering_scores = '/home/tpatten/Data/bop/ho3d/hand_obj_ren_scores.json'
    args.visualize = False
    args.save = True
    args.min_num_pixels = 8000
    args.view_step_size = 0.4
    args.selection_type = SelectionType.Uniform  # 1: SelectionType.Uniform, 2: SelectionType.Greedy
    args.score_type = ViewScore.Rendering  # 1: ViewScore.Rendering, 2: ViewScore.Segmentation

    # Create viewpoint selector
    view_selector = ViewpointSelector(args)
