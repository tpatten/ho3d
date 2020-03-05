# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

from utils.grasp_utils import *
import cv2
import argparse
import open3d as o3d
import copy
from os.path import join, exists


MODEL_PATH = '/v4rtemp/datasets/HandTracking/HO3D_v2/models/'


class SceneAligner:
    def __init__(self, cmd_args):
        self.args = cmd_args

    def compute_transform(self):
        source_pcd, target_pcd = self.load_clouds()

        tf = self.align_icp(source_pcd, target_pcd)

        if self.args.visualize:
            self.visualize(source_pcd, target_pcd, tf)

        return tf

    def load_clouds(self):
        clouds = []
        for i in range(2):
            # Get name
            if i == 0:
                in_name = self.args.source
            else:
                in_name = self.args.target
            # Get meta data
            meta_filename = join(self.args.ho3d_path, 'train', in_name, 'meta/0000.pkl')
            anno = read_hand_annotations(meta_filename)
            obj_rot = anno['objRot']
            obj_trans = anno['objTrans']
            obj_id = anno['objName']
            # Get cloud and transform
            obj_pcd = self.get_object_pcd(obj_id)
            pts = np.asarray(obj_pcd.points)
            pts = np.matmul(pts, cv2.Rodrigues(obj_rot)[0].T)
            pts += obj_trans
            # Now add the hand mesh
            hand_cloud_filename = join(self.args.ho3d_path, 'train', in_name, 'meta/hand_mesh_0000.ply')
            hand_cloud = o3d.io.read_point_cloud(hand_cloud_filename)
            pts = np.vstack((pts, hand_cloud.points))
            # Final cloud
            obj_pcd.points = o3d.utility.Vector3dVector(pts)
            clouds.append(obj_pcd)

        return clouds[0], clouds[1]

    def get_object_pcd(self, object_name):
        # Get the point cloud
        model_path = self.args.models_path
        if model_path == MODEL_PATH:
            obj_cloud_filename = join(self.args.models_path, object_name, 'cloud.ply')
            pcd = o3d.io.read_point_cloud(obj_cloud_filename)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 0.001)
        else:
            obj_cloud_filename = join(self.args.models_path, object_name, 'points.xyz')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.loadtxt(obj_cloud_filename))

        # Return the point cloud
        return pcd

    def align_icp(self, source_pcd, target_pcd):
        save_filename = join(self.args.ho3d_path, 'train', self.args.source, 'meta/transformation_icp.pkl')
        if exists(save_filename):
            print('Loading existing transformation file\n{}'.format(save_filename))
            transform = load_pickle_data(save_filename)
            return transform

        threshold = 0.02
        trans_file_name = join(self.args.ho3d_path, 'train', self.args.source, 'meta/transformation_init.pkl')
        trans_init = load_pickle_data(trans_file_name)

        print("Initial transformation is:")
        print(trans_init)
        source_pcd_init = copy.deepcopy(source_pcd)
        source_pcd_init.transform(trans_init)

        print("Initial alignment")
        evaluation = o3d.registration.evaluate_registration(source_pcd, target_pcd, threshold, trans_init)
        print(evaluation)

        print("Apply point-to-point ICP")
        reg_p2p = o3d.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")

        with open(save_filename, 'wb') as f:
            pickle.dump(reg_p2p.transformation, f)

        return reg_p2p.transformation

    def visualize(self, source_pcd, target_pcd, transform):
        # Get aligned cloud
        source_pcd_temp = copy.deepcopy(source_pcd)
        source_pcd_temp.transform(transform)

        # Load the gripper
        gripper_pcd_target = o3d.io.read_point_cloud(self.args.gripper_cloud_path)
        gripper_pcd_source = copy.deepcopy(gripper_pcd_target)
        gripper_transform_file = join(self.args.ho3d_path, 'train', self.args.target, 'meta/grasp_bl_0000.pkl')
        gripper_transform = load_pickle_data(gripper_transform_file)
        gripper_transform = gripper_transform.reshape(4, 4)
        gripper_pcd_target.transform(gripper_transform)

        tt = np.matmul(np.linalg.inv(transform), gripper_transform)
        gripper_pcd_source.transform(tt)
        gripper_pcd_source.paint_uniform_color([0.9, 0.1, 0.1])

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Plot the clouds
        source_pcd.paint_uniform_color([0.8, 0.3, 0.3])
        vis.add_geometry(source_pcd)
        gripper_pcd_source.paint_uniform_color([0.9, 0.1, 0.1])
        vis.add_geometry(gripper_pcd_source)

        target_pcd.paint_uniform_color([0.4, 0.4, 0.4])
        vis.add_geometry(target_pcd)
        gripper_pcd_target.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(gripper_pcd_target)

        source_pcd_temp.paint_uniform_color([0.3, 0.3, 0.8])
        vis.add_geometry(source_pcd_temp)

        vis.run()
        vis.destroy_window()


def transform_grasps_to_target(args, transform):
    # Load all grasps in source
    ids = os.listdir(join(args.ho3d_path, 'train', args.source, 'rgb'))
    for i in ids:
        id = i.split('.')[0]
        # Create the filename for the metadata file
        target_grasp_filename = os.path.join(args.ho3d_path, 'train', args.target, 'meta',
                                             'grasp_bl_' + str(id) + '.pkl')
        print('Processing file {}'.format(target_grasp_filename))

        gripper_transform = load_pickle_data(target_grasp_filename)
        gripper_transform = gripper_transform.reshape(4, 4)
        tt = np.matmul(np.linalg.inv(transform), gripper_transform)
        source_grasp_filename = os.path.join(args.ho3d_path, 'train', args.source, 'meta',
                                             'grasp_bl_' + str(id) + '.pkl')
        with open(source_grasp_filename, 'wb') as f:
            pickle.dump(tt, f)


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Transfer grasps')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    args.gripper_cloud_path = 'hand_open_new.pcd'
    args.target = 'ABF10'
    args.source = 'ABF14'
    args.visualize = False
    args.process_grasps = True

    aligner = SceneAligner(args)
    transform = aligner.compute_transform()
    if args.process_grasps:
        transform_grasps_to_target(args, transform)
