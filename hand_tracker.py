from __future__ import division
from utils.grasp_utils import *
import os
import cv2
import time
import numpy as np
import argparse
import open3d as o3d
import copy
from enum import IntEnum


POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]
NUM_POINTS = 21
THRESHOLD = 0.5


class JointShapes(IntEnum):
    SPHERE = 1
    BOX = 2


def to_iccv_format(joints):
    # MONOHAND [Wrist (0),
    #           TMCP (1), TPIP (2), TDIP (3), TTIP (4),
    #           IMCP (5), IPIP (6), IDIP (7), ITIP (8),
    #           MMCP (9), MPIP (10), MDIP (11), MTIP (12),
    #           RMCP (13), RPIP (14), RDIP (15), RTIP (16),
    #           PMCP (17), PPIP (18), PDIP (19), PTIP (20)]

    # ICCV     [Wrist,
    #           TMCP, IMCP, MMCP, RMCP, PMCP,
    #           TPIP, TDIP, TTIP,
    #           IPIP, IDIP, ITIP,
    #           MPIP, MDIP, MTIP,
    #           RPIP, RDIP, RTIP,
    #           PPIP, PDIP, PTIP]

    # joint_map = [0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

    # HO3D     [Wrist,
    #           IMCP, IPIP, IDIP,
    #           MMCP, MPIP, MDIP,
    #           PMCP, PPIP, PDIP
    #           RMCP, RPIP, RDIP,
    #           TMCP, TPIP, TDIP,
    #           TTIP, ITIP, MTIP, RTIP, PTIP]

    joint_map = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

    iccv_joints = np.zeros(joints.shape)
    for i in range(len(joints)):
        iccv_joints[i, :] = joints[joint_map[i], :]

    return iccv_joints


class HandTracker:
    def __init__(self, args):
        self.width = 640
        self.height = 480
        self.net = cv2.dnn.readNetFromCaffe(args.proto_file, args.weights_file)
        self.object_id = ''
        self.base_pcd = None
        self.pcd = None
        self.joints_gt = None
        self.cam_mat = None

        rgb_path = os.path.join(args.ho3d_path, 'train', args.target, 'rgb')
        depth_path = os.path.join(args.ho3d_path, 'train', args.target, 'depth')
        meta_path = os.path.join(args.ho3d_path, 'train', args.target, 'meta')

        if args.save:
            save_path = os.path.join(args.ho3d_path, 'train', args.target, 'hand')
            if not os.path.exists(save_path):
                try:
                    os.makedirs(save_path)
                except OSError:
                    print('ERROR: Unable to create the save directory {}'.format(save_path))
                    return

        file_list = sorted(os.listdir(rgb_path))
        for im in file_list:
            print(im)
            self.rgb_image, self.dep_image = self.get_images(os.path.join(rgb_path, im), os.path.join(depth_path, im))
            meta_filename = os.path.join(meta_path, im)
            meta_filename = meta_filename.replace(".png", ".pkl")
            anno = read_hand_annotations(meta_filename)
            self.joints_gt = anno['handJoints3D']
            obj_rot = anno['objRot']
            obj_trans = anno['objTrans']
            obj_id = anno['objName']
            self.cam_mat = anno['camMat']

            # Load the object point cloud
            if obj_id != self.object_id:
                self.object_id = obj_id
                self.base_pcd = o3d.geometry.PointCloud()
                self.base_pcd.points = o3d.utility.Vector3dVector(
                    np.loadtxt(os.path.join(args.models_path, obj_id, 'points.xyz')))
            # Rotate and translate the cloud
            self.pcd = copy.deepcopy(self.base_pcd)
            pts = np.asarray(self.pcd.points)
            pts = np.matmul(pts, cv2.Rodrigues(obj_rot)[0].T)
            pts += obj_trans
            self.pcd.points = o3d.utility.Vector3dVector(pts)
            self.pcd.paint_uniform_color([0.5, 0.5, 0.5])

            # Estimate the hand joint positions
            points_2d, self.joints_estimated = self.process()

            # Visualize
            if args.visualize:
                self.draw_image(points_2d)
                self.visualize_3D()

            if args.save:
                save_filename = os.path.join(save_path, im)
                save_filename = save_filename.replace(".png", ".pkl")
                with open(save_filename, 'wb') as f:
                    save_data = {}
                    save_data['handJoints3D'] = self.joints_estimated
                    pickle.dump(save_data, f)

    def process(self):
        self.width = self.rgb_image.shape[1]
        self.height = self.rgb_image.shape[0]
        aspect_ratio = self.width / self.height

        t = time.time()

        # Input image dimensions for the network
        in_height = 368
        in_width = int(((aspect_ratio*in_height)*8)//8)
        net_input = cv2.dnn.blobFromImage(self.rgb_image, 1.0 / 255, (in_width, in_height), (0, 0, 0),
                                          swapRB=False, crop=False)

        self.net.setInput(net_input)

        pred = self.net.forward()
        print("Time taken by network : {:.2f} secs".format(time.time() - t))

        points_2d = self.get_keypoints(pred)

        points_3d = self.get_world_coordinates(points_2d)
        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        points_3d = points_3d.dot(coord_change_mat.T)
        points_3d = to_iccv_format(points_3d)

        return points_2d, points_3d

    @staticmethod
    def get_images(rgb_filename, dep_filename):
        rgb_image = cv2.imread(rgb_filename)
        dep_image = cv2.imread(dep_filename)
        depth_scale = 0.00012498664727900177
        dep_image = dep_image[:, :, 2] + dep_image[:, :, 1] * 256
        dep_image = dep_image * depth_scale

        return rgb_image, dep_image

    def get_keypoints(self, pred):
        # Empty list to store the detected keypoints
        points = []

        for i in range(NUM_POINTS):
            # confidence map of corresponding body's part.
            prob_map = pred[0, i, :, :]
            prob_map = cv2.resize(prob_map, (self.width, self.height))

            # Find global maxima of the probability map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > THRESHOLD:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)
        
        return points

    def get_world_coordinates(self, points):
        ux = self.cam_mat[0][2]
        uy = self.cam_mat[1][2]
        i_fx = 1 / self.cam_mat[0][0]
        i_fy = 1 / self.cam_mat[1][1]

        points_3d = np.zeros((len(points), 3))
        for i in range(len(points)):
            if points[i] is not None:
                points_3d[i, 2] = self.dep_image[points[i][1], points[i][0]]
                points_3d[i, 0] = (points[i][0] - ux) * points_3d[i, 2] * i_fx
                points_3d[i, 1] = (points[i][1] - uy) * points_3d[i, 2] * i_fy
            else:
                # points_3d[i, :] = 0.
                points_3d[i, :] = None

        return points_3d

    def draw_image(self, points):
        # Draw the keypoints
        rgb_keypoints = np.copy(self.rgb_image)
        for i in range(len(points)):
            if points[i] is not None:
                cv2.circle(rgb_keypoints, (int(points[i][0]), int(points[i][1])), 8, (0, 255, 255),
                           thickness=-1, lineType=cv2.FILLED)
                cv2.putText(rgb_keypoints, "{}".format(i), (int(points[i][0]), int(points[i][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Draw Skeleton
        rgb_skeleton = np.copy(self.rgb_image)
        for pair in POSE_PAIRS:
            part_a = pair[0]
            part_b = pair[1]

            if points[part_a] and points[part_b]:
                cv2.line(rgb_skeleton, points[part_a], points[part_b], (0, 255, 255), 2)
                cv2.circle(rgb_skeleton, points[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(rgb_skeleton, points[part_b], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.imshow('Output-Keypoints', rgb_keypoints)
        cv2.imshow('Output-Skeleton', rgb_skeleton)

        # cv2.imwrite('Output-Keypoints.jpg', rgb_keypoints)
        # cv2.imwrite('Output-Skeleton.jpg', rgb_skeleton)

        cv2.waitKey(0)

    def visualize_3D(self):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Object cloud
        vis.add_geometry(self.pcd)

        # Ground truth hand joints
        self.visualize_hand(vis, self.joints_gt, shape=JointShapes.SPHERE)

        # Estimated hand joints
        self.visualize_hand(vis, self.joints_estimated, shape=JointShapes.BOX)

        # Close visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_hand(vis, joints, shape=JointShapes.SPHERE):
        # Set the colors for the fingers depending on the shape
        if shape == JointShapes.SPHERE:
            shape_finger_colors = finger_colors
        else:
            shape_finger_colors = finger_colors_box

        # Plot the finger joints
        for i in range(len(all_indices)):
            for j in range(len(all_indices[i])):
                if shape == JointShapes.SPHERE:
                    mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                else:
                    mm = o3d.create_mesh_box(width=joint_sizes[j], height=joint_sizes[j], depth=joint_sizes[j])
                mm.compute_vertex_normals()
                mm.paint_uniform_color(shape_finger_colors[i])
                trans3d = joints[all_indices[i][j]]
                tt = np.eye(4)
                tt[0:3, 3] = trans3d
                mm.transform(tt)
                vis.add_geometry(mm)

        # Plot lines between joints
        lines = [[0, 13], [0, 1], [0, 4], [0, 10], [0, 7],
                 [13, 14], [14, 15], [15, 16],
                 [1, 2], [2, 3], [3, 17],
                 [4, 5], [5, 6], [6, 18],
                 [10, 11], [11, 12], [12, 19],
                 [7, 8], [8, 9], [9, 20]]
        line_colors = [
            shape_finger_colors[1], shape_finger_colors[2], shape_finger_colors[3],
            shape_finger_colors[4], shape_finger_colors[5],
            shape_finger_colors[1], shape_finger_colors[1], shape_finger_colors[1],
            shape_finger_colors[2], shape_finger_colors[2], shape_finger_colors[2],
            shape_finger_colors[3], shape_finger_colors[3], shape_finger_colors[3],
            shape_finger_colors[4], shape_finger_colors[4], shape_finger_colors[4],
            shape_finger_colors[5], shape_finger_colors[5], shape_finger_colors[5]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(joints)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        vis.add_geometry(line_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand tracker using OpenCV')
    parser.add_argument("-target", type=str, help="Name of the target subset",
                        choices=['ABF10', 'BB10', 'GPMF10', 'GSF10', 'MDF10', 'ShSu10'], default='ABF10')
    args = parser.parse_args()
    args.proto_file = 'caffe_models/pose_deploy.prototxt'
    args.weights_file = 'caffe_models/pose_iter_102000.caffemodel'
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    # args.target = 'BB10'
    args.visualize = False
    args.save = True

    print(args)

    hand_tracker = HandTracker(args)

