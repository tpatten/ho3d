# author: Shreyas Hampali
# contact: hampali@icg.tugraz.at
# modifier for HANDS19: Anil Armagan
# contact: aarmagan@ic.ac.uk
# modifier for robot grasping: Tim Patten
# contact: patten@acin.tuwien.ac.at

"""
xhost +
docker container ls -a
docker container start [container-id]
docker exec -it [container-id] bash
export PATH="/usr/lib/nvidia-418/bin":${PATH}
export LD_LIBRARY_PATH="/usr/lib/nvidia-418:/usr/lib32/nvidia-418":${LD_LIBRARY_PATH}
"""

from utils.grasp_utils import *
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
import cv2
import argparse
import transforms3d as tf3d
import open3d as o3d
import copy
from os.path import join, exists


MODEL_PATH = '/v4rtemp/datasets/HandTracking/HO3D_v2/models/'


class RobotGraspAnnotator:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.gripper_pcd = o3d.io.read_point_cloud(self.args.gripper_cloud_path)
        self.object_id = ''
        self.base_pcd = None
        self.pcd = None
        self.kd_tree = None
        self.do_visualize = args.visualize
        self.do_save = args.save

        # Extract the annotated points on the gripper to determine the transformation
        self.left_tip = self.gripper_pcd.points[LEFT_TIP_IN_CLOUD]
        self.right_tip = self.gripper_pcd.points[RIGHT_TIP_IN_CLOUD]
        self.gripper_width = np.linalg.norm(self.left_tip - self.right_tip)
        self.gripper_mid_point = 0.5 * (self.left_tip + self.right_tip)
        self.gripper_length = np.linalg.norm(self.gripper_mid_point) + 0.02

        # Compute the annotations
        transforms, scores = self.compute_annotations()

        '''
        # Merge the grasps
        scores = np.array(scores)
        scores = (scores / np.sum(scores)).tolist()
        merged_transforms, merged_scores = self.merge_grasps(transforms, scores)
        print("Reduced transforms from {} to {}".format(len(transforms), len(merged_transforms)))

        # Generate symmetric grasps (180 degrees around z)
        merged_transforms, merged_scores = self.get_symmetric_grasps(merged_transforms, merged_scores)
        print("Including symmetric grasps, total annotations is {}".format(len(merged_transforms)))

        # Save the merged grasps to file
        save_filename = join(self.args.models_path, self.args.object_model, 'robot_grasps_merged.txt')
        save_grasp_annotations(save_filename, merged_transforms, merged_scores)

        # Visualize the grasps
        if args.visualize:
            self.visualize(merged_transforms)
        '''

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

    def compute_annotations(self):
        # Check if the file already exists
        '''
        save_filename = join(self.args.save_path, object_name, 'robot_grasps.txt')
        transforms, scores = read_grasp_annotations(save_filename)
        if len(transforms) > 0:
            return transforms, scores
        '''

        # Compute robot grasp annotation for each frame
        transforms = []
        scores = []
        dirs = os.listdir(join(self.base_dir, self.data_split))

        # For each directory in the split
        for d in dirs:
            ids = os.listdir(join(self.base_dir, self.data_split, d, 'rgb'))
            # For each frame in the directory
            for i in ids:
                # Get the id
                id = i.split('.')[0]
                # Create the filename for the metadata file
                meta_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', str(id) + '.pkl')
                print('Processing file {}'.format(meta_filename))

                save_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', 'grasp_' + str(id) + '.pkl')
                if self.do_save and exists(save_filename):
                    print('Already exists, skipping')
                else:
                    # Load the annotation
                    # joints3d_anno, obj_rot, obj_trans, obj_id, cam_mat = read_hand_annotations(meta_filename)
                    anno = read_hand_annotations(meta_filename)
                    joints3d_anno = anno['handJoints3D']
                    obj_rot = anno['objRot']
                    obj_trans = anno['objTrans']
                    obj_id = anno['objName']
                    cam_mat = anno['camMat']
                    # If new object id, must load new cloud
                    if obj_id != self.object_id:
                        self.object_id = obj_id
                        self.base_pcd = self.get_object_pcd(self.object_id)
                        # obj_cloud_filename = join(self.args.models_path, self.object_id, 'points.xyz')
                        # obj_cloud_filename = obj_cloud_filename.replace("points.xyz", "points.ply")
                        # o3d.io.write_point_cloud(obj_cloud_filename, self.base_pcd)
                    # Rotate and translate the cloud
                    self.pcd = copy.deepcopy(self.base_pcd)
                    pts = np.asarray(self.pcd.points)
                    pts = np.matmul(pts, cv2.Rodrigues(obj_rot)[0].T)
                    pts += obj_trans
                    self.pcd.points = o3d.utility.Vector3dVector(pts)
                    self.pcd.paint_uniform_color([0.4, 0.4, 0.9])
                    self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)

                    # Get the gripper transform
                    tf_gripper, grasp_points, mid_point, wrist_point = self.get_gripper_transform(joints3d_anno, obj_trans, obj_rot)
                    transforms.append(tf_gripper)
                    if grasp_points is not None:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)

                    # Create a cloud for visualization
                    rgb = read_RGB_img(self.base_dir, d, id, self.data_split)
                    rgb = o3d.geometry.Image(rgb.astype(np.uint8))
                    rgb = o3d.geometry.Image(rgb)
                    depth = read_depth_img(self.base_dir, d, id, self.data_split)
                    depth = o3d.geometry.Image(depth.astype(np.float32))

                    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb, depth)
                    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                    cam_intrinsics.set_intrinsics(640, 480, cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2])
                    scene_pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, cam_intrinsics)
                    scene_pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points) * 1000)
                    # Flip it, otherwise the pointcloud will be upside down
                    scene_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                    # Create hand mesh
                    _, hand_mesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])

                    if self.do_visualize:
                        self.visualize_hand_and_grasp(joints3d_anno, transforms[-1], grasp_points, mid_point, wrist_point, scene_pcd, hand_mesh)
                        # self.visualize_grasp([transforms[-1]])
                        # self.visualize_grasps_all(transforms)

                    if self.do_save:
                        save_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', 'grasp_' + str(id) + '.pkl')
                        save_grasp_annotation_pkl(save_filename, tf_gripper)
                        save_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', 'cloud_' + str(id) + '.ply')
                        o3d.io.write_point_cloud(save_filename, scene_pcd)

                        save_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', 'hand_mesh_' + str(id) + '.ply')
                        mesh = o3d.geometry.TriangleMesh()
                        if hasattr(hand_mesh, 'r'):
                            mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.r))
                            numVert = hand_mesh.r.shape[0]
                        elif hasattr(hand_mesh, 'v'):
                            mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.v))
                            numVert = hand_mesh.v.shape[0]
                        else:
                            raise Exception('Unknown Mesh format')
                        mesh.triangles = o3d.utility.Vector3iVector(np.copy(hand_mesh.f))
                        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.9, 0.4, 0.4]]), [numVert, 1]))
                        o3d.io.write_triangle_mesh(save_filename, mesh)

        # Write to file
        '''
        save_grasp_annotations(save_filename, transforms, scores)
        '''

        # Return the transforms
        return transforms, scores

    def get_gripper_transform(self, joints3d_anno, obj_trans, obj_rot):
        # Position of all joints
        joints3d = joints3d_anno
        # joints3d = joints3d - obj_trans
        # joints3d = np.matmul(joints3d, np.linalg.inv(cv2.Rodrigues(obj_rot)[0].T))

        # Get the grasp points
        finger_positions, grasp_points = self.get_grasp_points(joints3d)
        if np.any(np.isnan(grasp_points)) or np.all(grasp_points[0] - grasp_points[1]) == 0:
            return np.zeros((4, 4)), None, None, None

        # Transform the grasp points to a transformation for the gripper
        tf = self.grasp_to_transformation(finger_positions, grasp_points)
        mid_point, wrist_point = self.get_grasp_mid_and_wrist_points(finger_positions, grasp_points)
        # tf = self.grasp_to_transformation_aligned(grasp_points, mid_point, wrist_point)

        #if self.do_visualize:
        #    self.visualize_hand(joints3d, grasp_points, mid_point, wrist_point)

        # Return the transformation
        return tf, grasp_points, mid_point, wrist_point

    def get_point_grasp_probability(self, joints3d):
        # Get the fingertip positions and distances to the object
        finger_positions, finger_to_point_distances = self.get_finger_positions(joints3d)

        # wrist, t1, t2, t3, t4, i1, i2, i3, i4, m1, m2, m3, m4, r1, r2, r3, r4, p1, p2, p3, p4
        # valid_fingers = [0]
        # for i in range(1, len(tip_indices)):
        #    if finger_to_point_distances[tip_indices[i]] < MIN_TIP_DISTANCE:
        #        valid_fingers.append(i)
        valid_fingers = [0, 1, 2, 3, 4]

        kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        grasp_weights = np.zeros((len(self.pcd.points), 1))
        thumb_weights = np.zeros((len(self.pcd.points), 1))
        non_thumb_weights = np.zeros((len(self.pcd.points), 1))
        for vf in valid_fingers:
            joint_indices = range(1 + 4 * vf, 1 + 4 * vf + 4)
            for j in joint_indices:
                [_, idx, _] = kd_tree.search_radius_vector_3d(finger_positions[j], MAX_RADIUS_SEARCH)
                for i in idx:
                    grasp_weights[i] += 1 / np.linalg.norm(self.pcd.points[i] - finger_positions[j]) * \
                                        finger_weights[vf] * joint_weights[(j - 1) % 4]
                    if vf == 0:
                        thumb_weights[i] += grasp_weights[i]
                    else:
                        non_thumb_weights[i] += grasp_weights[i]

        if np.sum(grasp_weights) > 0 and np.sum(thumb_weights) and np.sum(non_thumb_weights):
            grasp_weights /= np.amax(grasp_weights)
            thumb_weights /= np.amax(thumb_weights)
            non_thumb_weights /= np.amax(non_thumb_weights)

        return finger_positions, grasp_weights, thumb_weights, non_thumb_weights

    def get_finger_positions(self, joints3d):
        # Iterate through fingertips and transform to object frame
        finger_positions = []
        finger_to_point_distances = []
        for i in range(len(all_indices)):
            for j in range(len(all_indices[i])):
                trans3d = joints3d[all_indices[i][j]]
                [_, idx, _] = self.kd_tree.search_knn_vector_3d(trans3d, 1)
                obj_pt = np.asarray(self.pcd.points)[idx[0], :]
                dd = np.linalg.norm(trans3d - obj_pt)
                finger_positions.append(trans3d)
                finger_to_point_distances.append(dd)

        return finger_positions, finger_to_point_distances

    def get_grasp_points(self, joints3d):
        # Get the grasp probabilities for each point on the model
        finger_positions, _, thumb_weights, non_thumb_weights = self.get_point_grasp_probability(joints3d)

        if self.args.force_hand_shape:
            return finger_positions, self.get_grasp_points_from_hand_shape(finger_positions)

        # For the thumb weights, get the centroid of all points with weight above threshold
        thumb_endpoint = np.zeros((3, 1))
        non_thumb_endpoint = np.zeros((3, 1))
        thumb_count = 0.
        non_thumb_count = 0.
        for i in range(len(self.pcd.points)):
            if thumb_weights[i] > 0.5:
                thumb_endpoint[0] += (thumb_weights[i] * self.pcd.points[i][0])
                thumb_endpoint[1] += (thumb_weights[i] * self.pcd.points[i][1])
                thumb_endpoint[2] += (thumb_weights[i] * self.pcd.points[i][2])
                thumb_count += thumb_weights[i]
            if non_thumb_weights[i] > 0.5:
                non_thumb_endpoint[0] += (non_thumb_weights[i] * self.pcd.points[i][0])
                non_thumb_endpoint[1] += (non_thumb_weights[i] * self.pcd.points[i][1])
                non_thumb_endpoint[2] += (non_thumb_weights[i] * self.pcd.points[i][2])
                non_thumb_count += non_thumb_weights[i]

        if thumb_count == 0 or non_thumb_count == 0:
            print('Error in getting grasp points from weights')
            return finger_positions, self.get_grasp_points_from_hand_shape(finger_positions)

        thumb_endpoint /= thumb_count
        non_thumb_endpoint /= non_thumb_count

        # Adjust the grasp points
        end_points = self.adjust_grasp_points(thumb_endpoint, non_thumb_endpoint)

        # Return the end points
        return finger_positions, end_points

    def get_grasp_points_from_hand_shape(self, finger_positions):
        print('Computing grasp using only hand shape')
        # The position of the thumb end point becomes the first grasp point
        thumb_endpoint = np.asarray(finger_positions[tip_indices[0]]).reshape((1, 3))

        # The mean position of the other fingers becomes the second grasp point
        non_thumb_endpoint = np.zeros((3, 1))
        #weight_count = 0.
        #for i in range(1, len(tip_indices)):
        #    non_thumb_endpoint[0] += (finger_positions[tip_indices[i]][0] * finger_weights[i])
        #    non_thumb_endpoint[1] += (finger_positions[tip_indices[i]][1] * finger_weights[i])
        #    non_thumb_endpoint[2] += (finger_positions[tip_indices[i]][2] * finger_weights[i])
        #    weight_count += finger_weights[i]
        #non_thumb_endpoint /= weight_count
        non_thumb_endpoint = np.asarray(finger_positions[tip_indices[1]]).reshape((1, 3))

        return self.adjust_grasp_point_distance(thumb_endpoint[0], non_thumb_endpoint.flatten())

    def adjust_grasp_points(self, thumb_point, non_thumb_point):
        # Get the point on the surface of the object
        try:
            kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
            [_, idx, _] = kd_tree.search_knn_vector_3d(thumb_point, 1)
            new_thumb_point = copy.deepcopy(np.asarray(self.pcd.points)[idx[0], :])
            [_, idx, _] = kd_tree.search_knn_vector_3d(non_thumb_point, 1)
            new_non_thumb_point = copy.deepcopy(np.asarray(self.pcd.points)[idx[0], :])
        except IndexError as error:
            print('ERROR adjusting grasp points, returning original points')
            new_thumb_point = thumb_point
            new_non_thumb_point = non_thumb_point

        return self.adjust_grasp_point_distance(new_thumb_point, new_non_thumb_point)

    def adjust_grasp_point_distance(self, thumb_point, non_thumb_point):
        # Find the distance between the two points
        end_point_distance = np.linalg.norm(thumb_point - non_thumb_point)
        # Get the vector from the thumb to the non thumb
        closing_direction = non_thumb_point - thumb_point
        closing_direction /= np.linalg.norm(closing_direction)
        # Get the distance to move each end point
        distance_to_move = 0.5 * (self.gripper_width - end_point_distance)
        # Adjust the end points
        new_thumb_point = thumb_point - (distance_to_move * closing_direction.flatten())
        new_non_thumb_point = non_thumb_point + (distance_to_move * closing_direction.flatten())

        return new_thumb_point, new_non_thumb_point

    def grasp_to_transformation(self, finger_positions, grasp_points):
        # Compute the closing direction
        closing_direction = grasp_points[1] - grasp_points[0]
        closing_direction /= np.linalg.norm(closing_direction)

        # Compute the mid points
        gripper_mid_point = (grasp_points[0] + grasp_points[1]) / 2

        # Compute the approach direction
        approach_direction = finger_positions[4] - finger_positions[2]
        approach_direction /= np.linalg.norm(approach_direction)

        # Compute the approach start point
        approach_start_point = copy.deepcopy(gripper_mid_point)
        approach_start_point -= (self.gripper_length * approach_direction)

        # Compute the rotation matrix
        #quat1 = tf3d.euler.euler2quat(0., np.pi / 2 - np.arcsin(approach_direction[2]),
        #                              np.pi / 2 + np.arctan2(closing_direction[1], closing_direction[0]))
        #quat2 = tf3d.euler.euler2quat(0., np.arcsin(closing_direction[2]), 0.)
        #rot = tf3d.quaternions.quat2mat(tf3d.quaternions.qmult(quat1, quat2))

        # z = approach
        # y = close
        # x = close X approach
        out_of_plane = np.cross(closing_direction, approach_direction)

        rot = np.eye(3)
        rot[:, 0] = out_of_plane
        rot[:, 1] = closing_direction
        rot[:, 2] = approach_direction

        # Convert to 4x4 matrix
        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[0, 3] = approach_start_point[0]
        tf[1, 3] = approach_start_point[1]
        tf[2, 3] = approach_start_point[2]

        # Return the transform
        return tf

    def grasp_to_transformation_aligned(self, grasp_points, gripper_mid_point_world, wrist_point_world):
        # Get first rotation
        rot1 = vectors_to_rotation_matrix(np.asarray(self.gripper_mid_point),
                                          np.asarray(gripper_mid_point_world) - np.asarray(wrist_point_world))

        # Transform the end points
        points_tf = np.zeros((3, 2))
        points_tf[:, 0] = self.left_tip[0]
        points_tf[:, 1] = self.right_tip[1]
        points_tf = np.matmul(rot1, points_tf).T

        # Rotate so that the planes are aligned
        norm_vec_world = grasp_points[0] - grasp_points[1]
        norm_vec_tf = points_tf[0] - points_tf[1]
        rot2 = vectors_to_rotation_matrix(norm_vec_tf, norm_vec_world)

        # Generate final rotation matrix
        quat1 = tf3d.quaternions.mat2quat(rot1)
        quat2 = tf3d.quaternions.mat2quat(rot2)
        rot_final = tf3d.quaternions.quat2mat(tf3d.quaternions.qmult(quat2, quat1))

        # Construct 4x4 transformation matrix
        tf = np.eye(4)
        tf[:3, :3] = rot_final
        tf[0, 3] = wrist_point_world[0]
        tf[1, 3] = wrist_point_world[1]
        tf[2, 3] = wrist_point_world[2]

        # Return transform
        return tf

    def get_grasp_mid_and_wrist_points(self, finger_positions, grasp_points):
        # Get the gripper configuration in the world frame
        gripper_mid_point = 0.5 * (grasp_points[0] + grasp_points[1])
        approach_direction = gripper_mid_point - finger_positions[0]
        approach_direction /= np.linalg.norm(approach_direction)
        wrist_point = gripper_mid_point - (self.gripper_length * approach_direction)

        # Adjust the wrist point so that its 90 degrees to the rotation
        closing_direction = grasp_points[1] - grasp_points[0]
        closing_direction /= np.linalg.norm(closing_direction)
        # Nearest point on line
        nearest_pt_on_line = np.asarray(nearest_point_on_line(wrist_point - 0.05 * closing_direction,
                                                              wrist_point + 0.05 * closing_direction,
                                                              gripper_mid_point))
        approach_direction = gripper_mid_point - nearest_pt_on_line
        approach_direction /= np.linalg.norm(approach_direction)
        wrist_point = gripper_mid_point - (self.gripper_length * approach_direction)

        # Return the mid point and wrist point
        return gripper_mid_point, wrist_point

    @staticmethod
    def merge_grasps(transforms, scores):
        merge_th_trans = 0.025
        merge_th_rotation = 7.5

        proposals = copy.deepcopy(transforms)
        prop_scores = copy.deepcopy(scores)
        merged = []
        merged_scores = []
        while len(proposals) > 0:
            tf_prop = proposals[0]
            score = prop_scores[0]
            del proposals[0]
            del prop_scores[0]
            merge_idx = []
            m_quat = [tf3d.quaternions.mat2quat(tf_prop[:3, :3])]
            m_trans = [tf_prop[:3, 3]]
            m_scores = [score]
            for idx, tf_temp in enumerate(proposals):
                tra_diff = np.linalg.norm(tf_prop[:3, 3] - tf_temp[:3, 3])
                if tra_diff < merge_th_trans:
                    rot_diff = np.abs(np.degrees(np.array(
                        tf3d.euler.mat2euler(np.matmul(np.linalg.inv(tf_prop[:3, :3]), tf_temp[:3, :3]), 'szyx'))))
                    # -180~180
                    if rot_diff[0] < merge_th_rotation or (180 - rot_diff[0]) < merge_th_rotation:  # consider the flipped hand
                        if rot_diff[1] < merge_th_rotation and rot_diff[2] < merge_th_rotation:
                            m_quat.append(tf3d.quaternions.mat2quat(tf_temp[:3, :3]))
                            m_trans.append(tf_temp[:3, 3])
                            m_scores.append(prop_scores[idx])
                            merge_idx.append(idx)
            tf_new = tf_prop
            if np.sum(m_scores) > 0 and len(merge_idx) > 0:
                m_weights = np.array(m_scores) / np.sum(m_scores)
                if len(m_quat) > 1:
                    tf_new = np.eye(4)
                    tf_new[:3, 3] = np.sum(np.array(m_trans) * np.expand_dims(m_weights, axis=1), axis=0) / np.sum(m_weights)
                    q_avg = quaternion_weighted_average_markley(np.array(m_quat), m_weights)
                    tf_new[:3, :3] = tf3d.quaternions.quat2mat(q_avg)
            merged.append(tf_new)
            merged_scores.append(np.sum(m_scores))
            for idx in range(len(merge_idx) - 1, -1, -1):
                del proposals[merge_idx[idx]]
                del prop_scores[merge_idx[idx]]
        return merged, merged_scores

    @staticmethod
    def get_symmetric_grasps(merged_transforms, merged_scores):
        # Rotate each grasp 180 degrees around the z axis
        num_transforms = len(merged_transforms)
        for i in range(num_transforms):
            quat_original = tf3d.quaternions.mat2quat(merged_transforms[i][:3, :3])
            quat_rot = tf3d.euler.euler2quat(0., 0., np.pi)
            rot = tf3d.quaternions.quat2mat(tf3d.quaternions.qmult(quat_original, quat_rot))

            # Convert to 4x4 matrix
            tf = np.eye(4)
            tf[:3, :3] = rot
            tf[0, 3] = merged_transforms[i][0, 3]
            tf[1, 3] = merged_transforms[i][1, 3]
            tf[2, 3] = merged_transforms[i][2, 3]

            # Append
            merged_transforms.append(tf)
            merged_scores.append(merged_scores[i])

        # Return the grasps that include the symmetries
        return merged_transforms, merged_scores

    def visualize_grasp(self, transform):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Plot the object cloud
        vis.add_geometry(self.pcd)

        self.plot_gripper_cloud(vis, self.gripper_pcd, transform)

        vis.run()
        vis.destroy_window()

    def visualize_grasps_all(self, transforms):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Plot the object cloud
        vis.add_geometry(self.pcd)

        for t in transforms:
            self.plot_gripper_cloud(vis, self.gripper_pcd, t)

        vis.run()
        vis.destroy_window()

    def visualize_hand(self, joints3d, grasp_points, mid_point, wrist_point):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Plot the object cloud
        vis.add_geometry(self.pcd)

        # Plot the finger joints
        for i in range(len(all_indices)):
            for j in range(len(all_indices[i])):
                mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                mm.compute_vertex_normals()
                mm.paint_uniform_color(finger_colors[i])
                trans3d = joints3d[all_indices[i][j]]
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
        line_colors = [finger_colors[1], finger_colors[2], finger_colors[3], finger_colors[4], finger_colors[5],
                       finger_colors[1], finger_colors[1], finger_colors[1],
                       finger_colors[2], finger_colors[2], finger_colors[2],
                       finger_colors[3], finger_colors[3], finger_colors[3],
                       finger_colors[4], finger_colors[4], finger_colors[4],
                       finger_colors[5], finger_colors[5], finger_colors[5]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(joints3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        vis.add_geometry(line_set)

        # Plot the grasp points
        self.plot_gripper_end_points(vis, grasp_points)

        vis.run()
        vis.destroy_window()

    def visualize_hand_and_grasp(self, joints3d, transform, grasp_points, mid_point, wrist_point, scene_pcd=None, hand_mesh = None):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

        # Plot the object cloud
        vis.add_geometry(self.pcd)

        # Plot the finger joints
        for i in range(len(all_indices)):
            for j in range(len(all_indices[i])):
                mm = o3d.create_mesh_sphere(radius=joint_sizes[j])
                mm.compute_vertex_normals()
                mm.paint_uniform_color(finger_colors[i])
                trans3d = joints3d[all_indices[i][j]]
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
        line_colors = [finger_colors[1], finger_colors[2], finger_colors[3], finger_colors[4], finger_colors[5],
                       finger_colors[1], finger_colors[1], finger_colors[1],
                       finger_colors[2], finger_colors[2], finger_colors[2],
                       finger_colors[3], finger_colors[3], finger_colors[3],
                       finger_colors[4], finger_colors[4], finger_colors[4],
                       finger_colors[5], finger_colors[5], finger_colors[5]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(joints3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        vis.add_geometry(line_set)

        # Plot the gripper cloud
        self.plot_gripper_cloud(vis, self.gripper_pcd, transform)

        # Plot the grasp points
        self.plot_gripper_end_points(vis, grasp_points)

        # Plot scene
        if scene_pcd is not None:
            vis.add_geometry(scene_pcd)

        # Visualize hand
        '''
        if hand_mesh is not None:
            mesh = o3d.geometry.TriangleMesh()
            if hasattr(hand_mesh, 'r'):
                mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.r) * 0.001)
                numVert = hand_mesh.r.shape[0]
            elif hasattr(hand_mesh, 'v'):
                mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.v) * 0.001)
                numVert = hand_mesh.v.shape[0]
            else:
                raise Exception('Unknown Mesh format')
            mesh.triangles = o3d.utility.Vector3iVector(np.copy(hand_mesh.f))
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.9, 0.4, 0.4]]), [numVert, 1]))
            o3d.visualization.draw_geometries([mesh])
        '''

        vis.run()
        vis.destroy_window()

    @staticmethod
    def plot_gripper_cloud(vis, gripper_pcd, transform):
        gripper_xyz = copy.deepcopy(np.asarray(gripper_pcd.points).T)
        gripper_xyz = np.matmul(transform[:3, :3], gripper_xyz)
        gripper_xyz[0, :] += transform[0, 3]
        gripper_xyz[1, :] += transform[1, 3]
        gripper_xyz[2, :] += transform[2, 3]
        gripper_pcd = o3d.geometry.PointCloud()
        gripper_pcd.points = o3d.utility.Vector3dVector(gripper_xyz.T)
        gripper_pcd.paint_uniform_color([0.4, 0.9, 0.4])
        vis.add_geometry(gripper_pcd)

    @staticmethod
    def plot_gripper_end_points(vis, end_points, box_dim=0.01):
        mm = o3d.create_mesh_box(width=box_dim, height=box_dim, depth=box_dim)
        mm.compute_vertex_normals()
        mm.paint_uniform_color([0.7, 0.0, 0.7])
        tt = np.eye(4)
        tt[0:3, 3] = end_points[0].flatten()
        mm.transform(tt)
        vis.add_geometry(mm)

        mm = o3d.create_mesh_box(width=box_dim, height=box_dim, depth=box_dim)
        mm.compute_vertex_normals()
        mm.paint_uniform_color([0.0, 0.7, 0.7])
        tt = np.eye(4)
        tt[0:3, 3] = end_points[1].flatten()
        mm.transform(tt)
        vis.add_geometry(mm)


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Robot grasp extraction')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    args.gripper_cloud_path = 'hand_open_new.pcd'
    args.force_hand_shape = True
    args.visualize = True
    args.save = False

    annotator = RobotGraspAnnotator(args)
