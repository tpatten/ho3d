# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

from utils.grasp_utils import *
import cv2
import argparse
import open3d as o3d
import transforms3d as tf3d
from os.path import join
import copy


MODEL_PATH = '/v4rtemp/datasets/HandTracking/HO3D_v2/models/'


def plot_points(vis, pcd, gripper_pcd, left_tip, right_tip, gripper_trans, gripper_mid_point, marker_radius):
    # Object point cloud
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)

    # Gripper cloud
    gripper_pcd.paint_uniform_color([0.4, 0.8, 0.4])
    vis.add_geometry(gripper_pcd)

    # Finger tip points, mid point and center point
    endpoints = []
    for i in range(2):
        mm = o3d.create_mesh_sphere(radius=marker_radius)
        mm.compute_vertex_normals()
        if i == 0:
            trans3d = left_tip
            mm.paint_uniform_color([1., 0., 0.])
        else:
            trans3d = right_tip
            mm.paint_uniform_color([0., 0., 1.])
        tt = np.eye(4)
        tt[0:3, 3] = trans3d
        mm.transform(tt)
        vis.add_geometry(mm)
        endpoints.append(trans3d)

    mm = o3d.create_mesh_sphere(radius=marker_radius)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0., 0., 0.])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_trans
    mm.transform(tt)
    vis.add_geometry(mm)
    endpoints.append(gripper_trans)

    mm = o3d.create_mesh_sphere(radius=marker_radius)
    mm.compute_vertex_normals()
    mm.paint_uniform_color([0., 0., 0.])
    tt = np.eye(4)
    tt[0:3, 3] = gripper_mid_point
    mm.transform(tt)
    vis.add_geometry(mm)
    endpoints.append(gripper_mid_point)

    lines = [[0, 1], [2, 3]]
    line_colors = [[1., 0., 1.], [0., 0., 0.]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(endpoints)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    vis.add_geometry(line_set)


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Visualize grasps')
    args = parser.parse_args()
    args.file = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/MDF10/meta/0138.pkl'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    args.gripper_cloud_path = 'hand_open_new.pcd'

    # Get meta data
    anno = read_hand_annotations(args.file)
    obj_rot = anno['objRot']
    obj_trans = anno['objTrans']
    obj_id = anno['objName']

    # Get object cloud and transform
    obj_cloud_filename = join(args.models_path, obj_id, 'points.xyz')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.loadtxt(obj_cloud_filename))
    pts = np.asarray(pcd.points)
    pts = np.matmul(pts, cv2.Rodrigues(obj_rot)[0].T)
    pts += obj_trans
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Get gripper and transform
    gripper_pcd = o3d.io.read_point_cloud(args.gripper_cloud_path)
    #pts = np.asarray(gripper_pcd.points)
    #f = open("hand_open.xyz", "w")
    #for p in pts:
    #    f.write('{} {} {}\n'.format(p[0], p[1], p[2]))
    #f.close()
    #sys.exit(0)
    gripper_pcd_processed = copy.deepcopy(gripper_pcd)
    gripper_pcd_processed_from_scaled = copy.deepcopy(gripper_pcd)
    gripper_transform_file = args.file
    gripper_transform_file = gripper_transform_file.replace("meta/", "meta/grasp_bl_")
    gripper_transform = load_pickle_data(gripper_transform_file)
    gripper_transform = gripper_transform.reshape(4, 4)
    # print(gripper_transform)
    # gripper_pcd.transform(gripper_transform)
    pts = np.asarray(gripper_pcd.points)
    pts = np.matmul(pts, np.linalg.inv(gripper_transform[:3, :3]))
    gripper_trans = gripper_transform[:3, 3]
    # pts += gripper_trans
    gripper_pcd.points = o3d.utility.Vector3dVector(pts)

    # Estimate the roll and pitch
    LEFT_TIP_IN_CLOUD = 3221
    RIGHT_TIP_IN_CLOUD = 5204
    left_tip = pts[LEFT_TIP_IN_CLOUD]
    right_tip = pts[RIGHT_TIP_IN_CLOUD]
    gripper_mid_point = 0.5 * (left_tip + right_tip)
    approach_vec = copy.deepcopy(gripper_mid_point)
    approach_vec /= np.linalg.norm(approach_vec)
    close_vec = right_tip - left_tip
    close_vec /= np.linalg.norm(close_vec)

    # Translate the gripper cloud
    pts = np.asarray(gripper_pcd.points)
    pts += gripper_trans
    gripper_pcd.points = o3d.utility.Vector3dVector(pts)

    # Translate the end points
    left_tip += gripper_trans
    right_tip += gripper_trans
    gripper_mid_point += gripper_trans

    # # Offset and scale
    pts = copy.deepcopy(np.asarray(pcd.points))
    offset = np.expand_dims(np.mean(pts, axis=0), 0)
    pts -= offset
    dist = np.max(np.sqrt(np.sum(pts ** 2, axis=1)), 0)
    pts /= dist
    pcd_scaled = o3d.geometry.PointCloud()
    pcd_scaled.points = o3d.utility.Vector3dVector(pts)

    pts = copy.deepcopy(np.asarray(gripper_pcd.points))
    pts -= offset
    pts /= dist
    gripper_pcd_scaled = o3d.geometry.PointCloud()
    gripper_pcd_scaled.points = o3d.utility.Vector3dVector(pts)

    left_tip_scaled = copy.deepcopy(left_tip)
    left_tip_scaled -= offset.flatten()
    left_tip_scaled /= dist
    right_tip_scaled = copy.deepcopy(right_tip)
    right_tip_scaled -= offset.flatten()
    right_tip_scaled /= dist
    gripper_mid_point_scaled = copy.deepcopy(gripper_mid_point)
    gripper_mid_point_scaled -= offset.flatten()
    gripper_mid_point_scaled /= dist
    gripper_trans_scaled = copy.deepcopy(gripper_trans)
    gripper_trans_scaled -= offset.flatten()
    gripper_trans_scaled /= dist

    # # Retrieve estimate from the approach and roll vectors
    # approach = z
    # close = y
    # x is cross
    out_of_plane = np.cross(close_vec, approach_vec)
    pts = np.asarray(gripper_pcd_processed.points)
    rot = np.eye(3)
    rot[:, 0] = out_of_plane
    rot[:, 1] = close_vec
    rot[:, 2] = approach_vec
    pts = np.matmul(pts, np.linalg.inv(rot))
    pts += gripper_trans
    gripper_pcd_processed.points = o3d.utility.Vector3dVector(pts)

    pts = np.asarray(gripper_pcd_processed_from_scaled.points)
    gripper_trans_from_scaled = copy.deepcopy(gripper_trans_scaled)
    gripper_trans_from_scaled *= dist
    gripper_trans_from_scaled += offset.flatten()
    pts = np.matmul(pts, np.linalg.inv(rot))
    pts += gripper_trans_from_scaled
    gripper_pcd_processed_from_scaled.points = o3d.utility.Vector3dVector(pts)

    # # Visualize

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Plot the coordinate frame
    vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.5))
    # vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

    # Plot original
    plot_points(vis, pcd, gripper_pcd, left_tip, right_tip, gripper_trans, gripper_mid_point, 0.005)

    # Plot scaled
    plot_points(vis, pcd_scaled, gripper_pcd_scaled, left_tip_scaled, right_tip_scaled, gripper_trans_scaled,
                gripper_mid_point_scaled, 0.03)

    # Plot processed
    gripper_pcd_processed.paint_uniform_color([0.8, 0.4, 0.4])
    vis.add_geometry(gripper_pcd_processed)
    gripper_pcd_processed_from_scaled.paint_uniform_color([0.4, 0.4, 0.8])
    vis.add_geometry(gripper_pcd_processed_from_scaled)

    vis.run()
    vis.destroy_window()
