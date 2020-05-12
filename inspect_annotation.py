import os
import numpy as np
import pickle
import open3d as o3d
import cv2
import copy
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def load_annotation(target_dir, file):
    anno_file_name = os.path.join(target_dir, 'meta', file)
    anno_file_name = anno_file_name.replace(".png", ".pkl")
    with open(anno_file_name, 'rb') as f:
        try:
            anno = pickle.load(f, encoding='latin1')
        except:
            anno = pickle.load(f)
    return anno


def get_object_cloud(anno_data, curr_object_name):
    object_name = anno_data['objName']
    object_rot = anno_data['objRot']
    object_tra = anno_data['objTrans']

    # Load and transform the object cloud
    if object_name != curr_object_name:
        curr_object_name = object_name
        obj_cloud_filename = os.path.join(models_path, object_name, 'points.xyz')
        base_object_pcd.points = o3d.utility.Vector3dVector(np.loadtxt(obj_cloud_filename))
    object_pcd = copy.deepcopy(base_object_pcd)
    pts = np.asarray(object_pcd.points)
    pts = np.matmul(pts, cv2.Rodrigues(object_rot)[0].T)
    pts += object_tra
    object_pcd.points = o3d.utility.Vector3dVector(pts)
    object_pcd.paint_uniform_color([0.4, 0.4, 0.9])

    return object_pcd


def get_hand_pcd(anno_data):
    _, hand_mesh = forwardKinematics(anno_data['handPose'], anno_data['handTrans'], anno_data['handBeta'])
    mesh = o3d.geometry.TriangleMesh()
    if hasattr(hand_mesh, 'r'):
        mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.r))
        num_vert = hand_mesh.r.shape[0]
    elif hasattr(hand_mesh, 'v'):
        mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.v))
        num_vert = hand_mesh.v.shape[0]
    else:
        raise Exception('Unknown Mesh format')
    mesh.triangles = o3d.utility.Vector3iVector(np.copy(hand_mesh.f))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.9, 0.4, 0.4]]), [num_vert, 1]))

    # Convert hand mesh to point cloud
    hand_cloud = o3d.geometry.PointCloud()
    hand_cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    hand_cloud.paint_uniform_color([0.9, 0.4, 0.4])

    return hand_cloud


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


def get_scene_pcd(target_dir, file, cam_mat):
    # Load the rgb image
    rgb_filename = os.path.join(target_dir, 'rgb', file)
    rgb = cv2.imread(rgb_filename)
    rgb = o3d.geometry.Image(rgb.astype(np.uint8))
    rgb = o3d.geometry.Image(rgb)

    # Load the depth image
    depth_scale = 0.00012498664727900177
    depth_filename = os.path.join(target_dir, 'depth', file)
    depth_img = cv2.imread(depth_filename)
    depth = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    depth = depth * depth_scale
    depth = o3d.geometry.Image(depth.astype(np.float32))

    # Convert to rgbd image
    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb, depth)

    # Generate the point cloud using the camera matrix
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.set_intrinsics(640, 480, cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2])
    scene_pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, cam_intrinsics)
    scene_pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points) * 1000)
    # Flip it, otherwise the point cloud will be upside down
    scene_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return scene_pcd


def is_frame_valid(object_pcd, hand_pcd, scene_pcd):
    # Create a kd tree of the scene
    kd_tree = o3d.geometry.KDTreeFlann(scene_pcd)

    # TODO: What to do with the measured distances?
    for pt in np.asarray(object_pcd.points):
        [_, idx, _] = kd_tree.search_knn_vector_3d(pt, 1)
        d = np.linalg.norm(pt - np.asarray(scene_pcd.points)[idx[0]])

    for pt in np.asarray(hand_pcd.points):
        [_, idx, _] = kd_tree.search_knn_vector_3d(pt, 1)
        d = np.linalg.norm(pt - np.asarray(scene_pcd.points)[idx[0]])

    return True


def visualize(object_pcd, hand_pcd, scene_pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Coordinate frame
    vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.05))

    # Object cloud
    vis.add_geometry(object_pcd)

    # Hand cloud
    vis.add_geometry(hand_pcd)

    # Scene cloud
    vis.add_geometry(scene_pcd)

    # Run and end
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    target_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/ABF10/"
    models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'

    curr_object_name = ''
    base_object_pcd = o3d.geometry.PointCloud()
    for file in sorted(os.listdir(os.path.join(target_dir, 'rgb'))):
        print('--- Inspecting file {}'.format(file))

        # Load the annotation file
        anno = load_annotation(target_dir, file)

        # Get the object point cloud
        object_pcd = get_object_cloud(anno, curr_object_name)

        # Create the hand point cloud
        hand_pcd = get_hand_pcd(anno)

        # Create the scene point cloud
        scene_pcd = get_scene_pcd(target_dir, file, anno['camMat'])

        is_frame_valid(object_pcd, hand_pcd, scene_pcd)

        # Visualize
        visualize(object_pcd, hand_pcd, scene_pcd)
