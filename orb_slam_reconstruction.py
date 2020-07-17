import os
import numpy as np
import transforms3d as tf3d
import open3d as o3d
import cv2
import pickle


ORB_SLAM_DIR = 'orb_slam_images'


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            words = metastr.split()
            mat = np.zeros(shape=(4, 4))
            mat[0, 3] = float(words[1])
            mat[1, 3] = float(words[2])
            mat[2, 3] = float(words[3])
            rot = tf3d.quaternions.quat2mat([float(words[7]), float(words[4]), float(words[5]), float(words[6])])
            mat[:3, :3] = rot
            mat[3, 3] = 1.0
            traj.append(CameraPose(words[0], mat))
            metastr = f.readline()
    return traj


def read_associations(filename):
    assocs = {}
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            words = metastr.split()
            assocs[words[0]] = words[2]
            metastr = f.readline()
    return assocs


def read_camera_intrinsics(camera_id=0, data_path=''):
    cam_params = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    cam_mat = np.zeros((3, 3))
    cam_mat[2, 2] = 1.
    valid_camera_id = True
    # TUM_1
    if camera_id == 1:
        cam_mat[0, 0] = 517.306408
        cam_mat[1, 1] = 516.469215
        cam_mat[0, 2] = 318.643040
        cam_mat[1, 2] = 255.313989
    # TUM_2
    elif camera_id == 2:
        cam_mat[0, 0] = 520.908620
        cam_mat[1, 1] = 521.007327
        cam_mat[0, 2] = 325.141442
        cam_mat[1, 2] = 249.701764
    # HO3D
    elif camera_id == 3:
        f_name = os.path.join(data_path, 'meta/0000.pkl')
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations pickle file at %s. Aborting.' % f_name)
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)
        print(pickle_data)
        cam_mat[0, 0] = pickle_data['camMat'][0, 0]
        cam_mat[1, 1] = pickle_data['camMat'][1, 1]
        cam_mat[0, 2] = pickle_data['camMat'][0, 2]
        cam_mat[1, 2] = pickle_data['camMat'][1, 2]
    else:
        valid_camera_id = False

    if valid_camera_id:
        cam_params.set_intrinsics(640, 480, cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2])

    return cam_params


if __name__ == "__main__":
    #print(o3d.__version__)
    ho3d = True
    data_path = '/home/tpatten/Data/Hands/HO3D/train/ABF10'
    #data_path = '/home/tpatten/Data/ORB_SLAM/TUM/rgbd_dataset_freiburg1_teddy'
    if ho3d:
        cam_params = read_camera_intrinsics(camera_id=3, data_path=data_path)
        camera_trajectory_filename = os.path.join(data_path, ORB_SLAM_DIR, 'KeyFrameTrajectory.txt')
        associations_filename = os.path.join(data_path, ORB_SLAM_DIR, 'associations.txt')
    else:
        cam_params = read_camera_intrinsics(camera_id=1)  # Or 2
        camera_trajectory_filename = os.path.join(data_path, 'KeyFrameTrajectory.txt')
        associations_filename = os.path.join(data_path, 'associations.txt')

    camera_poses = read_trajectory(camera_trajectory_filename)
    rgb_to_depth = read_associations(associations_filename)

    #volume = o3d.integration.ScalableTSDFVolume(
    #    voxel_length=4.0/512.0,
    #    sdf_trunc=0.04,
    #    color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=2.0/512.0,
        sdf_trunc=0.02,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_poses)):
        filename = camera_poses[i].metadata
        print("Integrate {:d}-th image into the volume".format(i))

        if ho3d:
            color = o3d.io.read_image(os.path.join(data_path, ORB_SLAM_DIR, 'rgb', filename + '.png'))
            depth_scale = 0.00012498664727900177
            depth = cv2.imread(os.path.join(data_path, 'depth', rgb_to_depth[filename] + '.png'))
            d_data = depth[:, :, 2] + depth[:, :, 1] * 256
            d_data = d_data * depth_scale
            depth = o3d.geometry.Image((d_data * 1000).astype(np.float32))  # X 1000 ?
        else:
            color = o3d.io.read_image(os.path.join(data_path, 'rgb', filename + '.png'))
            depth = o3d.io.read_image(os.path.join(data_path, 'depth', rgb_to_depth[filename] + '.png'))
            d_data = np.asarray(depth) / 5000
            depth = o3d.geometry.Image((d_data * 1000).astype(np.float32))

        rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(
            color, depth, depth_trunc=1.2, convert_rgb_to_intensity=False)

        volume.integrate(rgbd, cam_params, np.linalg.inv(camera_poses[i].pose))

        #if i == 40:
        #    break

    # Generate the mesh
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Save
    o3d.io.write_triangle_mesh(os.path.join(data_path, 'reconstruction.ply'), mesh)

    # Visualize
    o3d.visualization.draw_geometries([mesh])
