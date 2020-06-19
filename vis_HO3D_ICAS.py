"""
Visualize the projections using estimations
"""
import os
import pip
import argparse
from utils.vis_utils import *
import random
import copy
import matplotlib.pyplot as plt
import chumpy as ch
import pickle
import cv2
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'
model_scale = 0.001

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def read_icas_annotation(base_dir, file_id):
    object_pose_filename = os.path.join(base_dir, 'object_pose', file_id + '.pkl')
    if os.path.exists(object_pose_filename):
        object_pose_data = load_pickle_data(object_pose_filename)
    else:
        object_pose_data = None
        print('No object pose file {}'.format(object_pose_filename))

    hand_pose_filename = os.path.join(base_dir, 'hand_tracker', file_id + '.pkl')
    if not os.path.exists(hand_pose_filename):
        print('No hand pose file {}'.format(hand_pose_filename))
        return object_pose_data

    hand_pose_data = load_pickle_data(hand_pose_filename)
    if object_pose_data is None:
        return hand_pose_data

    return {**object_pose_data, **hand_pose_data}


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


def get_3d_box_points(vertices):
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    pts = []
    pts.append([x_min, y_min, z_min])  # 0
    pts.append([x_min, y_min, z_max])  # 1
    pts.append([x_min, y_max, z_min])  # 2
    pts.append([x_min, y_max, z_max])  # 3
    pts.append([x_max, y_min, z_min])  # 4
    pts.append([x_max, y_min, z_max])  # 5
    pts.append([x_max, y_max, z_min])  # 6
    pts.append([x_max, y_max, z_max])  # 7
    if x_max > 1:  # assume, this is mm scale
        return np.array(pts) * model_scale
    else:
        return np.array(pts)


if __name__ == '__main__':

    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=str, help="Path to annotated dataset")
    ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
    ap.add_argument("-id", required=False, type=str, default='0000', help="image ID")
    args = vars(ap.parse_args())

    # get arguments
    baseDir = args['data_path']
    YCBModelsDir = args['ycbModels_path']
    id = args['id']

    # read image, depths maps and annotations
    img = read_RGB_img(baseDir, '', id, '')
    depth = read_depth_img(baseDir, '', id, '')
    anno = read_icas_annotation(baseDir, id)
    camMat = np.array([538.391033533567, 0.0, 315.3074696331638,
                       0.0, 538.085452058436, 233.0483557773859,
                       0.0, 0.0, 1.0]).reshape((3,3))
    if anno is None:
        raise Exception('No annotations available')

    # get object 3D corner locations for the current pose
    if 'objName' in anno and 'objRot' in anno and 'objTrans' in anno:
        print('Loading object points {}'.format(anno['objName']))
        pcd = o3d.read_point_cloud(os.path.join(YCBModelsDir, anno['objName'] + '.ply'))
        objCorners = get_3d_box_points(np.asarray(pcd.points))
        objCornersTrans = np.matmul(objCorners, anno['objRot']) + anno['objTrans']
        objKps = project_3D_points(camMat, objCornersTrans, is_OpenGL_coords=False)
        objTF = np.eye(4)
        objTF[:3, :3] = anno['objRot']
        objTF[:3, 3] = anno['objTrans']
    else:
        objKps = None

    # get the hand joints
    if 'handJoints3D' in anno:
        handJoints3D = anno['handJoints3D']
        handKps = project_3D_points(camMat, handJoints3D, is_OpenGL_coords=False)
    else:
        handKps = None

    # Visualize
    # draw 2D projections of annotations on RGB image
    if objKps is not None:
        imgObject = copy.deepcopy(img)
        #imgObject = showObjJoints(imgObject, objKps, lineThickness=2)
        draw_3d_poses(imgObject, objCorners, objTF, camMat)

    if handKps is not None:
        imHand = copy.deepcopy(img)
        imHand = showHandJoints(imHand, handKps[jointsMapManoToSimple])

    # create matplotlib window
    fig = plt.figure(figsize=(2, 2))
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())

    # show RGB image
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.imshow(img[:, :, [2, 1, 0]])
    ax0.title.set_text('RGB Image')

    # show depth map
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.imshow(depth)
    ax1.title.set_text('Depth Map')

    # show object pose
    if objKps is not None:
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.imshow(imgObject[:, :, [2, 1, 0]])
        ax2.title.set_text('Object Pose')

    # show hand pose
    if handKps is not None:
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.imshow(imHand[:, :, [2, 1, 0]])
        ax3.title.set_text('Hand Pose')

    plt.show()
