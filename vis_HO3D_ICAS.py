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


def splitall(path):
    if path[-1] == '/':
        path = path[:-1]
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def read_icas_annotation(base_dir, file_id):
    meta_filename = os.path.join(base_dir, 'object_pose', file_id + '.pkl')
    pkl_data = load_pickle_data(meta_filename)
    return pkl_data


def ycbv_to_bop(id):
    if id == '004_sugar_box':
        return 'obj_000003'
    elif id == '005_tomato_soup_can':
        return 'obj_000004'
    elif id == '006_mustard_bottle':
        return 'obj_000005'
    elif id == '025_mug':
        return 'obj_000014'
    elif id == '035_power_drill':
        return 'obj_000015'
    else:
        raise Exception('Name of model is unavailable %s' % id)


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

    # get the object name
    objName = splitall(baseDir)[-2]
    objName = ycbv_to_bop(objName)

    # get object 3D corner locations for the current pose
    #objPts = np.loadtxt(os.path.join(YCBModelsDir, 'models', objName, 'points.xyz'))
    #objCorners = get_3d_box_points(objPts)
    pcd = o3d.read_point_cloud(os.path.join(YCBModelsDir, objName + '.ply'))
    objCorners = get_3d_box_points(np.asarray(pcd.points))
    objCornersTrans = np.matmul(objCorners, anno['objRot']) + anno['objTrans']

    ## get the hand Mesh from MANO model for the current pose
    #if split == 'train':
    #    handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
    #else:
    #    handJoints3D = None

    ## project to 2D
    #if split == 'train':
    #    handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
    #else:
    #    # Only root joint available in evaluation split
    #    handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)
    objKps = project_3D_points(camMat, objCornersTrans, is_OpenGL_coords=False)

    # Visualize
    # draw 2D projections of annotations on RGB image
    #if split == 'train':
    #    imgAnno = showHandJoints(img, handKps[jointsMapManoToSimple])
    #else:
    #    # show only projection of root joint in evaluation split
    #    imgAnno = showHandJoints(img, handKps)
    #    # show the hand bounding box
    #    imgAnno = show2DBoundingBox(imgAnno, anno['handBoundingBox'])
    #imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
    imgAnno = copy.deepcopy(img)
    imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)

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

    ## show 3D hand mesh
    #ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    #if split=='train':
    #    plot3dVisualize(ax2, handMesh, flip_x=False, isOpenGLCoords=True, c="r")
    #ax2.title.set_text('Hand Mesh')

    # show 2D projections of annotations on RGB image
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.imshow(imgAnno[:, :, [2, 1, 0]])
    ax3.title.set_text('3D Annotations projected to 2D')

    plt.show()
