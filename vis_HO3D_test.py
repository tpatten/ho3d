"""
Visualize the projections in published HO-3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
#    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
#    install('chumpy')
    import chumpy as ch

try:
    import pickle
except:
    #install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'

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


if __name__ == '__main__':

    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
    ap.add_argument("-split", required=False, type=str,
                    help="split type", choices=['train', 'evaluation'], default='train')
    ap.add_argument("-seq", required=False, type=str,
                    help="sequence name")
    ap.add_argument("-id", required=False, type=str,
                    help="image ID")
    ap.add_argument("-visType", required=False,
                    help="Type of visualization", choices=['open3d', 'matplotlib'], default='matplotlib')
    args = vars(ap.parse_args())

    baseDir = args['ho3d_path']
    YCBModelsDir = args['ycbModels_path']
    split = args['split']

    # some checks to decide if visualizing one single image or randomly picked images
    if args['seq'] is None:
        args['seq'] = random.choice(os.listdir(join(baseDir, split)))
        runLoop = True
    else:
        runLoop = False

    if args['id'] is None:
        args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
    else:
        pass


    while(True):
        seqName = args['seq']
        id = args['id']

        # read image, depths maps and annotations
        img = read_RGB_img(baseDir, seqName, id, split)
        depth = read_depth_img(baseDir, seqName, id, split)
        anno = read_annotation(baseDir, seqName, id, split)

        # get object 3D corner locations for the current pose
        objCorners = anno['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

        # get the hand Mesh from MANO model for the current pose
        if split == 'train':
            handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
        else:
            handJoints3D = None

        # project to 2D
        if split == 'train':
            handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
        else:
            # Only root joint available in evaluation split
            handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)
        objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)

        # Visualize
        if args['visType'] == 'open3d':
            # open3d visualization

            if not os.path.exists(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')):
                raise Exception('3D object models not available in %s'%(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')))

            # load object model
            objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))

            # apply current pose to the object model
            objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

            # # ----- OBJECT

            obj_uv = projectPoints(objMesh.v, anno['camMat'])

            #obj_uv2 = project_3D_points(anno['camMat'], objMesh.v, is_OpenGL_coords=True)

            #for uv in obj_uv:
            #    img[int(uv[1]), 640 - int(uv[0])] = [100, 100, 100]

            import open3d
            import copy

            # mask = copy.deepcopy(img)
            obj_mask = np.zeros((img.shape[0], img.shape[1]))
            for uv in obj_uv:
                obj_mask[int(uv[1]), 640 - int(uv[0])] = 255

            cv2.imshow("Object Mask", obj_mask)

            kernel = np.ones((5, 5), np.uint8)
            obj_closing = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
            #cv2.imshow("Closing", closing)

            obj_contour, _ = cv2.findContours(np.uint8(obj_closing), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            obj_contour_image = copy.deepcopy(img)
            obj_contour_image = cv2.drawContours(obj_contour_image, obj_contour, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.imshow("Object Contour", obj_contour_image)

            # # ----- HAND
            rn = DepthRenderer()
            rn.camera = ProjectPoints(v=vertices[0], rt=np.zeros(3), t=np.zeros(3), f=np.array([fx, fy]),
                                      c=np.array([u0, v0]), k=np.zeros(5))
            rn.frustum = {'near': 0.01, 'far': 1.5, 'width': img_w, 'height': img_h}
            rn.set(v=vertices[0], f=mano.faces, bgcolor=np.zeros(3))
            # render
            mano_rendered = rn.r
            # show
            cv2.imshow('Depth Image', depth)
            cv2.imshow('MANO Rendered', mano_rendered)
            cv2.imshow('Image GT Annotation', imgAnno_gt)
            cv2.imshow('Image MANO Annotation', imgAnno_mano)



            if hasattr(handMesh, 'r'):
                hand_vertices = open3d.utility.Vector3dVector(np.copy(handMesh.r))
            elif hasattr(handMesh, 'v'):
                hand_vertices = open3d.utility.Vector3dVector(np.copy(handMesh.v))

            hand_uv = projectPoints(hand_vertices, anno['camMat'])
            hand_mask = np.zeros((img.shape[0], img.shape[1]))
            for uv in hand_uv:
                hand_mask[int(uv[1]), 640 - int(uv[0])] = 255

            cv2.imshow("Hand Mask", hand_mask)

            # for each point, find its nearest and create a new point that is their mean
            hand_vertices = np.asarray(hand_vertices)
            hand_pcd = open3d.geometry.PointCloud()
            hand_pcd.points = open3d.utility.Vector3dVector(hand_vertices)
            kd_tree = open3d.geometry.KDTreeFlann(hand_pcd)
            for i in range(3):
                new_vertices = copy.deepcopy(hand_vertices)
                for j in range(hand_vertices.shape[0]):
                    _, idx, _ = kd_tree.search_knn_vector_3d(hand_vertices[j], 2)
                    mid_point = 0.5 * (hand_vertices[j] + hand_vertices[idx[1]])
                    new_vertices[j] = mid_point
                hand_vertices = np.vstack((hand_vertices, new_vertices))

            hand_uv = projectPoints(hand_vertices, anno['camMat'])
            hand_mask = np.zeros((img.shape[0], img.shape[1]))
            for uv in hand_uv:
                hand_mask[int(uv[1]), 640 - int(uv[0])] = 255

            cv2.imshow("Hand Mask 2", hand_mask)

            kernel = np.ones((10, 10), np.uint8)
            hand_closing = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("Hand Closing", hand_closing)

            hand_contour, _ = cv2.findContours(np.uint8(hand_closing), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            hand_contour_image = copy.deepcopy(img)
            hand_contour_image = cv2.drawContours(hand_contour_image, hand_contour, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.imshow("Hand Contour", hand_contour_image)

            cv2.waitKey(0)

            '''
            import open3d as o3d
            if hasattr(handMesh, 'r'):
                hand_vertices = o3d.utility.Vector3dVector(np.copy(handMesh.r))
            elif hasattr(handMesh, 'v'):
                hand_vertices = o3d.utility.Vector3dVector(np.copy(handMesh.v))

            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=480, left=0, top=0, visible=False)

            #mList = [handMesh, objMesh]
            #colorList = ['r', 'g']
            mList = [objMesh]
            colorList = ['g']
            o3dMeshList = []
            for i, m in enumerate(mList):
                mesh = o3d.geometry.TriangleMesh()
                numVert = 0
                if hasattr(m, 'r'):
                    mesh.vertices = o3d.utility.Vector3dVector(np.copy(m.r))
                    numVert = m.r.shape[0]
                elif hasattr(m, 'v'):
                    mesh.vertices = o3d.utility.Vector3dVector(np.copy(m.v))
                    numVert = m.v.shape[0]
                mesh.triangles = o3d.utility.Vector3iVector(np.copy(m.f))
                if colorList[i] == 'r':
                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
                elif colorList[i] == 'g':
                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
                vis.add_geometry(mesh)

            depth_vis = vis.capture_depth_float_buffer(do_render=True)
            vis.capture_depth_image("rendered_direct_depth.png", do_render=True, depth_scale=1000.0)
            depth_np = np.asarray(depth_vis)
            print(depth_np.shape)
            plt.imsave("rendered_depth.png", depth_np, dpi=1)

            #vis.run()
            #vis.destroy_window()

            import copy
            # mask = copy.deepcopy(img)
            mask = np.zeros((img.shape[0], img.shape[1]))
            for u in range(depth_np.shape[0]):
                for v in range(depth_np.shape[1]):
                    if depth_np[u, v] > 0:
                        # mask[u, v] = 255
                        mask[u + 35, v + 75] = 255
                        # mask[u + 35, v + 75] = [0, 0, 255]
            contour, _ = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            img_cont = cv2.drawContours(img, contour, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

            cv2.imshow("Image", img_cont)
            cv2.waitKey(0)
            '''

            '''
            # create matplotlib window
            fig = plt.figure(figsize=(1, 2))
            figManager = plt.get_current_fig_manager()
            figManager.resize(*figManager.window.maxsize())

            # show RGB image
            ax0 = fig.add_subplot(1, 2, 1)
            ax0.imshow(img[:, :, [2, 1, 0]])

            ax0 = fig.add_subplot(1, 2, 2)
            ax0.imshow(mask[:, :, [2, 1, 0]])

            plt.show()
            '''

        elif args['visType'] == 'matplotlib':

            # draw 2D projections of annotations on RGB image
            if split == 'train':
                imgAnno = showHandJoints(img, handKps[jointsMapManoToSimple])
            else:
                # show only projection of root joint in evaluation split
                imgAnno = showHandJoints(img, handKps)
                # show the hand bounding box
                imgAnno = show2DBoundingBox(imgAnno, anno['handBoundingBox'])
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

            # show 3D hand mesh
            ax2 = fig.add_subplot(2, 2, 3, projection="3d")
            if split=='train':
                plot3dVisualize(ax2, handMesh, flip_x=False, isOpenGLCoords=True, c="r")
            ax2.title.set_text('Hand Mesh')

            # show 2D projections of annotations on RGB image
            ax3 = fig.add_subplot(2, 2, 4)
            ax3.imshow(imgAnno[:, :, [2, 1, 0]])
            ax3.title.set_text('3D Annotations projected to 2D')

            plt.show()
        else:
            raise Exception('Unknown visualization type')

        if runLoop:
            args['seq'] = random.choice(os.listdir(join(baseDir, split)))
            args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
        else:
            break
