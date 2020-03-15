import argparse
from utils.grasp_utils import *
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import copy

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'
HAND_LABEL = 'hand'
OBJECT_LABEL = 'object'


class MaskExtractor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'

        self.rgb = None
        self.depth = None
        self.anno = None

        self.scene_kd_tree = None

        #with open(MANO_MODEL_PATH, 'rb') as f:
        #    model = pickle.load(f, encoding='latin1')
        #self.faces = model['f']

        self.compute_masks()

    def compute_masks(self):
        dirs = os.listdir(os.path.join(self.base_dir, self.data_split))

        # For each directory in the split
        for d in dirs:
            ids = os.listdir(os.path.join(self.base_dir, self.data_split, d, 'rgb'))
            # For each frame in the directory
            for i in ids:
                # Get the id
                id = i.split('.')[0]
                # Create the filename for the metadata file
                meta_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', str(id) + '.pkl')
                print('Processing file {}'.format(meta_filename))

                # Read image, depths maps and annotations
                self.rgb, self.depth, self.anno = self.load_data(d, id)

                # Get hand and masks
                _, hand_mesh = forwardKinematics(self.anno['handPose'], self.anno['handTrans'], self.anno['handBeta'])
                object_mesh = read_obj(os.path.join(self.base_dir, 'models', self.anno['objName'], 'textured_simple.obj'))
                object_mesh.v = np.matmul(object_mesh.v, cv2.Rodrigues(self.anno['objRot'])[0].T) + self.anno['objTrans']
                hand_mask, object_mask = self.get_masks([hand_mesh.r, hand_mesh.f], [object_mesh.v, object_mesh.f])

                #'''
                # Get visible masks
                # hand_mask_visible, object_mask_visible = self.get_visible_masks(hand_mask, object_mask)
                scene_cloud_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', 'cloud_' + str(id) + '.ply')
                scene_pcd = o3d.io.read_point_cloud(scene_cloud_filename)
                self.scene_kd_tree = o3d.geometry.KDTreeFlann(scene_pcd)
                hand_mask_visible, object_mask_visible = self.get_visible_masks(
                    [hand_mesh.r, hand_mesh.f], [object_mesh.v, object_mesh.f])
                #'''
                #hand_mask_visible = None
                #object_mask_visible = None

                # Visualize
                labels = [HAND_LABEL, OBJECT_LABEL]
                if self.args.visualize:
                    self.visualize([hand_mask, object_mask], [hand_mask_visible, object_mask_visible], labels)

                # Save
                if self.args.save:
                    self.save_masks([hand_mask, object_mask], [hand_mask_visible, object_mask_visible], labels)

                sys.exit(0)

    def load_data(self, seq_name, frame_id):
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)
        return rgb, depth, anno

    def get_masks(self, hand_mesh, object_mesh):
        # return self.get_hand_mask(hand_mesh), self.get_object_mask(object_mesh)
        return self.get_mask(hand_mesh), self.get_mask(object_mesh)

    '''
    def get_hand_mask(self, hand_mesh):
        if hasattr(hand_mesh, 'r'):
            hand_vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.r))
        elif hasattr(hand_mesh, 'v'):
            hand_vertices = o3d.utility.Vector3dVector(np.copy(hand_mesh.v))

        hand_uv = projectPoints(hand_vertices, self.anno['camMat'])

        hand_mask = np.zeros((self.rgb.shape[0], self.rgb.shape[1]))
        for uv in hand_uv:
            hand_mask[int(uv[1]), 640 - int(uv[0])] = 255

        for face in self.faces:
            triangle_cnt = [(640 - int(hand_uv[face[0]][0]), int(hand_uv[face[0]][1])),
                            (640 - int(hand_uv[face[1]][0]), int(hand_uv[face[1]][1])),
                            (640 - int(hand_uv[face[2]][0]), int(hand_uv[face[2]][1]))]

            cv2.drawContours(hand_mask, [np.asarray(triangle_cnt)], 0, 255, -1)

        return hand_mask

    def get_object_mask(self, object_mesh):
        # Get the UV coordinates
        obj_uv = projectPoints(object_mesh.v, self.anno['camMat'])

        # Mask the object points
        obj_mask = np.zeros((self.rgb.shape[0], self.rgb.shape[1]))
        for uv in obj_uv:
            obj_mask[int(uv[1]), 640 - int(uv[0])] = 255

        # Morphological closing operation
        kernel = np.ones((5, 5), np.uint8)
        obj_closing = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)

        return obj_closing
    '''

    def get_mask(self, mesh, apply_close=False):
        # Get the uv coordinates
        mesh_uv = projectPoints(mesh[0], self.anno['camMat'])

        # Generate mask by filling in the faces
        mask = np.zeros((self.rgb.shape[0], self.rgb.shape[1]))
        for face in mesh[1]:
            triangle_cnt = [(640 - int(mesh_uv[face[0]][0]), int(mesh_uv[face[0]][1])),
                            (640 - int(mesh_uv[face[1]][0]), int(mesh_uv[face[1]][1])),
                            (640 - int(mesh_uv[face[2]][0]), int(mesh_uv[face[2]][1]))]

            cv2.drawContours(mask, [np.asarray(triangle_cnt)], 0, 255, -1)

        # Morphological closing operation
        if apply_close:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def get_visible_masks(self, hand_mesh, object_mesh):
        return self.get_visible_mask(hand_mesh), self.get_visible_mask(object_mesh)

    def get_visible_mask(self, mesh, apply_close=False):
        dist_threshold = 0.0005  # 0.006116263647763826
        invalid_vertices = []
        for i in range(mesh[0].shape[0]):
            _, _, dist = self.scene_kd_tree.search_knn_vector_3d(mesh[0][i], 1)
            if dist[0] > dist_threshold:
                invalid_vertices.append(i)

        valid_faces = []
        counter = 0
        for face in mesh[1]:
            counter += 1
            print(str(counter) + '/' + str(mesh[1].shape[0]))
            valid_face = True
            for idx in invalid_vertices:
                if face[0] == idx or face[1] == idx or face[2] == idx:
                    valid_face = False
                    break
            if valid_face:
                valid_faces.append(face)
        valid_faces = np.asarray(valid_faces)

        mask_vis = self.get_mask([mesh[0], valid_faces], apply_close)

        '''
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the object cloud
        #vis.add_geometry(self.pcd)

        # Plot scene
        vis.add_geometry(scene_pcd)

        # Visualize hand
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.copy(vertices))
        # mesh.triangles = o3d.utility.Vector3iVector(np.copy(hand_mesh.f))
        mesh.triangles = o3d.utility.Vector3iVector(np.copy(faces))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.9, 0.4, 0.4]]), [vertices.shape[0], 1]))
        vis.add_geometry(mesh)

        vis.run()
        vis.destroy_window()
        '''

        return mask_vis

    def get_visible_masks_old(self, hand_mask, object_mask):
        hand_vis = copy.deepcopy(hand_mask)
        object_vis = copy.deepcopy(object_mask)

        # Get the pixels that overlap the hand and object
        overlap = np.zeros(hand_mask.shape)
        for u in range(hand_mask.shape[0]):
            for v in range(hand_mask.shape[1]):
                if hand_mask[u, v] == 255 and object_mask[u, v] == 255:
                    overlap[u, v] = 255

        # Remove overlapping pixels from the masks
        overlap_indices = np.where(overlap == 255)
        for i in range(len(overlap_indices[0])):
            hand_vis[overlap_indices[0][i], overlap_indices[1][i]] = 0
            object_vis[overlap_indices[0][i], overlap_indices[1][i]] = 0

        '''
        # Find all boundary points
        boundary = np.zeros(hand_mask.shape)
        for i in range(len(overlap_indices[0])):
            # Left
            if overlap[overlap_indices[0][i], overlap_indices[1][i] - 1] == 0:
                boundary[overlap_indices[0][i], overlap_indices[1][i]] = 255
            # Right
            elif overlap[overlap_indices[0][i], overlap_indices[1][i] + 1] == 0:
                boundary[overlap_indices[0][i], overlap_indices[1][i]] = 255
            # Top
            elif overlap[overlap_indices[0][i] - 1, overlap_indices[1][i]] == 0:
                boundary[overlap_indices[0][i], overlap_indices[1][i]] = 255
            # Bottom
            elif overlap[overlap_indices[0][i] + 1, overlap_indices[1][i]] == 0:
                boundary[overlap_indices[0][i], overlap_indices[1][i]] = 255
        '''

        # Find all boundary points
        left_lab = 'left'
        right_lab = 'right'
        top_lab = 'top'
        bottom_lab = 'bottom'
        boundary = {left_lab: [], right_lab: [], top_lab: [], bottom_lab: []}
        for i in range(len(overlap_indices[0])):
            # Left
            if overlap[overlap_indices[0][i], overlap_indices[1][i] - 1] == 0:
                boundary[left_lab].append((overlap_indices[0][i], overlap_indices[1][i]))
            # Right
            elif overlap[overlap_indices[0][i], overlap_indices[1][i] + 1] == 0:
                boundary[right_lab].append((overlap_indices[0][i], overlap_indices[1][i]))
            # Top
            elif overlap[overlap_indices[0][i] - 1, overlap_indices[1][i]] == 0:
                boundary[bottom_lab].append((overlap_indices[0][i], overlap_indices[1][i]))
            # Bottom
            elif overlap[overlap_indices[0][i] + 1, overlap_indices[1][i]] == 0:
                boundary[bottom_lab].append((overlap_indices[0][i], overlap_indices[1][i]))

        # For each boundary pixel, find if the nearest label is hand or object
        d_thresh = 0.00001
        depth_copy = copy.deepcopy(self.rgb)
        for p in boundary[left_lab]:
            if hand_mask[p[0], p[1] - 1] == 255:
                d_overlap = self.depth[p[0], p[1]]
                d_neighbor = self.depth[p[0], p[1] - 1]
                d_diff = np.abs(d_overlap - d_neighbor)
                print(d_diff)
                if d_diff < d_thresh:
                    print('hand -> hand')
                else:
                    print('hand -> object')
                depth_copy[p[0], p[1]] = [0, 0, 255]
                depth_copy[p[0], p[1] - 1] = [0, 255, 0]
            elif object_mask[p[0], p[1] - 1] == 255:
                d_overlap = self.depth[p[0], p[1]]
                d_neighbor = self.depth[p[0], p[1] - 1]
                d_diff = np.abs(d_overlap - d_neighbor)
                if d_diff < d_thresh:
                    print('object -> object')
                else:
                    print('object -> hand')
            else:
                print('none')
        cv2.imshow("Pixel", depth_copy)
        cv2.waitKey(0)

        return hand_vis, object_vis

    def draw_contour(self, mask):
        contour, _ = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        contour_image = copy.deepcopy(self.rgb)
        contour_image = cv2.drawContours(contour_image, contour, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        return contour_image

    @staticmethod
    def apply_mask(image, mask, erode=False):
        mask_to_apply = copy.deepcopy(mask)
        if erode:
            kernel = np.ones((8, 8), np.uint8)
            mask_to_apply = cv2.erode(mask_to_apply, kernel, iterations=1)

        masked_image = copy.deepcopy(image)
        for u in range(mask.shape[0]):
            for v in range(mask.shape[1]):
                if mask_to_apply[u, v] == 0:
                    masked_image[u, v] = 0
        return masked_image

    def visualize(self, masks, visible_masks, labels):
        # Create window
        fig = plt.figure(figsize=(2, 5))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())

        for i in range(len(masks)):
            # Show hand contour
            ax1 = fig.add_subplot(2, 5, 5 * i + 1)
            ax1.imshow(self.draw_contour(masks[i])[:, :, [2, 1, 0]])
            ax1.title.set_text(labels[i] + ' contour')

            # Show mask
            ax2 = fig.add_subplot(2, 5, 5 * i + 2)
            ax2.imshow(masks[i])
            ax2.title.set_text(labels[i] + ' mask')

            # Show masked RGB
            ax3 = fig.add_subplot(2, 5, 5 * i + 3)
            ax3.imshow(self.apply_mask(self.rgb, masks[i])[:, :, [2, 1, 0]])
            ax3.title.set_text(labels[i] + ' masked')

            # Show visible mask
            if visible_masks[i] is not None:
                ax4 = fig.add_subplot(2, 5, 5 * i + 4)
                ax4.imshow(visible_masks[i])
                ax4.title.set_text(labels[i] + ' visible mask')

            # Show visible masked RGB
            if visible_masks[i] is not None:
                ax5 = fig.add_subplot(2, 5, 5 * i + 5)
                if labels[i] == OBJECT_LABEL:
                    ax5.imshow(self.apply_mask(self.rgb, visible_masks[i], erode=True)[:, :, [2, 1, 0]])
                else:
                    ax5.imshow(self.apply_mask(self.rgb, visible_masks[i])[:, :, [2, 1, 0]])
                ax5.title.set_text(labels[i] + ' visible masked')

        plt.show()

    def save_masks(self, masks, visible_masks, labels):
        return True


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
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Object and hand mask extraction')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.models_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models'
    args.visualize = True
    args.save = False

    mask_extractor = MaskExtractor(args)
