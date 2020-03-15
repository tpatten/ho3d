# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

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
VALID_VAL = 255
DIST_THRESHOLD = 0.0005  # 0.006116263647763826
MORPH_CLOSE = True
HAND_MASK_DIR = 'hand'
OBJECT_MASK_DIR = 'object'
HAND_MASK_VISIBLE_DIR = 'hand_vis'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'


class MaskExtractor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.rgb = None
        self.depth = None
        self.anno = None
        self.scene_kd_tree = None
        self.mano_model = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)

        self.compute_masks()

    def compute_masks(self):
        dirs = os.listdir(os.path.join(self.base_dir, self.data_split))

        # For each directory in the split
        for d in dirs:
            # Create the directories to save mask data
            mask_dir = os.path.join(self.base_dir, self.data_split, d, 'mask')
            self.create_directories(mask_dir)

            # Get the scene ids
            frame_ids = os.listdir(os.path.join(self.base_dir, self.data_split, d, 'rgb'))

            # For each frame in the directory
            for fid in frame_ids:
                # Get the id
                frame_id = fid.split('.')[0]
                # Create the filename for the metadata file
                meta_filename = os.path.join(self.base_dir, self.data_split, d, 'meta', str(frame_id) + '.pkl')
                print('Processing file {}'.format(meta_filename))

                save_filename = os.path.join(mask_dir, HAND_MASK_DIR, str(frame_id) + '.png')
                if self.args.save and os.path.exists(save_filename):
                    print('Already exists, skipping')
                else:
                    # Read image, depths maps and annotations
                    self.rgb, self.depth, self.anno = self.load_data(d, frame_id)

                    # Get hand and object meshes
                    #_, hand_mesh = forwardKinematics(self.anno['handPose'], self.anno['handTrans'], self.anno['handBeta'])
                    hand_mesh = self.get_hand_mesh()
                    object_mesh = self.get_object_mesh()

                    # Get hand and object masks
                    hand_mask, object_mask = self.get_masks([hand_mesh.r, hand_mesh.f], [object_mesh.v, object_mesh.f],
                                                            apply_close=MORPH_CLOSE)

                    # Get visible masks
                    scene_cloud_filename = os.path.join(
                        self.base_dir, self.data_split, d, 'meta', 'cloud_' + str(frame_id) + '.ply')
                    scene_pcd = o3d.io.read_point_cloud(scene_cloud_filename)
                    self.scene_kd_tree = o3d.geometry.KDTreeFlann(scene_pcd)
                    hand_mask_visible = self.get_visible_mask([hand_mesh.r, hand_mesh.f], apply_close=MORPH_CLOSE)
                    object_mask_visible = self.subtract_mask(object_mask, hand_mask_visible)
                    kernel = np.ones((7, 7), np.uint8)
                    object_mask_visible = cv2.erode(object_mask_visible, kernel, iterations=1)

                    # Visualize
                    labels = [HAND_LABEL, OBJECT_LABEL]
                    if self.args.visualize:
                        self.visualize([hand_mask, object_mask], [hand_mask_visible, object_mask_visible], labels)

                    # Save
                    if self.args.save:
                        self.save_masks(mask_dir, frame_id, [hand_mask, object_mask],
                                        [hand_mask_visible, object_mask_visible])

                # sys.exit(0)
            # sys.exit(0)

    @staticmethod
    def create_directories(mask_dir):
        if not os.path.isdir(mask_dir):
            try:
                temp_dir = os.path.join(mask_dir, HAND_MASK_DIR)
                os.makedirs(temp_dir)
            except OSError:
                pass

            try:
                temp_dir = os.path.join(mask_dir, OBJECT_MASK_DIR)
                os.makedirs(temp_dir)
            except OSError:
                pass

            try:
                temp_dir = os.path.join(mask_dir, HAND_MASK_VISIBLE_DIR)
                os.makedirs(temp_dir)
            except OSError:
                pass

            try:
                temp_dir = os.path.join(mask_dir, OBJECT_MASK_VISIBLE_DIR)
                os.makedirs(temp_dir)
            except OSError:
                pass

    def load_data(self, seq_name, frame_id):
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)
        return rgb, depth, anno

    def get_hand_mesh(self):
        hand_mesh = copy.deepcopy(self.mano_model)
        hand_mesh.fullpose[:] = self.anno['handPose']
        hand_mesh.trans[:] = self.anno['handTrans']
        hand_mesh.betas[:] = self.anno['handBeta']
        return hand_mesh

    def get_object_mesh(self):
        object_mesh = read_obj(os.path.join(self.base_dir, 'models', self.anno['objName'], 'textured_simple.obj'))
        object_mesh.v = np.matmul(object_mesh.v, cv2.Rodrigues(self.anno['objRot'])[0].T) + self.anno['objTrans']
        return object_mesh

    def get_masks(self, hand_mesh, object_mesh, apply_close=False):
        return self.get_mask(hand_mesh, apply_close), self.get_mask(object_mesh, apply_close)

    def get_mask(self, mesh, apply_close=False):
        # Get the uv coordinates
        mesh_uv = projectPoints(mesh[0], self.anno['camMat'])

        # Generate mask by filling in the faces
        mask = np.zeros((self.rgb.shape[0], self.rgb.shape[1]))
        for face in mesh[1]:
            triangle_cnt = [(640 - int(mesh_uv[face[0]][0]), int(mesh_uv[face[0]][1])),
                            (640 - int(mesh_uv[face[1]][0]), int(mesh_uv[face[1]][1])),
                            (640 - int(mesh_uv[face[2]][0]), int(mesh_uv[face[2]][1]))]

            cv2.drawContours(mask, [np.asarray(triangle_cnt)], 0, VALID_VAL, -1)

        # Morphological closing operation
        if apply_close:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def get_visible_mask(self, mesh, apply_close=False):
        invalid_vertices = []
        for i in range(mesh[0].shape[0]):
            _, _, dist = self.scene_kd_tree.search_knn_vector_3d(mesh[0][i], 1)
            if dist[0] > DIST_THRESHOLD:
                invalid_vertices.append(i)

        valid_faces = []
        for face in mesh[1]:
            valid_face = True
            for idx in invalid_vertices:
                if face[0] == idx or face[1] == idx or face[2] == idx:
                    valid_face = False
                    break
            if valid_face:
                valid_faces.append(face)
        valid_faces = np.asarray(valid_faces)

        mask_vis = self.get_mask([mesh[0], valid_faces], apply_close)

        return mask_vis

    def draw_contour(self, mask):
        contour, _ = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        contour_image = copy.deepcopy(self.rgb)
        contour_image = cv2.drawContours(contour_image, contour, -1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        return contour_image

    @staticmethod
    def subtract_mask(mask_a, mask_b):
        mask_s = np.copy(mask_a)
        for u in range(mask_a.shape[0]):
            for v in range(mask_a.shape[1]):
                if mask_a[u, v] == VALID_VAL:
                    if mask_b[u, v] == VALID_VAL:
                        mask_s[u, v] = 0
        return mask_s

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

    @staticmethod
    def save_masks(mask_dir, frame_id, masks, visible_masks):
        cv2.imwrite(os.path.join(mask_dir, HAND_MASK_DIR, str(frame_id) + '.png'), masks[0])
        cv2.imwrite(os.path.join(mask_dir, OBJECT_MASK_DIR, str(frame_id) + '.png'), masks[1])
        cv2.imwrite(os.path.join(mask_dir, HAND_MASK_VISIBLE_DIR, str(frame_id) + '.png'), visible_masks[0])
        cv2.imwrite(os.path.join(mask_dir, OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png'), visible_masks[1])


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
    args.visualize = False
    args.save = True

    mask_extractor = MaskExtractor(args)
