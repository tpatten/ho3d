# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.vis_utils import *
import matplotlib.pyplot as plt
import copy
import json


VALID_VAL = 255
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class VisibilityRatioExtractor:
    def __init__(self, args):
        self.base_dir = args.ho3d_path
        self.data_split = args.split
        self.seg_path = args.seg_path
        self.models_path = args.models_path
        self.rgb = None
        self.obj_name = None
        self.object_mesh = None
        self.do_visualize = args.visualize
        self.save_filename = args.save_filename

        # Compute the masks
        self.compute_masks()

    def compute_masks(self):
        # Get all directories
        dirs = os.listdir(os.path.join(self.base_dir, self.data_split))

        # For each directory in the split
        dict = {}
        for d in dirs:
            # Check if segmentation available
            if not os.path.exists(os.path.join(self.base_dir, self.data_split, d, 'seg')):
                print('Segmentations unavailable for {}'.format(d))
                continue

            # Add directory to the dictionary
            dict[d] = []

            # Get the frame ids
            frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, d, 'rgb')))

            # For each frame in the directory
            for fid in frame_ids:
                # Get the id
                frame_id = fid.split('.')[0]
                print('=====>')
                print('Processing file {}'.format(frame_id))

                # Read image, depths maps and annotations
                anno, seg_mask = self.load_data(os.path.join(self.base_dir, self.data_split, d),
                                                os.path.join(self.seg_path, d), frame_id)

                # Get hand and object meshes
                object_mesh = self.get_object_mesh(anno)

                # Get object mask
                object_mask = self.get_mask(anno, [object_mesh.v, object_mesh.f])
                object_mask = cv2.resize(object_mask, (int(IMAGE_WIDTH/2), int(IMAGE_HEIGHT/2)),
                                         interpolation=cv2.INTER_NEAREST)
                kernel = np.ones((5, 5), np.uint8)
                object_mask = cv2.erode(object_mask, kernel, iterations=1)

                # Get occluded pixels
                occluded_object_mask = self.subtract_mask(object_mask, seg_mask)

                # Compute visibility ratio
                occlusion_ratio = np.count_nonzero(occluded_object_mask == VALID_VAL) /\
                                  np.count_nonzero(object_mask == VALID_VAL)
                visibility_ratio = 1. - occlusion_ratio
                print('Visibility {}'.format(visibility_ratio))

                # Add data to dictionary
                dict[d].append((frame_id, visibility_ratio))

                # Visualize
                if self.do_visualize:
                    self.visualize(object_mask, seg_mask, occluded_object_mask)

                # Save data
                with open(self.save_filename, 'w') as outfile:
                    json.dump(dict, outfile)

    @staticmethod
    def read_annotation(f_name):
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations pickle file at %s. Aborting.' % f_name)
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)

        return pickle_data

    def load_data(self, base_dir, seg_dir, frame_id):
        # Load the annotation
        meta_filename = os.path.join(base_dir, 'meta', str(frame_id) + '.pkl')
        anno = self.read_annotation(meta_filename)

        # Load the mask file
        mask_filename = os.path.join(seg_dir, 'seg', str(frame_id) + '.jpg')
        mask_rgb = cv2.imread(mask_filename)
        # Resize image to original size
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))

        # Generate binary mask
        for u in range(mask_rgb.shape[0]):
            for v in range(mask_rgb.shape[1]):
                if mask_rgb[u, v, 0] > 230 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] < 10:
                    mask[u, v] = VALID_VAL

        return anno, mask

    def get_object_mesh(self, anno):
        # Load a new mesh if current mesh is None or different
        if self.obj_name is None or self.obj_name != anno['objName']:
            print('Loading new object mesh {}'.format(anno['objName']))
            self.object_mesh = read_obj(os.path.join(self.models_path, anno['objName'], 'textured_simple.obj'))
            self.obj_name = anno['objName']
        object_mesh = copy.deepcopy(self.object_mesh)

        # Transform the mesh
        object_mesh.v = np.matmul(object_mesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

        return object_mesh

    @staticmethod
    def get_mask(anno, mesh):
        # Get the uv coordinates
        mesh_uv = projectPoints(mesh[0], anno['camMat'])

        # Generate mask by filling in the faces
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        for face in mesh[1]:
            triangle_cnt = [(IMAGE_WIDTH - int(mesh_uv[face[0]][0]), int(mesh_uv[face[0]][1])),
                            (IMAGE_WIDTH - int(mesh_uv[face[1]][0]), int(mesh_uv[face[1]][1])),
                            (IMAGE_WIDTH - int(mesh_uv[face[2]][0]), int(mesh_uv[face[2]][1]))]
            cv2.drawContours(mask, [np.asarray(triangle_cnt)], 0, VALID_VAL, -1)

        return mask

    @staticmethod
    def subtract_mask(mask_a, mask_b):
        # Compute the indices that are valid in both masks
        valid_idx = set(np.where(mask_a.flatten() == VALID_VAL)[0]) & set(np.where(mask_b.flatten() == VALID_VAL)[0])

        # Set the valid indices to 0
        mask_s = np.copy(mask_a).flatten()
        for v in valid_idx:
            mask_s[v] = 0

        # Reshape back to input shape
        mask_s = mask_s.reshape(mask_a.shape)

        return mask_s

    @staticmethod
    def visualize(mask, seg_mask, occluded_mask):
        # Create window
        fig = plt.figure(figsize=(2, 5))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())

        # Show masks
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(mask)

        # Show segmentation mask
        if seg_mask is not None:
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(seg_mask)

        # Show visible mask
        if occluded_mask is not None:
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(occluded_mask)

        plt.show()


if __name__ == '__main__':
    # Parse the arguments
    # Example:
    # python3 compute_visibility_ho3d.py '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/' '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/models' -split evaluation -save_filename /home/tpatten/Data/visibility_ratios.json
    parser = argparse.ArgumentParser(description="HO3D object visibility ratio extractor")
    parser.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    parser.add_argument("models_path", type=str, help="Path to ycb models directory")
    parser.add_argument("-split", required=False, type=str,
                        help="split type", choices=['train', 'evaluation'], default='evaluation')
    parser.add_argument("-seg_path", required=False, type=str,
                        help="Path to the segmentation mask directory, if not provided then it will search in the ho3d_path")
    parser.add_argument("-save_filename", required=False, type=str, help="Filename of the saved visibility ratios",
                        default="visibility_ratios.json")
    parser.add_argument("--visualize", action="store_true", help="Visualize masks")
    args = parser.parse_args()

    if args.seg_path is None:
        args.seg_path = os.path.join(args.ho3d_path, args.split)

    vre = VisibilityRatioExtractor(args)
