# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.vis_utils import *
import matplotlib.pyplot as plt
import json


DATA_SPLIT = 'evaluation'
VALID_VAL = 255
MORPH_CLOSE = False
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class VisibilityRatioExtractor:
    def __init__(self, args):
        self.base_dir = args.ho3d_path
        self.models_path = args.models_path
        self.rgb = None
        self.do_visualize = args.visualize
        self.save_filename = args.save_filename

        self.compute_masks()

    def compute_masks(self):
        # Get all directories
        dirs = os.listdir(os.path.join(self.base_dir, DATA_SPLIT))

        # For each directory in the split
        dict = {}
        for d in dirs:
            # Check if segmentation available
            if not os.path.exists(os.path.join(self.base_dir, DATA_SPLIT, d, 'seg')):
                print('Segmentations unavailable for {}'.format(d))
                continue

            # Add directory to the dictionary
            dict[d] = []

            # Get the frame ids
            frame_ids = sorted(os.listdir(os.path.join(self.base_dir, DATA_SPLIT, d, 'rgb')))

            # For each frame in the directory
            for fid in frame_ids:
                # Get the id
                frame_id = fid.split('.')[0]
                print('Processing file {}'.format(frame_id))

                # Begin processing
                print('=====>')

                # Read image, depths maps and annotations
                anno, seg_mask = self.load_data(os.path.join(self.base_dir, DATA_SPLIT, d), frame_id)

                # Get hand and object meshes
                print('getting mesh...')
                object_mesh = self.get_object_mesh(anno)

                # Get object mask
                print('getting masks...')
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
                dict[d].append({frame_id: visibility_ratio})

                # Visualize
                if self.do_visualize:
                    self.visualize(object_mask, seg_mask, occluded_object_mask)

                # Save data
                with open(self.save_filename, 'w') as outfile:
                    json.dump(dict, outfile)

                #sys.exit(0)

    @staticmethod
    def read_annotation(f_name):
        """ Loads the pickle data """
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations pickle file at %s. Aborting.' % f_name)
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)

        return pickle_data

    def load_data(self, base_dir, frame_id):
        # Load the annotation
        meta_filename = os.path.join(base_dir, 'meta', str(frame_id) + '.pkl')
        anno = self.read_annotation(meta_filename)

        # Load the mask file
        mask_filename = os.path.join(base_dir, 'seg', str(frame_id) + '.jpg')
        mask_rgb = cv2.imread(mask_filename)
        # Resize image to original size
        #mask_rgb = cv2.resize(mask_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))

        # Generate binary mask
        for u in range(mask_rgb.shape[0]):
            for v in range(mask_rgb.shape[1]):
                if mask_rgb[u, v, 0] > 230 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] < 10:
                    mask[u, v] = VALID_VAL

        return anno, mask

    def get_object_mesh(self, anno):
        object_mesh = read_obj(os.path.join(self.models_path, anno['objName'], 'textured_simple.obj'))
        object_mesh.v = np.matmul(object_mesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
        return object_mesh

    def get_mask(self, anno, mesh):
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
        mask_s = np.copy(mask_a)
        for u in range(mask_a.shape[0]):
            for v in range(mask_a.shape[1]):
                if mask_a[u, v] == VALID_VAL:
                    if mask_b[u, v] == VALID_VAL:
                        mask_s[u, v] = 0
        return mask_s

    def visualize(self, mask, seg_mask, occluded_mask):
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
    # parse the arguments
    parser = argparse.ArgumentParser(description='HO3D object visibility ratio extractor')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/'
    args.models_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/models'
    args.save_filename = '/home/tpatten/Data/visibility_ratios.json'
    args.visualize = False

    vre = VisibilityRatioExtractor(args)
