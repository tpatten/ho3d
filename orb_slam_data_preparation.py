# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import cv2
import copy
import png


HAND_MASK_DIR = 'hand'
OBJECT_MASK_DIR = 'object'
HAND_MASK_VISIBLE_DIR = 'hand_vis'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'
ORB_SLAM_DIR = 'orb_slam_images'
ORB_SLAM_DEPTH_SCALE = 5000


class Extractor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        if args.mask_erosion_kernel > 0:
            self.erosion_kernel = np.ones((args.mask_erosion_kernel, args.mask_erosion_kernel), np.uint8)
        else:
            self.erosion_kernel = None

        # Create the directories to save data if they do not exist
        self.orb_slam_dir = os.path.join(self.base_dir, self.data_split, self.args.scene, ORB_SLAM_DIR)
        if not os.path.isdir(self.orb_slam_dir):
            try:
                temp_dir = os.path.join(self.orb_slam_dir, 'rgb')
                os.makedirs(temp_dir)
            except OSError:
                pass

            try:
                temp_dir = os.path.join(self.orb_slam_dir, 'depth')
                os.makedirs(temp_dir)
            except OSError:
                pass

        # Load the images, mask and save
        self.process()

    def process(self):
        # Get the scene ids
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, self.args.scene, 'rgb')))

        # For each frame in the directory
        for frame in frame_ids:
            # Get the id
            fid = frame.split('.')[0]
            print('Processing file {}'.format(fid))

            # Read image, depths map and mask
            rgb, depth, mask = self.load_data(self.args.scene, fid)
            if mask is None:
                print('No mask available for frame {}'.format(fid))
                continue

            # Extract the masked images
            rgb_masked = self.apply_mask(rgb, mask)
            depth_masked = self.apply_mask(depth, mask)

            # Visualize
            if args.visualize:
                self.visualize([rgb, depth], [rgb_masked, depth_masked])

            # Save
            if self.args.save:
                # Save the rgb image
                cv2.imwrite(os.path.join(self.orb_slam_dir, 'rgb', str(fid) + '.png'), rgb_masked)
                # Convert depth image to ORB SLAM format (16-bit, scale by 5000)
                depth_masked = (depth_masked * ORB_SLAM_DEPTH_SCALE).astype(np.uint16)
                # Save the depth image
                cv2.imwrite(os.path.join(self.orb_slam_dir, 'depth', str(fid) + '.png'), depth_masked)

    def load_data(self, seq_name, frame_id):
        # Get the RGB and depth
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)

        # Get the mask
        mask_filename = os.path.join(self.base_dir, self.data_split, self.args.scene, 'mask',
                                     OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png')
        mask = None
        if os.path.exists(mask_filename):
            mask = cv2.imread(mask_filename)[:, :, 0]

        # Return
        return rgb, depth, mask

    def apply_mask(self, image, mask, erode=False):
        mask_to_apply = copy.deepcopy(mask)
        if erode and self.erosion_kernel is not None:
            mask_to_apply = cv2.erode(mask_to_apply, self.erosion_kernel, iterations=1)

        masked_image = copy.deepcopy(image)
        for u in range(mask.shape[0]):
            for v in range(mask.shape[1]):
                if mask_to_apply[u, v] == 0:
                    masked_image[u, v] = 0
        return masked_image

    @staticmethod
    def visualize(images, masks):
        # Create window
        fig = plt.figure(figsize=(2, 5))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())

        # Show RGB
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(images[0][:, :, [2, 1, 0]])
        ax1.title.set_text('RGB')

        # Show depth
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.imshow(images[1])
        ax1.title.set_text('DEPTH')

        # Show masked RGB
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.imshow(masks[0][:, :, [2, 1, 0]])
        ax2.title.set_text('RGB MASKED')

        # Show masked depth
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.imshow(masks[1])
        ax2.title.set_text('DEPTH MASKED')

        plt.show()



if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Masked image generator for use in the ORB SLAM pipeline')
    args = parser.parse_args()
    #args.ho3d_path = '/home/tpatten/Data/Hands/HO3D/'
    args.ho3d_path = '/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.scene = 'GPMF14'
    args.visualize = False
    args.save = True
    args.mask_erosion_kernel = 5

    # Extract the masked RGB and depth images and convert depth to ORB SLAM units
    extractor = Extractor(args)

'''
* The color images are stored as 640x480 8-bit RGB images in PNG format.
* The depth maps are stored as 640x480 16-bit monochrome images in PNG format.
* The color and depth images are already pre-registered using the OpenNI driver from PrimeSense, i.e., the pixels in the color and depth images correspond already 1:1.
* The depth images are scaled by a factor of 5000, i.e., a pixel value of 5000 in the depth image corresponds to a distance of 1 meter from the camera, 10000 to 2 meter distance, etc. A pixel value of 0 means missing value/no data.
'''