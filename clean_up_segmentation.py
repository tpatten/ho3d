import argparse
from utils.grasp_utils import *
import cv2
import imageio
from enum import IntEnum


MASK_FLOW_DIR = 'mask_hsdc_ofdd'
MASK_HAND_DIR = 'mask_hsdc'
MASK_PERSON_DIR = 'mask_person'
MASK_P2P = 'mask_pix2pose'
HO3D_PATH = '/home/tpatten/Data/bop/ho3d'
SPLIT = 'test'


class CleanUpMode(IntEnum):
    FLOW = 1
    MRCNN = 2


class OpticalFlowEstimator:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = args.split

    def process(self):
        # Get all sequence names
        root_dir = os.path.join(self.base_dir, self.data_split)
        dirs = [o for o in os.listdir(root_dir) if os.path.isdir("{}/{}".format(root_dir, o))]
        dirs.sort()

        for seq in dirs:
            # Directory where the data for this sequence is
            seq_dir = os.path.join(root_dir, seq)
            print('Processing {}'.format(seq_dir))
            # Create the output directory
            os.makedirs("{}/{}".format(seq_dir, self.args.output_dir), exist_ok=True)
            # Get all filenames
            frames = [f.split('.')[0] for f in os.listdir("{}/rgb".format(seq_dir))
                      if os.path.splitext(os.path.join(root_dir, seq, f))[1] == '.png']
            frames.sort()

            # For each frame in the sequence
            window_size = 100
            prev_idx = len(frames) - window_size
            for curr_idx in range(len(frames)):
                prev_rgb_img, prev_depth_img, prev_mask = self.load_data(seq, frames[prev_idx])
                rgb_img, depth_img, mask = self.load_data(seq, frames[curr_idx])
                static_mask = self.update_mask(prev_rgb_img, rgb_img, prev_depth_img, depth_img, prev_mask, mask)

                prev_idx += 1
                if prev_idx >= len(frames):
                    prev_idx = 0

                if (curr_idx % 25) == 0:
                    print(' -- {}'.format(curr_idx))
                if self.args.save:
                    cv2.imwrite("{}/{}/{}.png".format(seq_dir, self.args.output_dir, frames[curr_idx]),
                                static_mask.astype(np.uint8))

                #break

            #break

    def load_data(self, seq_name, frame_id):
        # RGB
        rgb_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'rgb', frame_id + '.png')
        rgb = cv2.imread(rgb_filename)

        # Depth
        depth_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'depth', frame_id + '.png')
        depth = imageio.imread(depth_filename).astype(np.float32) / 1000

        # Mask
        mask_filename = os.path.join(self.base_dir, self.data_split, seq_name, self.args.mask_dir, frame_id + '.png')
        mask = cv2.imread(mask_filename)[:, :, 0]

        return rgb, depth, mask

    def update_mask(self, p_rgb, c_rgb, p_dep, c_dep, p_mask, c_mask):
        # Compute the optical flow
        flow = self.dense_flow(p_rgb, c_rgb)

        flow_mag, flow_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Mask the optical flow
        flow_thresh = 0.01  # 0.01
        # static_flow_mask = (flow_mag < flow_thresh).astype(np.uint8)
        dynamic_flow_mask = (flow_mag >= flow_thresh).astype(np.uint8)

        # Compute depth difference
        depth_threshold = 0.01  # 0.01
        # static_depth_mask = (np.abs(p_dep - c_dep) < depth_threshold).astype(np.uint8)
        dynamic_depth_mask = (np.abs(p_dep - c_dep) >= depth_threshold).astype(np.uint8)

        # Get the new mask
        mask_new = np.copy(c_mask)
        mask_new = np.multiply(mask_new, dynamic_flow_mask)
        mask_new = np.multiply(mask_new, dynamic_depth_mask)
        kernel = np.ones((5, 5), np.uint8)
        mask_new = cv2.morphologyEx(mask_new, cv2.MORPH_OPEN, kernel)
        mask_new = cv2.morphologyEx(mask_new, cv2.MORPH_CLOSE, kernel)

        # Get all connected components in the mask
        num_labels, labels_im = cv2.connectedComponents(mask_new)
        segments = {}
        for u in range(labels_im.shape[0]):
            for v in range(labels_im.shape[1]):
                l = labels_im[u, v]
                if l > 0:
                    if l not in segments:
                        segments[l] = []
                    segments[l].append((u, v))

        # Remove any small segments from the object mask (likely to be incorrect segmentation of the hand)
        for s in segments:
            pixels = segments[s]
            # Segment must have more than 50 pixels
            if len(pixels) < 100:
                for u, v in pixels:
                    mask_new[u, v] = 0

        if self.args.visualize:
            width = int(320)
            height = int(240)
            gray = cv2.cvtColor(c_rgb, cv2.COLOR_BGR2GRAY)
            img_output = np.hstack((cv2.resize(gray, (width, height)), cv2.resize(c_mask, (width, height))))
            img_output = np.hstack((img_output, cv2.resize(mask_new, (width, height))))
            img_output = np.hstack((img_output, cv2.resize(dynamic_flow_mask, (width, height)) * 255))
            img_output = np.hstack((img_output, cv2.resize(dynamic_depth_mask, (width, height)) * 255))

            cv2.imshow('Flow', img_output)
            cv2.waitKey(0)

        return mask_new

    @staticmethod
    def dense_flow(p_rgb, c_rgb):
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(p_rgb, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(c_rgb, cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow


class PersonRemover:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = args.split

    def process(self):
        # Get all sequence names
        root_dir = os.path.join(self.base_dir, self.data_split)
        dirs = [o for o in os.listdir(root_dir) if os.path.isdir("{}/{}".format(root_dir, o))]
        dirs.sort()

        for seq in dirs:
            # Directory where the data for this sequence is
            seq_dir = os.path.join(root_dir, seq)
            print('Processing {}'.format(seq_dir))
            # Create the output directory
            os.makedirs("{}/{}".format(seq_dir, self.args.output_dir), exist_ok=True)
            # Get all filenames
            frames = [f.split('.')[0] for f in os.listdir("{}/rgb".format(seq_dir))
                      if os.path.splitext(os.path.join(root_dir, seq, f))[1] == '.png']
            frames.sort()

            # For each frame in the sequence
            for fid in frames:
                rgb, mask, mask_hand, mask_person = self.load_data(seq, fid)
                try:
                    mask_p2p = self.update_mask(mask, mask_hand, mask_person)
                except:
                    print('Error processing file {}/{}'.format(seq_dir, fid))
                    mask_p2p = mask

                # If the mask is completely empty (i.e. not detection), keep the original
                if np.sum(mask_p2p) == 0:
                    print('No detection in {}/{}'.format(seq_dir, fid))
                    mask_p2p = mask

                if self.args.visualize:
                    color = [1, 0, 0]
                    rgb_mask = apply_mask(np.copy(rgb), mask, color, alpha=0.5)
                    rgb_mask_person = apply_mask(np.copy(rgb), mask_person, color, alpha=0.5)
                    rgb_mask_p2p = apply_mask(np.copy(rgb), mask_p2p, color, alpha=0.5)
                    img_output = np.hstack((rgb_mask, rgb_mask_person))
                    img_output = np.hstack((img_output, rgb_mask_p2p))

                    cv2.imshow('Cleaned', img_output)
                    cv2.waitKey(0)

                if self.args.save:
                    cv2.imwrite("{}/{}/{}.png".format(seq_dir, self.args.output_dir, fid), mask_p2p.astype(np.uint8))

    def load_data(self, seq_name, frame_id):
        # RGB
        rgb_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'rgb', frame_id + '.png')
        rgb = cv2.imread(rgb_filename)

        # Mask
        mask_filename = os.path.join(self.base_dir, self.data_split, seq_name, self.args.mask_dir, frame_id + '.png')
        mask = cv2.imread(mask_filename)[:, :, 0]

        # Mask hand
        mask_hand_filename = os.path.join(self.base_dir, self.data_split, seq_name, self.args.mask_hand_dir,
                                          frame_id + '.png')
        mask_hand = cv2.imread(mask_hand_filename)[:, :, 0]

        # Mask person
        mask_person_filename = os.path.join(self.base_dir, self.data_split, seq_name, self.args.mask_person_dir,
                                            frame_id + '.png')
        mask_person = cv2.imread(mask_person_filename)[:, :, 0]

        return rgb, mask, mask_hand, mask_person

    def update_mask(self, mask, mask_hand, mask_person):
        # Inflate the part of the person that does not correspond to the hand
        dilation_kernel = np.ones((20, 20), np.uint8)
        mask_hand = cv2.dilate(mask_hand, dilation_kernel, iterations=1)
        mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_CLOSE, dilation_kernel)
        hand_pixels = np.argwhere(mask_hand != 0)
        left = hand_pixels[:, 0].min()
        right = hand_pixels[:, 0].max()
        bottom = hand_pixels[:, 1].min()
        top = hand_pixels[:, 1].max()

        mask_hand_bb = np.zeros_like(mask_hand)
        mask_person_no_hand = np.copy(mask_person)
        xv, yv = np.meshgrid(range(max(left - 5, 0), min(right + 5, mask.shape[0] - 1)),
                             range(max(bottom - 5, 0), min(top + 5, mask.shape[1] - 1)))
        for u, v in zip(xv.flatten(), yv.flatten()):
            mask_person_no_hand[u, v] = 0
            mask_hand_bb[u, v] = 255

        #mask_person_no_hand = cv2.dilate(mask_person_no_hand, dilation_kernel, iterations=1)
        #mask_person_no_hand = cv2.morphologyEx(mask_person_no_hand, cv2.MORPH_CLOSE, dilation_kernel)
        person_pixels = np.argwhere(mask_person != 0)
        for u, v in person_pixels:
            mask_person_no_hand[u, v] = 255
        mask_person = mask_person_no_hand

        if self.args.debug:
            img_output = np.hstack((mask_hand, mask_person))
            img_output = np.hstack((img_output, mask_hand_bb))
            img_output = np.hstack((img_output, mask_person_no_hand))
            cv2.imshow('masks', img_output)
            cv2.waitKey(0)

        # Remove all pixels in mask that are labeled as person
        mask_new = np.copy(mask)
        mask_person_reverse = cv2.bitwise_not(mask_person) / 255
        mask_new = np.multiply(mask, mask_person_reverse)

        mask_new = (mask_new != 0).astype(np.uint8)

        # Get all connected components in the mask
        num_labels, labels_im = cv2.connectedComponents(mask_new)
        segments = {}
        for u in range(labels_im.shape[0]):
            for v in range(labels_im.shape[1]):
                l = labels_im[u, v]
                if l > 0:
                    if l not in segments:
                        segments[l] = []
                    segments[l].append((u, v))

        # Remove any segments near the left, right or top borders
        tolerance = int(0.25 * min(mask.shape))
        edge_padding = 10
        for s in segments:
            pixels = segments[s]
            coords_x, coords_y = zip(*pixels)
            coords_x = np.asarray(coords_x)
            coords_y = np.asarray(coords_y)
            left = coords_y.min()
            right = coords_y.max()
            bottom = mask.shape[0] - coords_x.max()
            top = mask.shape[0] - coords_x.min()
            if self.args.debug:
                print(' -- Segment --')
                print('left {} right {} bottom {} top {}'.format(left, right, bottom, top))

            # Conditions for removing the cluster
            remove_cluster = False
            # If number of pixels in the cluster is too small
            if len(pixels) < 500:
                remove_cluster = True
                if self.args.debug:
                    print('Too small')
            # If the object is completely in the left, right or top image regions
            elif right < tolerance or left > (mask.shape[1] - tolerance) or bottom > (mask.shape[0] - tolerance):
                remove_cluster = True
                if self.args.debug:
                    print('Completely left/right/top')
            # If the left, right or top touches the image border
            elif left < edge_padding or right > (mask.shape[1] - edge_padding) or top > (mask.shape[0] - edge_padding):
                remove_cluster = True
                if self.args.debug:
                    print('Touching left/right/top')
            else:
                if self.args.debug:
                    print('Ok!')

            if self.args.debug:
                mask_temp = np.zeros_like(mask)
                for u, v in pixels:
                    mask_temp[u, v] = 255

                cv2.imshow('mask temp', mask_temp)
                cv2.waitKey(0)

            # Remove this cluster
            if remove_cluster:
                for u, v in pixels:
                    mask_new[u, v] = 0
                    if self.args.debug:
                        mask_temp[u, v] = 255

        return mask_new * 255


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Optical flow estimation and segmentation')
    args = parser.parse_args()
    args.ho3d_path = HO3D_PATH
    args.split = SPLIT
    args.visualize = False
    args.save = True
    args.debug = False

    args.mode = CleanUpMode.MRCNN  # CleanUpMode.FLOW, CleanUpMode.MRCNN

    # Estimate optical flow and clean segmentation
    if args.mode == CleanUpMode.FLOW:
        # Set arguments
        args.output_dir = MASK_FLOW_DIR
        args.mask_dir = MASK_HAND_DIR

        # Use optical flow to clean up the masks
        ofe = OpticalFlowEstimator(args)
        ofe.process()
    # Clean up be removing pixels associated to the person detection
    elif args.mode == CleanUpMode.MRCNN:
        # Set arguments
        args.output_dir = MASK_P2P
        args.mask_dir = MASK_FLOW_DIR
        args.mask_hand_dir = 'hand-seg-tpv'
        args.mask_person_dir = MASK_PERSON_DIR

        pr = PersonRemover(args)
        pr.process()

