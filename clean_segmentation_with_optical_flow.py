import argparse
from utils.grasp_utils import *
import cv2
import imageio
import random


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
            # Directory where the date for this sequence is
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

            break

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

        '''
        # 𝚙𝚛𝚎𝚟(y, x) = 𝚗𝚎𝚡𝚝(y + 𝚏𝚕𝚘𝚠(y, x)[1], x + 𝚏𝚕𝚘𝚠(y, x)[0])
        flow_pairs = []
        flow_img = np.copy(c_rgb)
        for u in range(p_rgb.shape[0]):
            for v in range(p_rgb.shape[1]):
                if c_mask[u, v] > 0 and p_mask[u, v] > 0:
                    c_uv = (u, v)
                    p_uv = (int(u + flow[u, v][0]), int(v + flow[u, v][1]))
                    flow_pairs.append((p_uv, c_uv))
                    if random.uniform(0, 1) > 0.95:
                        flow_img = cv2.line(flow_img, (p_uv[1], p_uv[0]), (c_uv[1], c_uv[0]), (0, 255, 0), 1)
                        flow_img = cv2.circle(flow_img, (c_uv[1], c_uv[0]), 2, (0, 0, 255), -1)
        cv2.imshow('Flow', flow_img)
        cv2.waitKey(0)
        '''

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


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Optical flow estimation and segmentation')
    args = parser.parse_args()
    args.output_dir = 'masks_hs_dc_clean'
    args.mask_dir = 'masks_hs_dc'
    args.ho3d_path = '/home/tpatten/Data/bop_test/ho3dbop'
    args.split = 'test'
    args.visualize = True
    args.save = False

    # Estimate optical flow and clean segmentation
    ofe = OpticalFlowEstimator(args)
    ofe.process()
