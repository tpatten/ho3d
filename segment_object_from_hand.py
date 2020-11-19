import argparse
from utils.grasp_utils import *
import open3d as o3d
import cv2
import imageio
import pcl


class ObjectSegmenter:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = args.split

    def process(self):
        # Get all sequence names
        root_dir = os.path.join(self.base_dir, self.data_split)
        dirs = [o for o in os.listdir(root_dir) if os.path.isdir("{}/{}".format(root_dir, o))]
        dirs.sort()
        dirs = ['000003', '000008']

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
            f_count = 0
            for frame in frames:
                _, rgb_img, cloud, pts_to_pixels, hand_mask = self.load_data(seq, frame)
                object_indices = self.segment_frame(cloud, hand_mask)
                object_mask = self.points_to_mask(rgb_img, hand_mask, cloud, pts_to_pixels, object_indices)
                f_count += 1
                if (f_count % 25) == 0:
                    print(' -- {}'.format(f_count))
                if self.args.save:
                    cv2.imwrite("{}/{}/{}.png".format(seq_dir, self.args.output_dir, frame),
                                object_mask.astype(np.uint8))

    def load_data(self, seq_name, frame_id):
        # Depth
        depth_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'depth', frame_id + '.png')
        depth = imageio.imread(depth_filename).astype(np.float32) / 1000

        # RGB
        rgb_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'rgb', frame_id + '.png')
        rgb = cv2.imread(rgb_filename)

        # Annotation
        anno = {'camMat': np.asarray([[614.627, 0.0, 320.262],
                                      [0.0, 614.101, 238.469],
                                      [0.0, 0.0, 1.0]])}

        # Cloud
        points, img_coords = self.image_to_world(depth, anno)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points)
        scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        #if self.args.visualize:
        #    o3d.visualization.draw_geometries([scene_pcd])

        # Mask
        erosion_kernel = np.ones((8, 8), np.uint8)
        hand_mask_filename = os.path.join(self.base_dir, self.data_split, seq_name, 'hand-seg-tpv', frame_id + '.png')
        hand_mask = cv2.imread(hand_mask_filename)[:, :, 0]
        hand_mask = cv2.dilate(hand_mask, erosion_kernel, iterations=1)

        return depth, rgb, scene_pcd, img_coords, hand_mask

    @staticmethod
    def image_to_world(depth, anno, z_cutoff=1.0):
        i_fx = 1. / anno['camMat'][0, 0]
        i_fy = 1. / anno['camMat'][1, 1]
        cx = anno['camMat'][0, 2]
        cy = anno['camMat'][1, 2]

        pts = []
        img_coords = []
        for v in range(depth.shape[0]):
            for u in range(depth.shape[1]):
                # Get x, y, z values
                z = depth[v, u]
                x = (u - cx) * z * i_fx
                y = (v - cy) * z * i_fy
                # Set all background points to have coordinate (0, 0, 0)
                if z > z_cutoff:
                    x = y = z = 0
                pts.append([x, y, z])
                img_coords.append((v, u))

        pts = np.asarray(pts)

        return pts, img_coords

    def segment_frame(self, cloud, hand_mask):
        # Color the hand points
        #colors = np.asarray(cloud.colors)
        hand_indices = set(np.where((hand_mask > 0).flatten())[0])
        #for h in hand_indices:
        #    colors[h] = [236.0/255, 188.0/255, 180.0/255]

        # Create kd tree with the hand coordinates
        points = np.asarray(cloud.points).astype(np.float32)
        if len(hand_indices) > 0:
            hand_pts = np.zeros((len(hand_indices), 3))
            counter = 0
            for h in hand_indices:
                hand_pts[counter] = points[h]
                counter += 1
            hand_pcd = o3d.geometry.PointCloud()
            hand_pcd.points = o3d.utility.Vector3dVector(hand_pts)
            hand_kd_tree = o3d.geometry.KDTreeFlann(hand_pcd)

        # Remove all background and hand points
        points_new = []
        points_to_cloud_map = []
        for i in range(points.shape[0]):
            if points[i][0] == 0 and points[i][1] == 0 and points[i][2] == 0:
                continue
            if i in hand_indices:
                continue
            points_new.append(points[i])
            points_to_cloud_map.append(i)
        points_new = np.asarray(points_new).astype(np.float32)

        # Euclidean clustering of the of the foreground clusters
        cloud_pcl = pcl.PointCloud()
        cloud_pcl.from_array(points_new)
        tree = cloud_pcl.make_kdtree()
        ec = cloud_pcl.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.002)
        ec.set_MinClusterSize(100)
        ec.set_MaxClusterSize(2500000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        '''
        import random
        for clust in cluster_indices:
            rgb = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            for c in clust:
                colors[points_to_cloud_map[c]] = rgb
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud])
        '''

        # Retain all clusters with more than 500 points
        cluster_indices_flat = []
        for clust in cluster_indices:
            if len(clust) > 500:
                c_indices = [points_to_cloud_map[c] for c in clust]
                if len(hand_indices) > 0:
                    d = self.dist_to_nearest_point(points, hand_kd_tree, c_indices)
                else:
                    d = 0
                if d < 0.005:
                    cluster_indices_flat.extend(c_indices)
                #else:
                #    print(d)

        '''
        cluster_indices_flat = []
        for clust in cluster_indices:
            cluster_indices_flat.extend([c for c in clust])
        cluster_indices = cluster_indices_flat
        colors = np.asarray(cloud.colors)
        rgb = [0.1, 0.8, 0.1]
        for c in cluster_indices:
            colors[c] = rgb

        cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([cloud])
        '''

        return cluster_indices_flat

    @staticmethod
    def dist_to_nearest_point(points, kdtree, search_indices):
        min_d = 1000
        for i in search_indices:
            _, _, d = kdtree.search_knn_vector_3d(points[i], 1)
            d = d.pop()
            if d < min_d:
                min_d = d

        return min_d

    def points_to_mask(self, rgb, hand, cloud, img_coords, object_indices):
        # Convert the point indices to pixel values
        mask = np.zeros((rgb.shape[0], rgb.shape[1]))
        for i in object_indices:
            u, v = img_coords[i]
            mask[u, v] = 255
        mask = mask.astype(np.uint8)

        # Get all connected components in the mask
        num_labels, labels_im = cv2.connectedComponents(mask)
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
            # Segment must have more than 100 pixels
            if len(pixels) < 100:
                for u, v in pixels:
                    mask[u, v] = 0

        if self.args.visualize:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            img_output = np.hstack((gray, hand))
            img_output = np.hstack((img_output, mask))

            cv2.imshow('Mask', img_output)
            cv2.waitKey(0)

        '''
        colors = np.asarray(cloud.colors)
        rgb = [0.1, 0.8, 0.1]
        for i in object_indices:
            colors[i] = rgb

        cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([cloud])
        '''

        return mask


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Object segmentation')
    args = parser.parse_args()
    args.output_dir = 'mask_hsdc'
    args.ho3d_path = '/home/tpatten/Data/bop_test/ho3d'
    args.split = 'train'
    args.visualize = False
    args.save = True

    # Segment the objects
    seg = ObjectSegmenter(args)
    seg.process()
