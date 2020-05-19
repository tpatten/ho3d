# examples/Python/Advanced/multiway_registration.py

import os
import open3d as o3d
import numpy as np
from utils.grasp_utils import *
from pykdtree.kdtree import KDTree
import cv2
from itertools import combinations
import copy

voxel_size = 0.001
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

base_dir = '/home/tpatten/Data/Hands/HO3D/'
data_split = 'train'
scene = 'ABF10'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'
mask_erosion_kernel = 5
outlier_rm_nb_neighbors = 500
outlier_rm_std_ratio = 0.001
post_proc_voxel_radius = 0.0002
post_proc_inlier_radius = post_proc_voxel_radius * 2.5
feature_alignment = True
icp_refinement = True
start = 50
skip = 10
max_num = 10  # 4


def load_point_clouds(voxel_size=0.0):
    frame_ids = sorted(os.listdir(os.path.join(base_dir, data_split, scene, 'rgb')))
    pcds = []
    pcds_down = []
    rgbs = []
    depths = []
    masks = []
    pixel_to_pt_maps = []
    counter = start
    num_processed = 0
    while counter < len(frame_ids):
        frame_id = frame_ids[counter].split('.')[0]

        # Read image, depths maps and annotations
        rgb, depth, anno, scene_pcd = load_data(scene, frame_id)

        # Read the mask
        mask_filename = os.path.join(base_dir, data_split, scene, 'mask',
                                     OBJECT_MASK_VISIBLE_DIR, str(frame_id) + '.png')
        mask = cv2.imread(mask_filename)[:, :, 0]
        mask = cv2.erode(mask, np.ones((mask_erosion_kernel, mask_erosion_kernel), np.uint8), iterations=1)

        # Extract the masked point cloud
        cloud, colors, pixel_to_pt = image_to_world(anno, rgb, depth, mask, cut_z=np.linalg.norm(anno['objTrans']) * 1.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        #pcd = remove_outliers(pcd)
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
        # pcd = remove_outliers(pcd)
        pcd_down = remove_outliers(pcd_down)
        radius_normal = voxel_size * 2
        o3d.geometry.estimate_normals(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pcds.append(pcd)
        pcds_down.append(pcd_down)
        rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)
        pixel_to_pt_maps.append(pixel_to_pt)

        # Increment counters
        counter += skip
        num_processed += 1

        if max_num > 0 and num_processed == max_num:
            break

    return pcds, pcds_down, rgbs, depths, masks, pixel_to_pt_maps


def load_data(seq_name, frame_id):
    rgb = read_RGB_img(base_dir, seq_name, frame_id, data_split)
    depth = read_depth_img(base_dir, seq_name, frame_id, data_split)
    anno = read_annotation(base_dir, seq_name, frame_id, data_split)

    # Create a cloud from the rgb and depth images
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_o3d, depth_o3d)
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    K = anno['camMat']
    cam_intrinsics.set_intrinsics(640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    scene_pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, cam_intrinsics)
    scene_pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points) * 1000)
    # scene_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return rgb, depth, anno, scene_pcd


def image_to_world(anno, rgb, depth, mask=None, cut_z=1000.):
    i_fx = 1. / anno['camMat'][0, 0]
    i_fy = 1. / anno['camMat'][1, 1]
    cx = anno['camMat'][0, 2]
    cy = anno['camMat'][1, 2]

    pts = []
    colors = []
    pixel_to_pt = np.ones((rgb.shape[0], rgb.shape[1])) * -1
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            if mask is None or mask[v, u] > 0:
                z = depth[v, u]
                if z > 0.001 and z < cut_z:
                    x = (u - cx) * z * i_fx
                    y = (v - cy) * z * i_fy
                    pts.append([x, y, z])
                    colors.append(rgb[v, u])
                    pixel_to_pt[v, u] = len(pts) - 1

    pts = np.asarray(pts)
    colors = np.asarray(colors) / 255

    return pts, colors, pixel_to_pt.astype(int)


def remove_outliers(cloud):
    in_cloud, _ = o3d.geometry.statistical_outlier_removal(cloud,
                                                           nb_neighbors=outlier_rm_nb_neighbors,
                                                           std_ratio=outlier_rm_std_ratio)
    return in_cloud


def icp_registration(source, target, transform_init=None):
    print("Apply point-to-plane ICP")

    if transform_init is None:
        icp_coarse = o3d.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.eye(4),
            o3d.registration.TransformationEstimationPointToPlane())
        transform_init = icp_coarse.transformation

    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        transform_init,
        o3d.registration.TransformationEstimationPointToPlane())

    transformation_icp = icp_fine.transformation

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)

    return transformation_icp, information_icp


def full_registration(pcds, pcds_down, rgbs, masks, pixel_to_pt_maps,
                      max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            # Register the frames
            # Coarse registration
            if feature_alignment:
                transformation_icp, information_icp = feature_registration(
                    (rgbs[source_id], pcds[source_id], masks[source_id], pixel_to_pt_maps[source_id]),
                    (rgbs[target_id], pcds[target_id], masks[target_id], pixel_to_pt_maps[target_id]))
                if icp_refinement:
                    # Fine registration with ICP
                    transformation_icp, information_icp = icp_registration(
                        pcds_down[source_id], pcds_down[target_id], transformation_icp)
            else:
                # Coarse and fine registration with ICP
                transformation_icp, information_icp = icp_registration(
                    pcds_down[source_id], pcds_down[target_id], None)

            # Build pose graph
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


def post_process(originals, voxel_Radius, inlier_Radius):
    """
   Merge segments so that new points will not be add to the merged
   model if within voxel_Radius to the existing points, and keep a vote
   for if the point is issolated outside the radius of inlier_Radius at
   the timeof the merge

   Parameters
   ----------
   originals : List of open3d.Pointcloud classe
     6D pontcloud of the segments transformed into the world frame
   voxel_Radius : float
     Reject duplicate point if the new point lies within the voxel radius
     of the existing point
   inlier_Radius : float
     Point considered an outlier if more than inlier_Radius away from any
     other points

   Returns
   ----------
   points : (n,3) float
     The (x,y,z) of the processed and filtered pointcloud
   colors : (n,3) float
     The (r,g,b) color information corresponding to the points
   vote : (n, ) int
     The number of vote (seen duplicate points within the voxel_radius) each
     processed point has reveived
   """

    for point_id in range(len(originals)):

        if point_id == 0:
            vote = np.zeros(len(originals[point_id].points))
            points = np.array(originals[point_id].points, dtype=np.float64)
            colors = np.array(originals[point_id].colors, dtype=np.float64)
        else:
            points_temp = np.array(originals[point_id].points, dtype=np.float64)
            colors_temp = np.array(originals[point_id].colors, dtype=np.float64)

            dist, index = nearest_neighbour(points_temp, points)
            new_points = np.where(dist > voxel_Radius)
            points_temp = points_temp[new_points]
            colors_temp = colors_temp[new_points]
            inliers = np.where(dist < inlier_Radius)
            vote[(index[inliers],)] += 1
            vote = np.concatenate([vote, np.zeros(len(points_temp))])
            points = np.concatenate([points, points_temp])
            colors = np.concatenate([colors, colors_temp])

    return points, colors, vote


def nearest_neighbour(a, b):
    """
    find the nearest neighbours of a in b using KDTree
    Parameters
    ----------
    a : (n, ) numpy.ndarray
    b : (n, ) numpy.ndarray

    Returns
    ----------
    dist : n float
      Euclidian distance of the closest neighbour in b to a
    index : n float
      The index of the closest neighbour in b to a in terms of Euclidian distance
    """
    tree = KDTree(b)
    dist, index = tree.query(a)
    return dist, index


def feature_registration(source, target, MIN_MATCH_COUNT=9):  #12
    """
    Obtain the rigid transformation from source to target
    first find correspondence of color images by performing fast registration
    using SIFT features on color images.
    The corresponding depth values of the matching keypoints is then used to
    obtain rigid transformation through a ransac process.


    Parameters
    ----------
    source : ((n,m) uint8, (n,m) float)
      The source color image and the corresponding 3d pointcloud combined in a list
    target : ((n,m) uint8, (n,m) float)
      The target color image and the corresponding 3d pointcloud combined in a list
    MIN_MATCH_COUNT : int
      The minimum number of good corresponding feature points for the algorithm  to
      trust the pairwise registration result with feature matching only

    Returns
    ----------
    transform: (4,4) float or None
      The homogeneous rigid transformation that transforms source to the target's
      frame
      if None, registration result using feature matching only cannot be trusted
      either due to no enough good matching feature points are found, or the ransac
      process does not return a solution

    """
    print("Apply SIFT registration")

    rgb_src, pts_src, mask_src, ppmap_src = source
    rgb_des, pts_des, mask_des, ppmap_des = target

    # pts_des.points = o3d.utility.Vector3dVector(np.asarray(pts_des.points) + 0.2)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(rgb_src, None)
    kp2, des2 = sift.detectAndCompute(rgb_des, None)

    # Find good mathces
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # If number of good matching feature point is less than the MIN_MATCH_COUNT
    if len(good) < MIN_MATCH_COUNT:
        print('Not enough matches: {}'.format(len(good)))
        return None, None

    print('Number good matches: {}'.format(len(good)))
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    '''
    # Visualize matches in the image
    h = rgb_src.shape[0]
    w = rgb_src.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(rgb_des, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(rgb_src, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()
    '''

    bad_match_index = np.where(np.array(matchesMask) == 0)

    src_index = np.vstack(src_pts).squeeze()
    src_index = np.delete(src_index, tuple(bad_match_index[0]), axis=0)
    src_index[:, [0, 1]] = src_index[:, [1, 0]]
    src_index = tuple(src_index.T.astype(np.int32))

    dst_index = np.vstack(dst_pts).squeeze()
    dst_index = np.delete(dst_index, tuple(bad_match_index[0]), axis=0)
    dst_index[:, [0, 1]] = dst_index[:, [1, 0]]
    dst_index = tuple(dst_index.T.astype(np.int32))

    src_good = []
    dst_good = []
    for i in range(len(src_index[0])):
        u_s = src_index[0][i]
        v_s = src_index[1][i]
        u_d = dst_index[0][i]
        v_d = dst_index[1][i]
        if ppmap_src[u_s, v_s] > 0 and ppmap_des[u_d, v_d] > 0:
            pt_s = np.asarray(pts_src.points)[ppmap_src[u_s, v_s]]
            pt_d = np.asarray(pts_des.points)[ppmap_des[u_s, v_d]]
            src_good.append(pt_s)
            dst_good.append(pt_d)

    # Get rigid transforms between two sets of feature points through RANSAC
    transform = match_ransac(np.asarray(src_good), np.asarray(dst_good), tol=100)
    print(np.asarray(transform).reshape((4, 4)))

    if transform is None:
        print('RANSAC returned None')
        return None, None

    #'''
    # Visualize the matches in 3D
    print('3D SIFT visualization')

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pts1 = copy.deepcopy(pts_src)
    pts1.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(pts1)

    pts2 = copy.deepcopy(pts_des)
    # pts2.points = o3d.utility.Vector3dVector(np.asarray(pts2.points) + 0.2)
    pts2.paint_uniform_color([0.0, 0.0, 1.0])
    vis.add_geometry(pts2)

    pts_tf = copy.deepcopy(pts_src)
    pts_tf.transform(transform)
    pts_tf.paint_uniform_color([0.0, 1.0, 0.0])
    vis.add_geometry(pts_tf)

    points = []
    lines = []
    line_colors = []
    for i in range(len(src_index[0])):
        u_s = src_index[0][i]
        v_s = src_index[1][i]
        u_d = dst_index[0][i]
        v_d = dst_index[1][i]
        if ppmap_src[u_s, v_s] > 0 and ppmap_des[u_d, v_d] > 0:
            pt_s = np.asarray(pts1.points)[ppmap_src[u_s, v_s]]
            pt_d = np.asarray(pts2.points)[ppmap_des[u_s, v_d]]
            points.append(pt_s)
            points.append(pt_d)
            lines.append([len(points) - 1, len(points) - 2])
            line_colors.append([0.0, 1.0, 0.0])

            mm = o3d.create_mesh_sphere(radius=0.0025)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([1., 0., 1.])
            tt = np.eye(4)
            tt[0:3, 3] = pt_s
            mm.transform(tt)
            vis.add_geometry(mm)

            mm = o3d.create_mesh_sphere(radius=0.0025)
            mm.compute_vertex_normals()
            mm.paint_uniform_color([0., 1., 1.])
            tt = np.eye(4)
            tt[0:3, 3] = pt_d
            mm.transform(tt)
            vis.add_geometry(mm)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    vis.add_geometry(line_set)

    # End visualizer
    vis.run()
    vis.destroy_window()
    #'''


    # Return transform
    return transform, o3d.registration.get_information_matrix_from_point_clouds(
        pts_src, pts_des, max_correspondence_distance_fine, np.array(transform))


def match_ransac(p, p_prime, tol=0.001):
    """
    A ransac process that estimates the transform between two set of points
    p and p_prime.
    The transform is returned if the RMSE of the smallest 70% is smaller
    than the tol.

    Parameters
    ----------
    p : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    target : (n,3) float
      The target 3d pointcloud as a numpy.ndarray
    tol : float
      A transform is considered found if the smallest 70% RMSE error between the
      transformed p to p_prime is smaller than the tol

    Returns
    ----------
    transform: (4,4) float or None
      The homogeneous rigid transformation that transforms p to the p_prime's
      frame
      if None, the ransac does not find a sufficiently good solution

    """

    leastError = None
    R = None
    t = None
    # the smallest 70% of the error is used to compute RMSE
    k = int(len(p) * 0.7)
    assert len(p) == len(p_prime)
    for comb in combinations(range(0, len(p)), 3):

        index = (np.array([comb[0], comb[1], comb[2]]),)
        R_temp, t_temp = rigid_transform_3D(p[index], p_prime[index])
        R_temp = np.array(R_temp)
        t_temp = (np.array(t_temp).T)[0]
        transformed = (np.dot(R_temp, p.T).T) + t_temp
        error = (transformed - p_prime) ** 2
        error = np.sum(error, axis=1)
        error = np.sqrt(error)

        RMSE = np.sum(error[np.argpartition(error, k)[:k]]) / k

        print('RMSE', RMSE)

        if RMSE < tol:
            R = R_temp
            t = t_temp

            transform = [[R[0][0], R[0][1], R[0][2], t[0]],
                         [R[1][0], R[1][1], R[1][2], t[1]],
                         [R[2][0], R[2][1], R[2][2], t[2]],
                         [0, 0, 0, 1]]
            return transform

        return None


def rigid_transform_3D(A, B):
    """
    Estimate a rigid transform between 2 set of points of equal length
    through singular value decomposition(svd), return a rotation and a
    transformation matrix

    Parameters
    ----------
    A : (n,3) float
      The source 3d pointcloud as a numpy.ndarray
    B : (n,3) float
      The target 3d pointcloud as a numpy.ndarray

    Returns
    ----------
    R: (3,3) float
      A rigid rotation matrix
    t: (3) float
      A translation vector

    """

    assert len(A) == len(B)
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = AA.T * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return (R, t)


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds, pcds_down, rgbs, depths, masks, pixel_to_pt_maps = load_point_clouds(voxel_size)

    o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds, pcds_down, rgbs, masks, pixel_to_pt_maps,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down)

    print("Make a combined point cloud")
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = o3d.geometry.voxel_down_sample(pcd_combined, voxel_size)
    # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])

    pcds_processed, colors_processed, _ = post_process(pcds, post_proc_voxel_radius, post_proc_inlier_radius)
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(pcds_processed)
    pcd_combined.colors = o3d.utility.Vector3dVector(colors_processed)
    o3d.visualization.draw_geometries([pcd_combined])
