# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import argparse
from utils.grasp_utils import *
import cv2
import open3d as o3d
import copy
import transforms3d as tf3d
from enum import IntEnum
import json
from scipy.spatial.transform.rotation import Rotation
import pcl


OBJECT_MASK_VISIBLE_DIR = 'object_vis'
COORD_CHANGE_MAT = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
ICP_THRESH = 0.02
YCB_MODELS_DIR = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/'
bop_dir = '/home/tpatten/Data/bop/ho3d/'
ho3d_to_bop = {'ABF10': '000001', 'ABF11': '000002', 'ABF12': '000003', 'BB12': '000008', 'GPMF12': '000013',
               'GSF12': '000018', 'MC1': '000021', 'MC4': '000023', 'MDF12': '000028', 'SB12': '000033',
               'SM2': '000035', 'SM3': '000036', 'SMu1': '000039', 'SMu40': '000040', 'SS1': '000043',
               'ShSu12': '000047', 'SiBF12': '000052'}


class ICPMethod(IntEnum):
    Point2Point = 1
    Point2Plane = 2


class RegMethod(IntEnum):
    GT = 1
    GT_ICP = 2
    GT_ICP_FULL = 4


class ModelReconstructor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.ho3d_path
        self.data_split = 'train'
        self.mask_dir = args.mask_dir
        self.rgb = None
        self.depth = None
        self.anno = None
        self.scene_pcd = None
        self.prev_cloud = None
        self.prev_pose = np.eye(4)
        self.erosion_kernel = {'obj': None, 'hand': None}
        if args.mask_erosion_kernel[0] > 0:
            self.erosion_kernel['obj'] = np.ones((args.mask_erosion_kernel[0], args.mask_erosion_kernel[0]), np.uint8)
        if args.mask_erosion_kernel[1] > 0:
            self.erosion_kernel['hand'] = np.ones((args.mask_erosion_kernel[1], args.mask_erosion_kernel[1]), np.uint8)
        self.volume = None
        self.pose_annotation_global_rot = np.eye(3)
        self.pose_annotation_global_tra = np.asarray([0., 0., 0.])

        # Load the rendering scores file
        if os.path.isfile(self.args.hand_obj_rendering_scores):
            with open(self.args.hand_obj_rendering_scores) as json_file:
                self.hand_obj_rendering_scores = json.load(json_file)
        else:
            self.hand_obj_rendering_scores = None

        # Load models in the renderer
        self.renderer = None
        if self.args.inpaint_with_rendering:
            from bop_toolkit_lib import renderer
            width = 640
            height = 480
            renderer_modalities = []
            renderer_modalities.append('rgb')
            renderer_modalities.append('depth')
            renderer_mode = '+'.join(renderer_modalities)
            self.renderer = renderer.create_renderer(width, height, 'python', mode=renderer_mode, shading='flat')
            self.add_objects_to_renderer()

        # Reconstruct the model
        loaded_pcd = None
        if self.args.model_file == '':
            loaded_pcd = self.reconstruct_object_model()

        if self.args.visualize:
            if not self.args.model_file == '':
                loaded_pcd = self.load_object_model(self.args.model_file)

                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                #                             outlier_rm_std_ratio=self.args.outlier_rm_std_ratio * 0.0001,
                #                             raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                #                             raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)
                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                #                             raduis_rm_min_nb_points=1250,
                #                             raduis_rm_radius=self.args.voxel_size * 10)

                #loaded_pcd = remove_outliers(loaded_pcd, outlier_rm_nb_neighbors=0.,
                #                             outlier_rm_std_ratio=0.,
                #                             raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                #                             raduis_rm_radius=self.args.voxel_size * 5)

            loaded_pcd = remove_outliers(loaded_pcd,
                                         outlier_rm_nb_neighbors=250,
                                         outlier_rm_std_ratio=0.01,
                                         raduis_rm_min_nb_points=0, raduis_rm_radius=0)

            print('Visualizing loaded pcd...')
            self.visualize(mask_pcds=[loaded_pcd])
            # self.visualize(mask_pcds=[self.load_object_model(self.args.model_file)])

            if self.args.save and loaded_pcd is not None:
                loaded_pcd = o3d.geometry.voxel_down_sample(loaded_pcd, 0.001)
                o3d.geometry.estimate_normals(loaded_pcd,
                                              search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                  radius=0.1, max_nn=30))
                points = np.asarray(loaded_pcd.points)
                normals = np.asarray(loaded_pcd.normals)

                if self.args.model_file == '':
                    filename = os.path.join(self.base_dir, 'reconstructions',
                                            self.args.scene + '_' + str(self.args.reg_method).split('.')[1])
                    if self.args.reg_method == RegMethod.GT_ICP or self.args.reg_method == RegMethod.GT_ICP_FULL:
                        filename += '_' + str(self.args.icp_method).split('.')[1]
                    filename += '_start' + str(self.args.start_frame) + '_max' + str(self.args.max_num) + \
                                '_skip' + str(self.args.skip)
                    #if self.mask_dir != OBJECT_MASK_VISIBLE_DIR:
                    #    filename += '_segHO3D'
                    if self.hand_obj_rendering_scores is not None:
                        filename += '_renFilter'

                    if self.args.viewpoint_file != '':
                        filename = os.path.join(self.base_dir, 'reconstructions',
                                                self.args.scene + '_' + str(self.args.viewpoint_file).split('.')[0])

                    if self.args.pose_annotation_file != '':
                        filename = os.path.join(self.base_dir, 'reconstructions', self.args.scene + '_AnnoPoses')

                    if self.args.apply_inpainting:
                        filename += '_inPaint'
                    if self.args.inpaint_with_rendering:
                        filename += 'Rend'

                    filename += '_clean.xyz'
                else:
                    filename = self.args.model_file.split('.')[0]
                    #if self.mask_dir != OBJECT_MASK_VISIBLE_DIR:
                    #    filename += '_segHO3D'
                    filename += '_clean.xyz'
                print('Saving to: {}'.format(filename))
                #f = open(filename, "w")
                #for i in range(len(points)):
                #    f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
                #                                         normals[i, 0], normals[i, 1], normals[i, 2]))
                #f.close()

                #print('Saving to: {}'.format(filename.replace('.xyz', '.ply')))
                #o3d.io.write_point_cloud(filename.replace('.xyz', '.ply'), loaded_pcd)

    def reconstruct_object_model(self):
        # If the input scene contains digit identifier, then it is the only subdir
        if ''.join([i for i in self.args.scene if not i.isdigit()]) == self.args.scene:
            # Get directories for this scene
            sub_dirs = sorted(os.listdir(os.path.join(self.base_dir, self.data_split)))
            sub_dirs = [s for s in sub_dirs if ''.join([i for i in s if not i.isdigit()]) == self.args.scene and
                        os.path.isdir(os.path.join(self.base_dir, self.data_split, s))]
            self.args.skip = 50
        else:
            sub_dirs = [self.args.scene]

        cam_poses = []
        est_cam_poses = []
        mask_pcds = []
        scene_pcds = []
        processed_frames = []
        for s in sub_dirs:
            s_cam_poses, s_est_cam_poses, s_mask_pcds, s_scene_pcds, s_processed_frames = self.reconstruct_from_scene(s)
            if len(sub_dirs) < 2:
                cam_poses.extend(s_cam_poses)
                est_cam_poses.extend(s_est_cam_poses)
                mask_pcds.extend(s_mask_pcds)
                scene_pcds.extend(s_scene_pcds)
                processed_frames.extend(s_processed_frames)

        # Get the mesh from the TSDF
        tsdf_mesh = None
        if self.args.construct_tsdf:
            tsdf_mesh = self.volume.extract_triangle_mesh()
            tsdf_mesh.compute_vertex_normals()

        # If alignment to CAD model is required
        if self.args.pose_annotation_file != '' and self.args.align_to_cad:
            tsdf_mesh_aligned = self.align_to_cad(tsdf_mesh)
        else:
            tsdf_mesh_aligned = None

        # Save
        combined_pcd = None
        if self.args.save:
            combined_pcd = self.save_clouds_and_camera_poses(self.base_dir, processed_frames, cam_poses, est_cam_poses,
                                                             mask_pcds, tsdf_mesh, mesh_aligned=tsdf_mesh_aligned)

        # Visualize
        if self.args.visualize:
            scene_pcds = None
            print('Visualizing aligned clouds...')
            if self.args.reg_method == RegMethod.GT:
                self.visualize(cam_poses, mask_pcds, scene_pcds=scene_pcds)
            else:
                self.visualize(est_cam_poses, mask_pcds, scene_pcds=scene_pcds, gt_poses=cam_poses)
            if tsdf_mesh_aligned is not None:
                print('Visualizing TSDF volume - ALIGNED...')
                self.visualize_volume(tsdf_mesh_aligned)
            else:
                print('Visualizing TSDF volume...')
                self.visualize_volume(tsdf_mesh)

        if combined_pcd is None:
            return self.combine_clouds(mask_pcds)
        else:
            return combined_pcd

    def reconstruct_from_scene(self, scene_id):
        # Get the scene_id in the order list of directories in order to extract the matching hand/object visibilities
        if self.hand_obj_rendering_scores is not None:
            sub_dirs = sorted([f for f in os.listdir(os.path.join(self.base_dir, self.data_split))
                               if os.path.isdir(os.path.join(self.base_dir, self.data_split, f))])
            bop_scene_id = str(sub_dirs.index(scene_id) + 1)
            print('BOP ID is {} from HO-3D ID {}'.format(bop_scene_id, scene_id))
            scene_rendering_scores = self.hand_obj_rendering_scores[bop_scene_id]
        else:
            scene_rendering_scores = None

        # Get the scene ids
        frame_ids = sorted(os.listdir(os.path.join(self.base_dir, self.data_split, scene_id, 'rgb')))

        # If using given viewpoints
        if self.args.viewpoint_file != '':
            filename = os.path.join(self.base_dir, self.data_split, scene_id, self.args.viewpoint_file)
            if os.path.isfile(filename):
                with open(filename) as json_file:
                    frame_ids = json.load(json_file)['frame_ids']
                frame_ids = sorted([f.zfill(4) + '.png' for f in frame_ids])
            self.args.skip = 1
            self.args.max_num = -1

        # If using annotated poses
        relative_poses = None
        if self.args.pose_annotation_file != '':
            filename = os.path.join(self.base_dir, self.data_split, scene_id, self.args.pose_annotation_file)
            if os.path.isfile(filename):
                with open(filename) as json_file:
                    relative_poses = json.load(json_file)
                frame_ids = relative_poses.keys()
                frame_ids = sorted([f.zfill(4) + '.png' for f in frame_ids])
            # Get the global transformation
            if self.args.align_to_cad:
                _, _, anno, _ = self.load_data(scene_id, frame_ids[0].split('.')[0])
                self.pose_annotation_global_rot = cv2.Rodrigues(anno['objRot'])[0].T
                self.pose_annotation_global_tra = anno['objTrans']

        # Load the camera parameters
        cam_params = self.read_camera_intrinsics(scene_id)

        # Initialize the TSDF volume
        if self.args.construct_tsdf and self.volume is None:
            self.volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=self.args.sdf_voxel_length,
                sdf_trunc=self.args.sdf_trunc,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        # For each frame in the directory
        cam_poses = []
        est_cam_poses = []
        mask_pcds = []
        scene_pcds = []
        processed_frames = []
        counter = self.args.start_frame
        num_processed = 0
        while counter < len(frame_ids):
            # Get the id
            frame_id = frame_ids[counter].split('.')[0]
            # Create the filename for the metadata file
            meta_filename = os.path.join(self.base_dir, self.data_split, scene_id, 'meta',
                                         str(frame_id) + '.pkl')
            if self.args.max_num <= 0:
                print('[{}/{}] Processing file {}'.format(counter, len(frame_ids), meta_filename))
            else:
                print('[{}/{}] Processing file {}'.format(num_processed, self.args.max_num, meta_filename))

            # Check this is a validly annotated frame
            if scene_rendering_scores is not None and args.viewpoint_file == '':
                frame_id_key = str(int(frame_id))
                # [0] num_rendered, [1] visible_of_rendered, [2] valid_of_rendered, [3] valid_of_visible
                # Number of pixels of the object must be more than min_num_pixels
                if scene_rendering_scores[frame_id_key]['objs_scores'][0][0] < self.args.min_num_pixels:
                    print('Too few visible pixels: {}'.format(
                        scene_rendering_scores[frame_id_key]['objs_scores'][0][0]))
                    counter += 1
                    continue
                # Ratio of visible (i.e., correctly rendered) object pixels must be above min_ratio_valid
                if scene_rendering_scores[frame_id_key]['objs_scores'][0][3] < self.args.min_ratio_valid:
                    print('Object has low validity of pixels: {}'.format(
                        scene_rendering_scores[frame_id_key]['objs_scores'][0][3]))
                    counter += 1
                    continue
                # Ratio of visible (i.e., correctly rendered) hand pixels must be above min_ratio_valid
                if scene_rendering_scores[frame_id_key]['hand_scores'][3] < self.args.min_ratio_valid:
                    print('Hand has low validity of pixels: {}'.format(
                        scene_rendering_scores[frame_id_key]['hand_scores'][3]))
                    counter += 1
                    continue
                # Ratio of visible (i.e., correctly rendered) scene pixels must be above min_ratio_valid
                if scene_rendering_scores[frame_id_key]['scores'][3] < self.args.min_ratio_valid:
                    print('Scene has low validity of pixels: {}'.format(
                        scene_rendering_scores[frame_id_key]['scores'][3]))
                    counter += 1
                    continue

            # Read image, depths maps and annotations
            self.rgb, self.depth, self.anno, scene_pcd = self.load_data(scene_id, frame_id)

            # Read the mask
            mask, mask_hand = self.load_mask(scene_id, frame_id)
            if mask is None:
                print('No mask available for frame {}'.format(frame_id))
                counter += self.args.skip
                continue

            masked_rgb = self.apply_mask(self.rgb, mask, mask_hand)
            masked_depth = self.apply_mask(self.depth, mask, mask_hand)

            # Inpaint the depth image
            if self.args.apply_inpainting:
                if self.args.inpaint_with_rendering:
                    # Render the depth from the model
                    rendered_depth = self.render_depth(cam_params, pose_in_frames=relative_poses,
                                                       frame_id=str(int(frame_id)))
                    # Fill the missing values with rendered depth values
                    filled_depth = self.fill_from_rendering(self.depth, masked_depth, rendered_depth, mask_hand)
                    d_diff = np.abs(rendered_depth - filled_depth)
                    d_diff[filled_depth == 0] = 0.0
                    for u in range(filled_depth.shape[0]):
                        for v in range(filled_depth.shape[1]):
                            if d_diff[u, v] >= 0.01:  # 0.005
                                filled_depth[u, v] = 0

                    if self.args.save:
                        # Save new mask file
                        new_mask = np.zeros_like(mask)
                        for u in range(filled_depth.shape[0]):
                            for v in range(filled_depth.shape[1]):
                                if filled_depth[u, v] != 0:
                                    new_mask[u, v] = 255
                        # Mask file
                        mask_filename = os.path.join(bop_dir, self.data_split, ho3d_to_bop[scene_id], self.mask_dir,
                                                     str(frame_id).zfill(6) + '_refined.png')
                        cv2.imwrite(mask_filename, new_mask)

                    # Visualize
                    if self.args.visualize_inpainting:
                        img_output = np.vstack((np.hstack((self.depth, masked_depth)),
                                                np.hstack((rendered_depth, filled_depth))))
                        cv2.imshow('Depth image', img_output)
                        cv2.waitKey(0)

                        # Visualize an image showing the difference between the filled depth and the rendered depth
                        rgb_overlay = np.copy(self.rgb)
                        for u in range(d_diff.shape[0]):
                            for v in range(d_diff.shape[1]):
                                if d_diff[u, v] > 0.005:
                                    #rgb_overlay[u, v] = [0, 0, d_diff[u, v]]
                                    rgb_overlay[u, v] = [255, 0, 0]

                        img_output = np.hstack((self.rgb, rgb_overlay))
                        cv2.imshow('Depth diff image', img_output)
                        cv2.waitKey(0)

                        new_mask = np.zeros_like(mask)
                        for u in range(filled_depth.shape[0]):
                            for v in range(filled_depth.shape[1]):
                                if filled_depth[u, v] > 0 and d_diff[u, v] < 0.005:
                                    new_mask[u, v] = 255

                        image1 = np.copy(self.rgb)
                        image2 = np.copy(self.rgb)
                        mask1 = mask > 0
                        mask2 = new_mask > 0
                        image1[np.invert(mask1)] = image1[np.invert(mask1)] * 0.3
                        image2[np.invert(mask2)] = image2[np.invert(mask2)] * 0.3

                        img_output = np.hstack((image1, image2))
                        cv2.imshow('Masks', img_output)
                        cv2.waitKey(0)

                        cv2.destroyAllWindows()

                    # Set the depth and remove the mask
                    self.depth = filled_depth
                    masked_depth = filled_depth
                    mask = None
                else:
                    inpainted_depth = self.get_inpainted_depth(masked_depth, mask, mask_hand)
                    masked_depth = inpainted_depth

            # Extract the masked point cloud
            cloud, colors = self.image_to_world(self.rgb, self.depth, mask,
                                                cut_z=np.linalg.norm(self.anno['objTrans']) * 1.1)

            if cloud.shape[0] == 0:
                print('Empty cloud for frame {}'.format(frame_id))
                counter += self.args.skip
                continue

            # Transform the cloud and get the camera position
            cloud, cam_pose, est_cloud, est_cam_pose = self.transform_to_object_frame(
                cloud, pose_in_frames=relative_poses, frame_id=str(int(frame_id)))

            # Create the point cloud for visualization
            mask_pcd = o3d.geometry.PointCloud()
            if self.args.reg_method == RegMethod.GT:
                mask_pcd.points = o3d.utility.Vector3dVector(cloud)
            else:
                mask_pcd.points = o3d.utility.Vector3dVector(est_cloud)
            mask_pcd.colors = o3d.utility.Vector3dVector(colors)
            # mask_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Remove outliers
            if not self.args.inpaint_with_rendering:
                mask_pcd = remove_outliers(mask_pcd,
                                           outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                                           outlier_rm_std_ratio=self.args.outlier_rm_std_ratio,
                                           raduis_rm_min_nb_points=self.args.raduis_rm_min_nb_points,
                                           raduis_rm_radius=self.args.voxel_size * self.args.raduis_rm_radius_factor)

                # mask_pcd = remove_outliers(mask_pcd,
                #                           outlier_rm_nb_neighbors=self.args.outlier_rm_nb_neighbors,
                #                           outlier_rm_std_ratio=self.args.outlier_rm_std_ratio)

            # Add to SDF volume
            if self.args.construct_tsdf:
                masked_rgb_tsdf = o3d.geometry.Image(masked_rgb)
                masked_depth_tsdf = o3d.geometry.Image((masked_depth * 1000).astype(np.float32))
                rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(
                    masked_rgb_tsdf, masked_depth_tsdf, depth_trunc=self.args.sdf_depth_trunc,
                    convert_rgb_to_intensity=False)
                self.volume.integrate(rgbd, cam_params, np.linalg.inv(cam_pose))

            # Add to lists
            cam_poses.append(cam_pose)
            est_cam_poses.append(est_cam_pose)
            mask_pcds.append(mask_pcd)
            scene_pcds.append(scene_pcd)
            processed_frames.append(frame_id)

            # Increment counters
            counter += self.args.skip
            num_processed += 1

            # Update the previous cloud and pose
            if self.args.reg_method == RegMethod.GT_ICP_FULL:
                if self.prev_cloud is None:
                    self.prev_cloud = mask_pcd
                else:
                    self.prev_cloud.points.extend(mask_pcd.points)
                    self.prev_cloud.colors.extend(mask_pcd.colors)
            else:
                self.prev_cloud = mask_pcd
            self.prev_pose = est_cam_pose

            # Down sample
            if self.args.voxel_size > 0:
                self.prev_cloud = o3d.geometry.voxel_down_sample(self.prev_cloud, self.args.voxel_size)

            # Exit if reached the limit
            if self.args.max_num > 0 and num_processed == self.args.max_num:
                break

            #if counter > 3:
            #    break

            ## Save intermediate results
            #if self.args.save and self.args.save_intermediate and num_processed % 20 == 0:
            #    base_dir = os.path.join(self.base_dir, self.data_split, scene_id)
            #    self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses, mask_pcds,
            #                                      frame_id=frame_id)

        print('Processed {} frames out of {}'.format(num_processed, len(frame_ids)))

        return cam_poses, est_cam_poses, mask_pcds, scene_pcds, processed_frames

    def load_data(self, seq_name, frame_id):
        rgb = read_RGB_img(self.base_dir, seq_name, frame_id, self.data_split)
        depth = read_depth_img(self.base_dir, seq_name, frame_id, self.data_split)
        anno = read_annotation(self.base_dir, seq_name, frame_id, self.data_split)

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

    def load_mask(self, scene_id, frame_id):
        mask = None
        hand = None

        mask_filename = os.path.join(bop_dir, self.data_split, ho3d_to_bop[scene_id], self.mask_dir,
                                     str(frame_id).zfill(6) + '.png')
        hand_filename = os.path.join(bop_dir, self.data_split, ho3d_to_bop[scene_id], 'hand-seg-tpv',
                                     str(frame_id).zfill(6) + '.png')

        if not os.path.exists(mask_filename):
            print('No mask file: {}'.format(mask_filename))
            return mask, hand

        if not os.path.exists(hand_filename):
            print('No hand mask file: {}'.format(hand_filename))
            return mask, hand

        # Load the mask file
        mask = cv2.imread(mask_filename)[:, :, 0]

        # Load the mask hand file
        hand = cv2.imread(hand_filename)[:, :, 0]

        # Apply erosion
        if self.erosion_kernel['obj'] is not None:
            mask = cv2.erode(mask, self.erosion_kernel['obj'], iterations=1)
        if self.erosion_kernel['hand'] is not None:
            hand = cv2.dilate(hand, self.erosion_kernel['hand'], iterations=1)

        return mask, hand

    def load_mask_ho3d(self, scene_id, frame_id):
        mask = None
        hand = None

        # Load the mask file
        mask_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2_segmentations_rendered/'
        mask_filename = os.path.join(mask_dir, self.data_split, scene_id, 'seg', str(frame_id) + '.jpg')

        if not os.path.exists(mask_filename):
            return mask, hand

        mask_rgb = cv2.imread(mask_filename)

        # Generate binary mask
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))
        hand = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))
        for u in range(mask_rgb.shape[0]):
            for v in range(mask_rgb.shape[1]):
                if mask_rgb[u, v, 0] > 230 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] < 10:
                    mask[u, v] = 255
                if mask_rgb[u, v, 0] < 10 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] > 230:
                    hand[u, v] = 255

        # Resize image to original size
        mask = cv2.resize(mask, (self.rgb.shape[1], self.rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        hand = cv2.resize(hand, (self.rgb.shape[1], self.rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply erosion
        if self.erosion_kernel['obj'] is not None:
            mask = cv2.erode(mask, self.erosion_kernel['obj'], iterations=1)
        if self.erosion_kernel['hand'] is not None:
            # hand = cv2.erode(hand, self.erosion_kernel['hand'], iterations=1)
            hand = cv2.dilate(hand, self.erosion_kernel['hand'], iterations=1)

        return mask, hand

    def render_depth(self, cam_intrinsics, pose_in_frames=None, frame_id=None):
        # Get the ground truth transformation
        do_inverse = False
        if pose_in_frames is None or frame_id not in pose_in_frames:
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(self.anno['objRot'])[0]
            pose[:3, 3] = self.anno['objTrans']
        else:
            pose = np.asarray(pose_in_frames[frame_id]).reshape((4, 4))
            do_inverse = True

        # From OpenGL coordinates
        rotx = np.eye(4)
        rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
        # pose = rotx @ pose
        pose = np.matmul(rotx, pose)

        if do_inverse:
            pose = np.linalg.inv(pose)

        # fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        fx, fy = cam_intrinsics.get_focal_length()
        cx, cy = cam_intrinsics.get_principal_point()
        rendering = self.renderer.render_object(self.anno['objName'], pose[:3, :3], pose[:3, 3], fx, fy, cx, cy)
        depth = rendering['depth']

        return depth

    def add_objects_to_renderer(self):
        # Get all the .ply files in the model directory
        model_dir_name = 'reconstructions_new'

        obj_ids = sorted([f for f in os.listdir(os.path.join(YCB_MODELS_DIR, model_dir_name))
                          if os.path.isdir(os.path.join(YCB_MODELS_DIR, model_dir_name, f))])

        for obj_id in obj_ids:
            model_path = os.path.join(os.path.join(YCB_MODELS_DIR, model_dir_name), obj_id, 'mesh_for_rendering.ply')
            print('model_path', model_path)
            self.renderer.add_object(obj_id.replace('.ply', ''), model_path, surf_color=None)

    def read_camera_intrinsics(self, seq_name):
        cam_params = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        cam_mat = np.zeros((3, 3))
        cam_mat[2, 2] = 1.

        f_name = os.path.join(self.base_dir, self.data_split, seq_name, 'meta/0000.pkl')
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations pickle file at %s. Aborting.' % f_name)
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)

        cam_mat[0, 0] = pickle_data['camMat'][0, 0]
        cam_mat[1, 1] = pickle_data['camMat'][1, 1]
        cam_mat[0, 2] = pickle_data['camMat'][0, 2]
        cam_mat[1, 2] = pickle_data['camMat'][1, 2]

        cam_params.set_intrinsics(640, 480, cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2])

        return cam_params

    @staticmethod
    def fill_from_rendering(depth, depth_masked, depth_rendered, mask_hand):
        depth = depth.flatten()
        depth_rendered = depth_rendered.flatten()
        mask_hand = mask_hand.flatten()
        depth_filled = np.copy(depth_masked).flatten()
        '''
        for i in range(len(depth_rendered)):
            #if depth_filled[i] == 0 and depth_rendered[i] > 0 and \
            #   np.abs(depth_rendered[i] - depth[i]) < 0.1:# and depth[i] < 0.5:
            #    depth_filled[i] = depth_rendered[i]
            if depth_filled[i] == 0 and depth_rendered[i] > 0:
                # Only fill in the depth if this is not a hand pixel or if not background
                if mask_hand[i] == 0 and depth[i] < 0.5:
                    ## Invalid pixel due to sensor (depth[i] == 0) -> fill with rendered depth
                    #if depth[i] == 0:
                    #    depth_filled[i] = depth_rendered[i]
                    # Invalid because of misaligned mask (depth[i] ~= depth_rendered[i]) -> fill with original depth
                    if depth[i] > depth_rendered[i] and np.abs(depth_rendered[i] - depth[i]) < 0.1:
                        depth_filled[i] = depth[i]
        '''

        for i in range(len(depth_rendered)):
            # Remove all depth values that are not in the rendered depth
            if depth_filled[i] > 0 and depth_rendered[i] == 0:
                depth_filled[i] = 0.0
            #'''
            if depth_filled[i] == 0 and depth_rendered[i] > 0:
                # Only fill in the depth if this is not a hand pixel or if not background
                if mask_hand[i] == 0 and depth[i] < 0.5:
                    # Invalid because of misaligned mask (depth[i] ~= depth_rendered[i]) -> fill with original depth
                    #if depth[i] == 0:
                    #   depth_filled[i] = depth_rendered[i]
                    if depth[i] > depth_rendered[i] and np.abs(depth_rendered[i] - depth[i]) < 0.1:
                        depth_filled[i] = depth[i]
            #'''

        # Reshape back to input shape
        depth = depth_rendered.reshape(depth_masked.shape)
        depth_rendered = depth_rendered.reshape(depth_masked.shape)
        mask_hand = mask_hand.reshape(depth_masked.shape)
        depth_filled = depth_filled.reshape(depth_masked.shape)

        return depth_filled

    def image_to_world(self, rgb, depth, mask=None, cut_z=1000.):
        i_fx = 1. / self.anno['camMat'][0, 0]
        i_fy = 1. / self.anno['camMat'][1, 1]
        cx = self.anno['camMat'][0, 2]
        cy = self.anno['camMat'][1, 2]

        pts = []
        colors = []
        for v in range(rgb.shape[0]):
            for u in range(rgb.shape[1]):
                if mask is None or mask[v, u] > 0:
                    z = depth[v, u]
                    if z > 0.001 and z < cut_z:
                        x = (u - cx) * z * i_fx
                        y = (v - cy) * z * i_fy
                        pts.append([x, y, z])
                        colors.append(rgb[v, u])

        pts = np.asarray(pts)
        colors = np.asarray(colors) / 255

        return pts, colors

    @staticmethod
    def apply_mask(image, mask_obj, mask_hand):
        # Get the indices that are not in the mask
        valid_idx = np.where(mask_obj.flatten() == 0)[0]

        # Set the invalid indices to 0
        masked_image = np.copy(image).flatten()
        for v in valid_idx:
            masked_image[v] = 0  # [0, 0, 0]

        # Remove any pixels belonging to the hand
        valid_idx = np.where(mask_hand.flatten() != 0)[0]
        for v in valid_idx:
            masked_image[v] = 0

        # Reshape back to input shape
        masked_image = masked_image.reshape(image.shape)

        #masked_image = copy.deepcopy(image)
        #for u in range(mask.shape[0]):
        #    for v in range(mask.shape[1]):
        #        if mask_to_apply[u, v] == 0:
        #            masked_image[u, v] = 0

        return masked_image

    def get_inpainted_depth(self, masked_depth, mask_obj, mask_hand):
        dep_shape = masked_depth.shape
        masked_depth = masked_depth.flatten()
        # Mask out values that have too large depth values
        for v in range(len(masked_depth)):
            if masked_depth[v] > self.args.max_depth_tsdf:
                masked_depth[v] = 0
        masked_depth = masked_depth.reshape(dep_shape)

        # Get the mask that needs to be inpainted
        inpaint_mask = np.multiply((mask_obj == 255), (masked_depth == 0)).astype(np.uint8)
        # Inpaint the depth image
        inpainted_depth = self.inpaint(masked_depth, mask=inpaint_mask)

        '''
        # Inpaint the hand pixels
        if mask_hand is not None:
            inpaint_mask_hand = np.multiply((mask_hand == 255), (masked_depth == 0)).astype(np.uint8)
            inpainted_depth = self.depth_fill(inpainted_depth, mask=inpaint_mask_hand)

            # Fill in pixels from morphological close
            mask_morph = (inpainted_depth > 0).astype(np.uint8)
            mask_closed = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
            inpaint_mask_morph = ((mask_closed - mask_morph) == 1).astype(np.uint8)
            inpainted_depth = self.depth_fill(inpainted_depth, mask=inpaint_mask_morph)
        '''

        return inpainted_depth

    @staticmethod
    def inpaint(img_in, mask=None, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img_out = cv2.copyMakeBorder(img_in, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        if mask is None:
            mask = (img_out == missing_value).astype(np.uint8)
        else:
            mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy
        scale = np.abs(img_out).max()
        img_out = img_out.astype(np.float32) / scale  # Has to be float32, 64 not supported
        img_out = cv2.inpaint(img_out, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img_out = img_out[1:-1, 1:-1]
        img_out = img_out * scale

        return img_out

    @staticmethod
    def depth_fill(img_in, mask=None, radius=20, iterations=1):
        # Get the indices in the mask that need to be inpainted
        mask_coords = np.unravel_index(np.where(mask.flatten() == 1)[0], mask.shape)
        mask_coords = list(zip(mask_coords[1], mask_coords[0]))

        # Iterate through the list and fill in missing values
        img_out = np.copy(img_in)
        search_radius = radius
        for i in range(iterations):
            # For each mask pixel
            valid_mask_coords = []
            for center in mask_coords:
                # If the image pixel is 0, then it should be inpainted
                if img_out[center[1], center[0]] == 0:
                    # Get all neighboring pixels within a radius
                    img_canvas = np.zeros(img_out.shape, np.uint8)
                    cv2.circle(img_canvas, center, int(search_radius), 255, -1)
                    neighbor_pixels = np.where(img_canvas == 255)
                    neighbor_pixels = list(zip(neighbor_pixels[0], neighbor_pixels[1]))
                    # Get the valid neighboring pixels
                    valid_npx = []
                    for npx in neighbor_pixels:
                        if img_out[npx[0], npx[1]] > 0:
                            valid_npx.append((npx[0], npx[1]))
                    # If there are valid neighbors, add this to the list of valid mask pixels
                    if len(valid_npx) > 0:
                        valid_mask_coords.append((len(valid_npx), valid_npx, center))

            # Sort the mask pixels according to the count of neighbors
            valid_mask_coords = sorted(valid_mask_coords, key=lambda x: x[0])
            # print(valid_mask_coords)

            # Get the mean of the neighbors and fill the pixel, starting with pixels with the most valid neighbors
            for mask_el in valid_mask_coords:
                depth = 0
                for npx in mask_el[1]:
                    depth += img_out[npx[0], npx[1]]
                img_out[mask_el[2][1], mask_el[2][0]] = depth / float(mask_el[0])

            # Reduce the search radius for the next iteration
            search_radius /= 2
            if search_radius < 1:
                search_radius = 1

        return img_out

    def transform_to_object_frame(self, cloud, pose_in_frames=None, frame_id=None):
        # Get the ground truth transformation
        if pose_in_frames is None or frame_id not in pose_in_frames:
            print('...HO-3D pose')
            # print('Object pose not provided for frame {}'.format(frame_id))
            obj_trans = np.copy(self.anno['objTrans'])
            # print('obj_trans.shape {}'.format(obj_trans.shape))
            obj_trans = obj_trans.dot(COORD_CHANGE_MAT.T)
            # print('obj_trans.shape {}'.format(obj_trans.shape))
            obj_rot = np.copy(self.anno['objRot'])
            # print('obj_rot.shape {}'.format(obj_rot.shape))
            obj_rot = obj_rot.flatten().dot(COORD_CHANGE_MAT.T).reshape(self.anno['objRot'].shape)
            # print('obj_rot.shape {}'.format(obj_rot.shape))
            rot_max = cv2.Rodrigues(obj_rot)[0].T
            # print('rot_max.shape {}'.format(rot_max.shape))

            cloud_tfd = np.copy(cloud)
            cloud_tfd -= obj_trans
            cloud_tfd = np.matmul(cloud_tfd, np.linalg.inv(rot_max))

            cam_pose = np.eye(4)
            cam_pose[:3, :3] = rot_max
            cam_pose[:3, 3] = np.matmul(-obj_trans, np.linalg.inv(rot_max))

        else:
            print('...annotated pose')
            '''
            pose = poses[i]
            rotx = np.eye(4)
            rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
            pose = np.matmul(rotx, pose)

            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(cloud)
            vis_pcd.transform(pose)
            '''

            t_mat = np.asarray(pose_in_frames[frame_id]).reshape((4, 4))
            rotx = np.eye(4)
            rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
            t_mat = np.matmul(rotx, t_mat)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud)
            pc.transform(t_mat)

            cloud_tfd = np.asarray(pc.points)
            cam_pose = t_mat

        if self.args.reg_method == RegMethod.GT or self.prev_cloud is None:
            return cloud_tfd, cam_pose, cloud_tfd, cam_pose
        else:
            # Down sample
            source_cloud = o3d.geometry.PointCloud()
            source_cloud.points = o3d.utility.Vector3dVector(cloud)
            if self.args.voxel_size > 0:
                source_cloud = o3d.geometry.voxel_down_sample(source_cloud, self.args.voxel_size)
            # Perform registration
            est_cam_pose = np.copy(self.register_new_cloud(source_cloud, cam_pose))
            # Transform the cloud
            est_cloud_tfd = np.copy(cloud)
            est_cloud_trans = np.matmul(-est_cam_pose[0:3, 3], est_cam_pose[0:3, 0:3])
            est_cloud_tfd -= est_cloud_trans
            est_cloud_tfd = np.matmul(est_cloud_tfd, np.linalg.inv(est_cam_pose[0:3, 0:3]))
            euler_gt = tf3d.euler.mat2euler(cam_pose[:3, :3])
            trans_gt = cam_pose[:3, 3]
            euler_est = tf3d.euler.mat2euler(est_cam_pose[:3, :3])
            trans_est = est_cam_pose[:3, 3]
            print('X {:.4f}\tY {:.4f}\tZ {:.4f}\tRx {:.4f}\tRy {:.4f}\tRz {:.4f}'.format(
                abs(trans_gt[0] - trans_est[0]) * 100, abs(trans_gt[1] - trans_est[1]) * 100,
                abs(trans_gt[2] - trans_est[2]) * 100,
                np.degrees(abs(euler_gt[0] - euler_est[0])), np.degrees(abs(euler_gt[1] - euler_est[1])),
                np.degrees(abs(euler_gt[2] - euler_est[2]))))

            return cloud_tfd, cam_pose, est_cloud_tfd, est_cam_pose

    def register_new_cloud(self, cloud, initial_transform):
        return self.align_icp(cloud, initial_transform)

    def align_icp(self, cloud, trans_init):
        # Get the threshold
        threshold = ICP_THRESH

        # Perform ICP
        if self.args.icp_method == ICPMethod.Point2Point:
            reg_p2p = o3d.registration.registration_icp(cloud, self.prev_cloud, threshold, trans_init,
                                                        o3d.registration.TransformationEstimationPointToPoint())
        elif self.args.icp_method == ICPMethod.Point2Plane:
            if not cloud.has_normals():
                radius_normal = self.args.voxel_size * 2
                o3d.geometry.estimate_normals(
                    cloud,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            if not self.prev_cloud.has_normals():
                radius_normal = self.args.voxel_size * 2
                o3d.geometry.estimate_normals(
                    self.prev_cloud,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            reg_p2p = o3d.registration.registration_icp(cloud, self.prev_cloud, threshold, trans_init,
                                                        o3d.registration.TransformationEstimationPointToPlane())
        else:
            print('Unrecognized ICP method {}'.format(self.args.icp_method))
            return None

        return reg_p2p.transformation

    def align_to_cad(self, mesh_tsdf):
        # Load the CAD model to align the mesh to
        ho3d_to_ycb_map_path = '/home/tpatten/Data/Hands/HO3D/ho3d_to_ycb.json'
        with open(os.path.join(ho3d_to_ycb_map_path)) as f:
            model_name_data = json.load(f)
        scene_key = ''.join([i for i in self.args.scene if not i.isdigit()])
        ycb_model_filename = os.path.join(
            YCB_MODELS_DIR, 'models', model_name_data[scene_key]['ycbv'], 'mesh.ply')
        target = o3d.io.read_point_cloud(ycb_model_filename)

        points = copy.deepcopy(np.asarray(mesh_tsdf.vertices))

        # Create a point cloud from the mesh and initially align it to the global frame
        source = o3d.geometry.PointCloud()
        #points = np.matmul(points, np.linalg.inv(self.pose_annotation_global_rot))
        #centroid = np.mean(points, axis=0)
        #points -= centroid
        points -= self.pose_annotation_global_tra
        points = np.matmul(points, np.linalg.inv(self.pose_annotation_global_rot))
        source.points = o3d.utility.Vector3dVector(points)

        # Run ICP to align the mesh to the model
        threshold = 0.02
        o3d.geometry.estimate_normals(source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.geometry.estimate_normals(target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2l = o3d.registration.registration_icp(
            source, target, threshold, np.eye(4),
            o3d.registration.TransformationEstimationPointToPlane())
        source.transform(reg_p2l.transformation)

        # Transform the mesh
        mesh_aligned = copy.deepcopy(mesh_tsdf)
        mesh_aligned.vertices = source.points

        triangle_normals = np.asarray(mesh_tsdf.triangle_normals)
        triangle_normals = np.matmul(triangle_normals, np.linalg.inv(self.pose_annotation_global_rot))
        triangle_normals = np.matmul(triangle_normals, reg_p2l.transformation[:3, :3])
        mesh_aligned.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)

        vertex_normals = np.asarray(mesh_tsdf.vertex_normals)
        vertex_normals = np.matmul(vertex_normals, np.linalg.inv(self.pose_annotation_global_rot))
        vertex_normals = np.matmul(vertex_normals, reg_p2l.transformation[:3, :3])
        mesh_aligned.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)

        return mesh_aligned

    @staticmethod
    def visualize(cam_poses=None, mask_pcds=None, scene_pcds=None, gt_poses=None):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Plot the coordinate frame
        # vis.add_geometry(o3d.create_mesh_coordinate_frame(size=0.5))

        # Plot the object cloud
        for m in mask_pcds:
            vis.add_geometry(m)

        # Plot scene
        if scene_pcds is not None:
            for s in scene_pcds:
                vis.add_geometry(s)

        # Plot camera pose
        if cam_poses is not None:
            for m in cam_poses:
                points = m[:3, :].T
                points[0:3, :] *= 0.175
                points[0:3, :] += np.tile(points[3, :], (3, 1))
                lines = [[3, 0], [3, 1], [3, 2]]
                line_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                vis.add_geometry(line_set)

        # Plot the ground truth camera poses
        if gt_poses is not None:
            for m in gt_poses:
                points = m[:3, :].T
                points[0:3, :] *= 0.1
                points[0:3, :] += np.tile(points[3, :], (3, 1))
                lines = [[3, 0], [3, 1], [3, 2]]
                line_colors = [[.6, .4, .4], [.4, .6, .4], [.4, .4, .6]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                vis.add_geometry(line_set)

        # End visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_volume(mesh):
        # Visualize
        if mesh is not None:
            o3d.visualization.draw_geometries([mesh])

    def save_clouds_and_camera_poses(self, base_dir, frame_ids, cam_poses, est_cam_poses, mask_pcds, mesh,
                                     mesh_aligned=None, frame_id=0):
        down_pcd = None

        # Combine and downsample the clouds
        if len(mask_pcds) > 0:
            down_pcd = self.combine_clouds(mask_pcds)

            # Extract the points and normals
            points = np.asarray(down_pcd.points)
            normals = np.asarray(down_pcd.normals)

        # Create the filename and write the data
        filename = os.path.join(self.base_dir, 'reconstructions',
                                self.args.scene + '_' + str(self.args.reg_method).split('.')[1])
        if self.args.reg_method == RegMethod.GT_ICP or self.args.reg_method == RegMethod.GT_ICP_FULL:
            filename += '_' + str(self.args.icp_method).split('.')[1]
        filename += '_start' + str(self.args.start_frame) + '_max' + str(self.args.max_num) +\
                    '_skip' + str(self.args.skip)
        #if self.mask_dir != OBJECT_MASK_VISIBLE_DIR:
        #    filename += '_segHO3D'
        if self.hand_obj_rendering_scores is not None:
            filename += '_renFilter'

        if self.args.viewpoint_file != '':
            filename = os.path.join(self.base_dir, 'reconstructions',
                                    self.args.scene + '_' + str(self.args.viewpoint_file).split('.')[0])

        if self.args.pose_annotation_file != '':
            filename = os.path.join(self.base_dir, 'reconstructions', self.args.scene + '_AnnoPoses')

        if self.args.apply_inpainting:
            filename += '_inPaint'
        if self.args.inpaint_with_rendering:
            filename += 'Rend'

        #if int(frame_id) > 0:
        #    filename += '_frame' + str(frame_id)

        filename += '.xyz'

        if down_pcd is not None:
            #print('Saving to: {}'.format(filename))
            #f = open(filename, "w")
            #for i in range(len(points)):
            #    f.write('{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2],
            #                                         normals[i, 0], normals[i, 1], normals[i, 2]))
            #f.close()

            # Write at .ply
            print('Saving to: {}'.format(filename.replace('.xyz', '.ply')))
            o3d.io.write_point_cloud(filename.replace('.xyz', '.ply'), down_pcd)

        # Save the mesh as .ply
        filename = filename.replace('.xyz', '')
        if self.args.viewpoint_file == '':
            filename += '_visRatio' + str(self.args.min_ratio_valid).replace('.', '-')
        filename += '_tsdf.ply'
        print('Saving to: {}'.format(filename))
        o3d.io.write_triangle_mesh(filename, mesh)

        # Save the aligned mesh
        if mesh_aligned is not None:
            filename = filename.replace('.ply', '_aligned.ply')
            print('Saving to: {}'.format(filename))
            o3d.io.write_triangle_mesh(filename, mesh_aligned)

        return down_pcd

    @staticmethod
    def combine_clouds(mask_pcds):
        if len(mask_pcds) == 0:
            return None

        all_points = np.asarray(mask_pcds[0].points)
        all_colors = np.asarray(mask_pcds[0].colors)
        for i in range(1, len(mask_pcds)):
            all_points = np.vstack((all_points, np.asarray(mask_pcds[i].points)))
            all_colors = np.vstack((all_colors, np.asarray(mask_pcds[i].colors)))

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        down_pcd = o3d.geometry.voxel_down_sample(combined_pcd, voxel_size=0.001)

        o3d.geometry.estimate_normals(down_pcd,
                                      search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                          radius=0.1, max_nn=30))

        return down_pcd

    @staticmethod
    def load_object_model(filename):
        print('Loading file {}'.format(filename))
        pcd = o3d.geometry.PointCloud()
        pts = np.loadtxt(filename)[:, :3]
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd


def remove_outliers(cloud, outlier_rm_nb_neighbors=0., outlier_rm_std_ratio=0.,
                    raduis_rm_min_nb_points=0, raduis_rm_radius=0.):
    # Copy the input cloud
    in_cloud = copy.deepcopy(cloud)

    if o3d.__version__ == "0.7.0.0":
        # Statistical outlier removal
        if outlier_rm_nb_neighbors > 0 and outlier_rm_std_ratio > 0:
            in_cloud, _ = o3d.geometry.statistical_outlier_removal(
                in_cloud, nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
        # Radius outlier removal
        if raduis_rm_min_nb_points > 0 and raduis_rm_radius > 0:
            in_cloud, _ = o3d.geometry.radius_outlier_removal(
                in_cloud, nb_points=raduis_rm_min_nb_points, radius=raduis_rm_radius)
    else:
        # Statistical outlier removal
        if outlier_rm_nb_neighbors > 0 and outlier_rm_std_ratio > 0:
            in_cloud, _ = in_cloud.remove_statistical_outlier(
                nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
        # Radius outlier removal
        if raduis_rm_min_nb_points > 0 and raduis_rm_radius > 0:
            in_cloud, _ = in_cloud.remove_radius_outlier(nb_points=raduis_rm_min_nb_points, radius=raduis_rm_radius)

    return in_cloud


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HO-3D Object model reconstruction')
    parser.add_argument("scene", type=str, help="Sequence of the dataset")
    parser.add_argument("--min_ratio_valid", type=float, default=0.1, help="The threshold at which visibility is set")
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2'
    args.model_file = ''
    args.mask_dir = 'mask_hsdc_ofdd'
    args.hand_obj_rendering_scores = '/home/tpatten/Data/bop/ho3d/hand_obj_ren_scores.json'
    args.viewpoint_file = 'views_Uniform_Segmentation_step0-3.json'
    args.pose_annotation_file = 'pair_pose.json'
    args.align_to_cad = True
    args.visualize = False
    args.save = True
    args.save_intermediate = False
    args.icp_method = ICPMethod.Point2Plane
    # Point2Point=1, Point2Plane=2
    args.reg_method = RegMethod.GT
    # GT=1, GT_ICP=2, GT_ICP_FULL=3
    args.start_frame = 0
    args.max_num = -1
    args.skip = 1
    args.mask_erosion_kernel = [5, 5]
    args.outlier_rm_nb_neighbors = 2500  # Higher is more aggressive
    args.outlier_rm_std_ratio = 0.000001  # Smaller is more aggressive
    args.raduis_rm_min_nb_points = 250  # Don't change
    args.raduis_rm_radius_factor = 10  # Don't change
    args.voxel_size = 0.001
    args.sdf_voxel_length = 0.001
    args.sdf_trunc = 0.02
    args.sdf_depth_trunc = 1.2
    args.max_depth_tsdf = 1.1
    args.construct_tsdf = True
    args.min_num_pixels = 8000
    # args.min_ratio_valid = 0.10
    args.apply_inpainting = True
    args.inpaint_with_rendering = True
    args.visualize_inpainting = False

    # Create the reconstruction
    reconstructor = ModelReconstructor(args)
