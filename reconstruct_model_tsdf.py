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


OBJECT_MASK_VISIBLE_DIR = 'object_vis'
COORD_CHANGE_MAT = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
ICP_THRESH = 0.02
RANSAC_THRESH = 0.015
YCB_MODELS_DIR = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/'


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
        if args.mask_erosion_kernel > 0:
            self.erosion_kernel = np.ones((args.mask_erosion_kernel, args.mask_erosion_kernel), np.uint8)
        else:
            self.erosion_kernel = None
        self.volume = None

        # TODO: Come back to testing poses from Kiru's annotation tool
        # self.test_func()

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
                    filename = os.path.join(self.base_dir, self.data_split,
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
                        filename = os.path.join(self.base_dir, self.data_split,
                                                self.args.scene + '_' + str(self.args.viewpoint_file).split('.')[0])

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

    def test_func(self):
        # Load two of the images
        scene_id = 'ABF10'
        frame_ids = ['0090', '0514', '1055']
        poses = [np.eye(4),
                 np.asarray([0.62578040423448, 0.6975679028668021, -0.34899556811819804, 0.13193419399246284,
                             -0.5044116046496037, 0.7031957775374543, 0.5010834576677885, -0.15048727462816902,
                             0.594951946606566, -0.13753079416445993, 0.7919074831604712, 0.17977535288721627,
                             0.0, 0.0, 0.0, 1.0]).reshape((4, 4)),
                 np.asarray([-0.14613412359310565, -0.9361789072739534, 0.3197090419381649, -0.05513735799376178,
                             -0.9823675359963918, 0.099231970655764, -0.15845201235752068, 0.16534490025361823,
                             0.1166140735162308, -0.3372270297219925, -0.9341729434547074, 0.9158804802458833,
                             0.0, 0.0, 0.0, 1.0]).reshape((4, 4))]

        use_gt = False
        do_inv = False
        do_coord_change = True
        aligned_clouds = []
        for i in range(len(frame_ids)):
            # Read image, depths maps and annotations
            self.rgb, self.depth, self.anno, _ = self.load_data(scene_id, frame_ids[i])
            # Read the mask
            mask, mask_hand = self.load_mask(scene_id, frame_ids[i])
            # Extract the masked point cloud
            cloud, colors = self.image_to_world(self.rgb, self.depth, mask, cut_z=2.0)

            # Transform the cloud and get the camera position
            if use_gt:
                obj_trans = np.copy(self.anno['objTrans'])
                if do_coord_change:
                    obj_trans = obj_trans.dot(COORD_CHANGE_MAT.T)
                obj_rot = np.copy(self.anno['objRot'])
                if do_coord_change:
                    obj_rot = obj_rot.flatten().dot(COORD_CHANGE_MAT.T).reshape(self.anno['objRot'].shape)
                rot_max = cv2.Rodrigues(obj_rot)[0].T
                # print(obj_rot)
                # print(rot_max)
                # print(cv2.Rodrigues(rot_max)[0])

                cloud_tfd = np.copy(cloud)
                cloud_tfd -= obj_trans
                cloud_tfd = np.matmul(cloud_tfd, np.linalg.inv(rot_max))
            else:
                t_mat = poses[i]
                obj_trans = t_mat[:3, 3].reshape((3,))
                rot_max = t_mat[:3, :3]

                if do_coord_change:
                    # rotx = np.eye(4)
                    # from scipy.spatial.transform.rotation import Rotation
                    # rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
                    # pose = np.eye(4)
                    # pose[:3, :3] = rot_max
                    # pose[:3, 3] = obj_trans
                    # obj_trans = pose[:3, 3].reshape((3,))
                    # rot_max = pose[:3, :3]

                    obj_trans = obj_trans.dot(COORD_CHANGE_MAT.T)
                    obj_rot = cv2.Rodrigues(rot_max)[0]
                    obj_rot = obj_rot.flatten().dot(COORD_CHANGE_MAT.T).reshape(obj_rot.shape)
                    rot_max = cv2.Rodrigues(obj_rot)[0].T

                if not do_inv:
                    cloud_tfd = np.copy(cloud)
                    cloud_tfd += obj_trans
                    cloud_tfd = np.matmul(cloud_tfd, rot_max)
                else:
                    cloud_tfd = np.copy(cloud)
                    cloud_tfd -= obj_trans
                    cloud_tfd = np.matmul(cloud_tfd, np.linalg.inv(rot_max))
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(cloud_tfd)
            if i == 0:
                vis_pcd.paint_uniform_color([0.9, 0.1, 0.1])
            elif i == 1:
                vis_pcd.paint_uniform_color([0.1, 0.9, 0.1])
            elif i == 2:
                vis_pcd.paint_uniform_color([0.1, 0.1, 0.9])
            else:
                vis_pcd.colors = o3d.utility.Vector3dVector(colors)
            aligned_clouds.append(vis_pcd)

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for c in aligned_clouds:
            vis.add_geometry(c)
        vis.run()
        vis.destroy_window()

        sys.exit(0)

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

        # Save
        combined_pcd = None
        if self.args.save:
            base_dir = os.path.join(self.base_dir, self.data_split)
            combined_pcd = self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses,
                                                             mask_pcds, tsdf_mesh)

        # Visualize
        if self.args.visualize:
            scene_pcds = None
            print('Visualizing aligned clouds...')
            if self.args.reg_method == RegMethod.GT:
                self.visualize(cam_poses, mask_pcds, scene_pcds=scene_pcds)
            else:
                self.visualize(est_cam_poses, mask_pcds, scene_pcds=scene_pcds, gt_poses=cam_poses)
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

            masked_rgb = self.apply_mask(self.rgb, mask)
            masked_depth = self.apply_mask(self.depth, mask)

            # Inpaint the depth image
            if self.args.apply_inpainting:
                if self.args.inpaint_with_rendering:
                    # Render the depth from the model
                    rendered_depth = self.render_depth(cam_params)
                    # Fill the missing values with rendered depth values
                    masked_depth = self.fill_from_rendering(masked_depth, rendered_depth)
                    self.depth = masked_depth
                    mask = None
                else:
                    inpainted_depth = self.get_inpainted_depth(masked_depth, mask, mask_hand)
                    masked_depth = inpainted_depth
                # if self.args.inpaint_with_rendering:
                #    right_img = rendered_depth
                # else:
                #    right_img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
                # img_output = np.vstack((np.hstack((self.depth, right_img)),
                #                        np.hstack((masked_depth, inpainted_depth))))
                # cv2.imshow('Depth image', img_output)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                #masked_depth = inpainted_depth
                #masked_depth = rendered_depth
                #self.depth = rendered_depth
                #mask = None

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

            # Save intermediate results
            if self.args.save and self.args.save_intermediate and num_processed % 20 == 0:
                base_dir = os.path.join(self.base_dir, self.data_split, scene_id)
                self.save_clouds_and_camera_poses(base_dir, processed_frames, cam_poses, est_cam_poses, mask_pcds,
                                                  frame_id=frame_id)

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

        # If my masks
        if self.mask_dir == OBJECT_MASK_VISIBLE_DIR:
            mask_filename = os.path.join(self.base_dir, self.data_split, scene_id, 'mask',
                                         self.mask_dir, str(frame_id) + '.png')
            if not os.path.exists(mask_filename):
                return mask, hand

            mask = cv2.imread(mask_filename)[:, :, 0]
        # Otherwise, ho3d masks
        else:
            # Load the mask file
            mask_filename = os.path.join(self.mask_dir, self.data_split, scene_id, 'seg', str(frame_id) + '.jpg')

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
        if self.erosion_kernel is not None:
            mask = cv2.erode(mask, self.erosion_kernel, iterations=1)
            hand = cv2.erode(hand, self.erosion_kernel, iterations=1)

        return mask, hand

    def render_depth(self, cam_intrinsics):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(self.anno['objRot'])[0]
        pose[:3, 3] = self.anno['objTrans']

        # From OpenGL coordinates
        from scipy.spatial.transform.rotation import Rotation
        rotx = np.eye(4)
        rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
        # pose = rotx @ pose
        pose = np.matmul(rotx, pose)

        # fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        fx, fy = cam_intrinsics.get_focal_length()
        cx, cy = cam_intrinsics.get_principal_point()
        rendering = self.renderer.render_object(self.anno['objName'], pose[:3, :3], pose[:3, 3], fx, fy, cx, cy)
        depth = rendering['depth']

        return depth

    def add_objects_to_renderer(self):
        # Get all the .ply files in the model directory
        # model_dir_name = 'models'
        model_dir_name = 'reconstructions'
        obj_ids = sorted(os.listdir(os.path.join(YCB_MODELS_DIR, model_dir_name)))
        for obj_id in obj_ids:
            model_path = os.path.join(os.path.join(YCB_MODELS_DIR, model_dir_name), obj_id, 'mesh.ply')
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
    def fill_from_rendering(depth, rendered_depth):
        filled_depth = np.copy(depth).flatten()
        rendered_depth = rendered_depth.flatten()
        for i in range(len(rendered_depth)):
            if filled_depth[i] == 0 and rendered_depth[i] > 0:
                filled_depth[i] = rendered_depth[i]

        # Reshape back to input shape
        filled_depth = filled_depth.reshape(depth.shape)
        rendered_depth = rendered_depth.reshape(rendered_depth.shape)

        return filled_depth

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

    def apply_mask(self, image, mask):
        # Get the indices that are not in the mask
        valid_idx = np.where(mask.flatten() == 0)[0]

        # Set the invalid indices to 0
        masked_image = np.copy(image).flatten()
        for v in valid_idx:
            masked_image[v] = 0  # [0, 0, 0]

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
            t_mat = np.asarray(pose_in_frames[frame_id]).reshape((4, 4))
            obj_trans = t_mat[:3, 3].reshape((3,))
            rot_max = t_mat[:3, :3]

            rotx = np.eye(4)
            from scipy.spatial.transform.rotation import Rotation
            rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
            pose = np.eye(4)
            pose[:3, :3] = rot_max
            pose[:3, 3] = obj_trans
            obj_trans = pose[:3, 3].reshape((3,))
            rot_max = pose[:3, :3]

            do_inv = True
            if not do_inv:
                cloud_tfd = np.copy(cloud)
                cloud_tfd += obj_trans
                cloud_tfd = np.matmul(cloud_tfd, rot_max)
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = rot_max
                cam_pose[:3, 3] = np.matmul(obj_trans, rot_max)
            else:
                cloud_tfd = np.copy(cloud)
                cloud_tfd -= obj_trans
                cloud_tfd = np.matmul(cloud_tfd, np.linalg.inv(rot_max))
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = rot_max
                cam_pose[:3, 3] = np.matmul(-obj_trans, np.linalg.inv(rot_max))

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

    def save_clouds_and_camera_poses(self, base_dir, frame_ids, cam_poses, est_cam_poses, mask_pcds, mesh, frame_id=0):
        down_pcd = None

        # Combine and downsample the clouds
        if len(mask_pcds) > 0:
            down_pcd = self.combine_clouds(mask_pcds)

            # Extract the points and normals
            points = np.asarray(down_pcd.points)
            normals = np.asarray(down_pcd.normals)

        # Create the filename and write the data
        filename = os.path.join(base_dir, self.args.scene + '_' + str(self.args.reg_method).split('.')[1])
        if self.args.reg_method == RegMethod.GT_ICP or self.args.reg_method == RegMethod.GT_ICP_FULL:
            filename += '_' + str(self.args.icp_method).split('.')[1]
        filename += '_start' + str(self.args.start_frame) + '_max' + str(self.args.max_num) +\
                    '_skip' + str(self.args.skip)
        #if self.mask_dir != OBJECT_MASK_VISIBLE_DIR:
        #    filename += '_segHO3D'
        if self.hand_obj_rendering_scores is not None:
            filename += '_renFilter'

        if self.args.viewpoint_file != '':
            filename = os.path.join(self.base_dir, self.data_split,
                                    self.args.scene + '_' + str(self.args.viewpoint_file).split('.')[0])

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

        # Write the mesh as .ply
        filename = filename.replace('.xyz', '')
        if self.args.viewpoint_file == '':
            filename += '_visRatio' + str(self.args.min_ratio_valid).replace('.', '-')
        filename += '_tsdf.ply'
        print('Saving to: {}'.format(filename))
        o3d.io.write_triangle_mesh(filename, mesh)

        '''
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(down_pcd)
        vis.run()
        vis.destroy_window()
        '''

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
    # args.ho3d_path = '/home/tpatten/Data/Hands/HO3D/'
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2'
    args.model_file = ''
    # args.model_file = '/home/tpatten/Data/Hands/HO3D/train/ABF10/GT_start0_max-1_skip1.xyz'
    # args.mask_dir = OBJECT_MASK_VISIBLE_DIR
    args.mask_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2_segmentations_rendered/'
    args.hand_obj_rendering_scores = '/home/tpatten/Data/bop/ho3d/hand_obj_ren_scores.json'
    args.viewpoint_file = 'views_Uniform_Segmentation_step0-3.json'
    args.pose_annotation_file = ''  # 'pose_hand_annotation.json'
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
    args.mask_erosion_kernel = 5
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
    args.apply_inpainting = False
    args.inpaint_with_rendering = False

    # Create the reconstruction
    reconstructor = ModelReconstructor(args)
