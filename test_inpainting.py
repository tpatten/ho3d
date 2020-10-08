from utils.grasp_utils import *
import argparse
import cv2
import numpy as np
import os
import open3d as o3d

from bop_toolkit_lib import renderer
import trimesh
import copy


data_split = 'train'
OBJECT_MASK_VISIBLE_DIR = 'object_vis'
YCB_MODELS_DIR = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/'


def inpaint(img_in, mask=None, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    print('==> Inpainting')
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
    img_out = cv2.inpaint(img_out, mask, 10, cv2.INPAINT_NS)   # INPAINT_NS, INPAINT_TELEA

    # Back to original size and value range.
    img_out = img_out[1:-1, 1:-1]
    img_out = img_out * scale

    return img_out


def inpaint2(img_in, mask=None, radius=20, iterations=1):
    print('==> Inpainting2')
    # Get the indices in the mask that need to be inpainted
    mask_coords = np.unravel_index(np.where(mask.flatten() == 1)[0], mask.shape)
    mask_coords = list(zip(mask_coords[1], mask_coords[0]))

    # Iterate through the list and fill in missing values
    img_out = np.copy(img_in)
    search_radius = radius
    for i in range(iterations):
        print('Iteration {}'.format(i))
        # For each mask pixel
        print('Getting pixels to be filled')
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
        print('Getting depth for each pixel')
        for mask_el in valid_mask_coords:
            depth = 0
            for npx in mask_el[1]:
                depth += img_out[npx[0], npx[1]]
            img_out[mask_el[2][1], mask_el[2][0]] = depth / float(mask_el[0])

        # Reduce the search radius for the next iteration
        search_radius /= 2
        if search_radius < 1:
            search_radius = 1

        '''
        print('Preparing visualization')
        img_canvas = np.zeros(img_in.shape, np.uint8)
        for mask_el in valid_mask_coords:
            cv2.circle(img_canvas, mask_el[2], 1, 255, -1)
        for u in range(img_in.shape[0]):
            for v in range(img_in.shape[1]):
                if img_in[u, v] > 0:
                    img_canvas[u, v] = 100

        img_canvas = img_canvas.astype(np.float32) / 255.0
        img_output = np.hstack((img_in, mask))
        img_output = np.hstack((img_output, img_canvas))
        img_output = np.hstack((img_output, img_out))
        cv2.imshow('Depth image', img_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # sys.exit(0)
        '''
    # sys.exit(0)

    return img_out


def load_data(base_dir, mask_dir, seq_name, frame_id):
    print('==> Loading data')
    # Depth image
    depth = read_depth_img(base_dir, seq_name, frame_id, data_split)

    # Mask
    mask_obj, mask_hand, mask_scene = load_mask(mask_dir, seq_name, frame_id)

    return depth, mask_obj, mask_hand, mask_scene


def load_mask(mask_dir, seq_name, frame_id):
    print('==> Loading mask')
    mask = None

    # Load the mask file
    mask_filename = os.path.join(mask_dir, data_split, seq_name, 'seg', str(frame_id) + '.jpg')
    if not os.path.exists(mask_filename):
        return mask
    mask_rgb = cv2.imread(mask_filename)

    # Generate binary object mask
    mask_obj = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))
    mask_hand = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))
    mask_scene = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]))
    for u in range(mask_rgb.shape[0]):
        for v in range(mask_rgb.shape[1]):
            if mask_rgb[u, v, 0] > 230 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] < 10:
                mask_obj[u, v] = 255
            if mask_rgb[u, v, 0] < 10 and mask_rgb[u, v, 1] < 10 and mask_rgb[u, v, 2] > 230:
                mask_hand[u, v] = 255
            # if mask_rgb[u, v, 0] > 1 or mask_rgb[u, v, 1] > 1 or mask_rgb[u, v, 2] > 1:
            if mask_obj[u, v] == 255 or mask_hand[u, v] == 255:
                mask_scene[u, v] = 255

    # Resize image to original size
    mask_obj = cv2.resize(mask_obj, (640, 480), interpolation=cv2.INTER_NEAREST)
    mask_hand = cv2.resize(mask_hand, (640, 480), interpolation=cv2.INTER_NEAREST)
    mask_scene = cv2.resize(mask_scene, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Apply erosion
    erosion_kernel = np.ones((7, 7), np.uint8)
    mask_obj = cv2.erode(mask_obj, erosion_kernel, iterations=1)
    mask_hand = cv2.erode(mask_hand, erosion_kernel, iterations=1)

    # mask_scene = cv2.dilate(mask_scene, np.ones((5, 5), np.uint8), iterations=1)

    return mask_obj, mask_hand, mask_scene


def load_object_and_pose(base_dir, seq_name, frame_id):
    # Read annotation file
    anno = read_annotation(base_dir, seq_name, frame_id, data_split)

    # Make the 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(anno['objRot'])[0]
    pose[:3, 3] = anno['objTrans']

    # Load the object model
    # load object model
    obj_mesh = read_obj(os.path.join(YCB_MODELS_DIR, 'models', anno['objName'], 'textured_simple.obj'))

    ## apply current pose to the object model
    #objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

    # From OpenGL coordinates
    from scipy.spatial.transform.rotation import Rotation
    rotx = np.eye(4)
    rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
    # pose = rotx @ pose
    pose = np.matmul(rotx, pose)

    o3d_mesh = o3d.geometry.TriangleMesh()
    if hasattr(obj_mesh, 'r'):
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.copy(obj_mesh.r))
    elif hasattr(obj_mesh, 'v'):
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.copy(obj_mesh.v))
    else:
        raise Exception('Unknown Mesh format')
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.copy(obj_mesh.f))

    return anno['objName'], o3d_mesh, pose, anno['camMat']


def apply_mask(image, mask):
    print('==> Applying mask')
    # Get the indices that are not in the mask
    valid_idx = np.where(mask.flatten() == 0)[0]

    # Set the invalid indices to 0
    masked_image = np.copy(image).flatten()
    for v in valid_idx:
        masked_image[v] = 0  # [0, 0, 0]

    # Mask out values that have too large depth values
    for v in range(len(masked_image)):
        if masked_image[v] > 1.5:
            masked_image[v] = 0

    # Reshape back to input shape
    masked_image = masked_image.reshape(image.shape)

    return masked_image


def get_depth_voxel(voxel, pose, cam_intrinsics, im_size):
    print('==> Getting rendered depth')
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=im_size[1], height=im_size[0], visible=False)
    vis.add_geometry(voxel)
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    c_pose = np.copy(pose)
    param.extrinsic = c_pose
    print(param.intrinsic)
    print(param.intrinsic.intrinsic_matrix)
    param.intrinsic.intrinsic_matrix = cam_intrinsics
    # fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    # param.intrinsic = o3d.PinholeCameraIntrinsic(im_size[1], im_size[0], fx, fy, cx, cy)
    print(param.intrinsic)
    print(param.intrinsic.intrinsic_matrix)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(False)

    return np.array(depth)


def render_depth(obj_id, pose, cam_intrinsics):
    fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    rendering = ren.render_object(obj_id, pose[:3, :3], pose[:3, 3], fx, fy, cx, cy)
    depth = rendering['depth']

    return depth


def add_objects_to_renderer(ren):
    # Get all the .ply files in the model directory
    obj_ids = sorted(os.listdir(os.path.join(YCB_MODELS_DIR, 'models')))
    for obj_id in obj_ids:
        model_path = os.path.join(os.path.join(YCB_MODELS_DIR, 'models'), obj_id, 'mesh.ply')
        print('model_path', model_path)
        ren.add_object(obj_id.replace('.ply', ''), model_path, surf_color=None)


def convert_to_ply():
    # Get the subdirectories
    subdirs = sorted(os.listdir(os.path.join(YCB_MODELS_DIR, 'models')))

    for s in subdirs:
        obj_mesh = read_obj(os.path.join(YCB_MODELS_DIR, 'models', s, 'textured_simple.obj'))
        # From OpenGL coordinates
        from scipy.spatial.transform.rotation import Rotation
        rotx = np.eye(4)
        rotx[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_dcm()
        # pose = rotx @ pose
        pose = np.eye(4)
        pose = np.matmul(rotx, pose)

        o3d_mesh = o3d.geometry.TriangleMesh()
        if hasattr(obj_mesh, 'r'):
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.copy(obj_mesh.r))
        elif hasattr(obj_mesh, 'v'):
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.copy(obj_mesh.v))
        else:
            raise Exception('Unknown Mesh format')
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.copy(obj_mesh.f))

        # Save as .ply
        filename = os.path.join(YCB_MODELS_DIR, 'models', s, 'mesh.ply')
        o3d.io.write_triangle_mesh(filename, o3d_mesh)

        # Export with trimesh
        mesh = trimesh.load(filename)
        mesh.export(filename)


def depth_to_cloud(depth, cam_intrinsics):
    i_fx = 1. / cam_intrinsics[0, 0]
    i_fy = 1. / cam_intrinsics[1, 1]
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]

    pts = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z > 0.01 and z < 1.0:
                x = (u - cx) * z * i_fx
                y = (v - cy) * z * i_fy
                pts.append([x, y, z])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(pts))

    return cloud


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
    parser = argparse.ArgumentParser(description='Test inpainting of depth image')
    # parser.add_argument("depth_image_filename", type=str, help="Sequence of the dataset")
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2'
    args.mask_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2_segmentations_rendered/'
    args.scene = 'ABF10'
    args.model_render = False

    if args.model_render:
        width = 640
        height = 480
        renderer_modalities = []
        renderer_modalities.append('rgb')
        renderer_modalities.append('depth')
        renderer_mode = '+'.join(renderer_modalities)
        ren = renderer.create_renderer(width, height, 'python', mode=renderer_mode, shading='flat')
        add_objects_to_renderer(ren)

    frame_ids = sorted(os.listdir(os.path.join(args.ho3d_path, data_split, args.scene, 'rgb')))
    for i in range(500, len(frame_ids)):
        # Get the id
        frame_id = frame_ids[i].split('.')[0]

        # Load the image
        depth, mask_obj, mask_hand, mask_scene = load_data(args.ho3d_path, args.mask_dir, args.scene, frame_id)

        # Load the object pose
        obj_id, object_mesh, object_pose, camK = load_object_and_pose(args.ho3d_path, args.scene, frame_id)

        if args.model_render:
            rendered_depth = render_depth(obj_id, object_pose, camK)
            # rendered_depth = get_depth_voxel(object_mesh, object_pose, camK, depth.shape)

        # Mask the image
        depth_masked = apply_mask(depth, mask_obj)

        '''
        img_output = np.hstack((depth, depth_masked))
        img_output = np.hstack((img_output, rendered_depth))
        cv2.imshow('Depth image', img_output)
        cv2.waitKey(0)
        sys.exit(0)
        '''

        # Compute the inpaint mask as any pixel in the object mask that does not have a valid value
        # (either missing as NaN or removed because value too large)
        inpaint_mask1 = np.multiply((mask_obj == 255), (depth_masked == 0)).astype(np.uint8)

        # Get inpainted depth image
        depth_inpainted1 = inpaint(depth_masked, mask=inpaint_mask1)

        '''
        # Inpaint the hand pixels
        inpaint_mask2 = np.multiply((mask_hand == 255), (depth_masked == 0)).astype(np.uint8)
        depth_inpainted2 = inpaint2(depth_inpainted1, mask=inpaint_mask2)

        # Fill in pixels from morphological close
        morph_mask = (depth_inpainted2 > 0).astype(np.uint8)
        closed_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        inpaint_mask3 = ((closed_mask - morph_mask) == 1).astype(np.uint8)
        #depth_inpainted3 = inpaint(depth_inpainted2, mask=inpaint_mask3)
        depth_inpainted3 = inpaint2(depth_inpainted2, mask=inpaint_mask3)

        img_output = np.hstack((morph_mask, closed_mask))
        img_output = np.hstack((img_output, inpaint_mask3))
        img_output = np.hstack((img_output, depth_inpainted3))
        cv2.imshow('Morphological closing', img_output)
        cv2.waitKey(0)

        # Get new pixels
        new_pixels1 = np.logical_and((depth_inpainted1 > 0), (depth_masked == 0))
        new_pixels2 = np.logical_and((depth_inpainted2 > 0), (depth_masked == 0))
        new_pixels3 = np.logical_and((depth_inpainted3 > 0), (depth_masked == 0))

        # Output image
        img_output = np.hstack((depth, depth_masked))
        img_output = np.vstack((img_output, np.hstack((inpaint_mask1, depth_inpainted1))))
        img_output = np.vstack((img_output, np.hstack((inpaint_mask2, depth_inpainted2))))
        img_output = np.vstack((img_output, np.hstack((inpaint_mask3, depth_inpainted3))))
        # img_output = np.vstack((img_output, np.hstack((new_pixels1, new_pixels2))))

        # Show the images
        scale_percent = 50
        width = int(img_output.shape[1] * scale_percent / 100)
        height = int(img_output.shape[0] * scale_percent / 100)
        img_output = cv2.resize(img_output, (width, height))
        cv2.imshow('Depth image', img_output)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        '''

        cloud = depth_to_cloud(depth_inpainted1, camK)
        outlier_rm_nb_neighbors = 250  # Higher is more aggressive
        outlier_rm_std_ratio = 0.00001  # Smaller is more aggressive
        raduis_rm_min_nb_points = 250  # Don't change
        raduis_rm_radius_factor = 10  # Don't change
        voxel_size = 0.001
        print('==> Removing outliers')
        print('Input points: {}'.format(np.asarray(cloud.points).shape[0]))
        cloud = remove_outliers(cloud,
                                outlier_rm_nb_neighbors=outlier_rm_nb_neighbors,
                                outlier_rm_std_ratio=outlier_rm_std_ratio,
                                raduis_rm_min_nb_points=0, raduis_rm_radius=0)
                                # raduis_rm_min_nb_points=raduis_rm_min_nb_points,
                                # raduis_rm_radius=voxel_size * raduis_rm_radius_factor)
        print('Output points: {}'.format(np.asarray(cloud.points).shape[0]))
        o3d.io.write_point_cloud('/home/tpatten/Data/cloud.ply', cloud)
        # Ball pivoting
        print('==> Computing normals')
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print('==> Generating mesh')
        radii = np.asarray([0.005, 0.01, 0.02, 0.04])
        mesh_recon = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii))
        o3d.io.write_triangle_mesh('/home/tpatten/Data/mesh.ply', mesh_recon)

        sys.exit(0)
