import os
import open3d as o3d
import json
import pickle
import cv2
import numpy as np
import copy

ho3d_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2'
scene_id = 'AP12'
pose_annotation_file = 'pair_pose.json'
ho3d_to_ycb_map_path = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/ho3d_to_ycb.json'
ycb_models_dir = '/home/tpatten/Data/Hands/HO3D_V2/HO3D_v2/'
bop_model_path = '/home/tpatten/Data/bop/ycbv/models_eval'
save_to_file = True
visualize = False


if __name__ == '__main__':
    tfs = {}

    # Load the mesh
    mesh_tsdf = o3d.io.read_triangle_mesh(os.path.join(ho3d_dir, 'reconstructions', scene_id + '_AnnoPoses_tsdf.ply'))

    # Load the annotated frames
    filename = os.path.join(ho3d_dir, 'train', scene_id, pose_annotation_file)
    print('Loading pose file {}'.format(filename))
    if os.path.isfile(filename):
        with open(filename) as json_file:
            relative_poses = json.load(json_file)
        frame_ids = relative_poses.keys()
        frame_ids = sorted([f.zfill(4) for f in frame_ids])
    else:
        raise Exception('{} does not exist'.format(filename))

    # Load the annotation file
    frame_id = frame_ids[0]
    print('Frame id {}'.format(frame_id))
    pickle_filename = os.path.join(ho3d_dir, 'train', scene_id, 'meta', frame_id + '.pkl')
    with open(pickle_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    anno = pickle_data

    # Get the transformation of the first annotated frame to the YCB object frame
    pose_annotation_global_rot = cv2.Rodrigues(anno['objRot'])[0].T
    pose_annotation_global_tra = anno['objTrans']

    # Transform the mesh to the YCB object
    source = o3d.geometry.PointCloud()
    points = copy.deepcopy(np.asarray(mesh_tsdf.vertices))
    #points -= pose_annotation_global_tra
    #points = np.matmul(points, np.linalg.inv(pose_annotation_global_rot))
    #source.points = o3d.utility.Vector3dVector(points)

    #'''
    source.points = o3d.utility.Vector3dVector(points)
    rot_mat = np.eye(4)
    rot_mat[:3, 3] = -pose_annotation_global_tra
    source.transform(rot_mat)
    print('TF 0:')
    print(rot_mat)
    tfs['0'] = list(rot_mat.flatten())

    rot_mat = np.eye(4)
    rot_mat[:3, :3] = pose_annotation_global_rot
    source.transform(rot_mat)
    print('TF 1:')
    print(rot_mat)
    tfs['1'] = list(rot_mat.flatten())
    #'''

    # Load the CAD model to align the mesh to
    with open(os.path.join(ho3d_to_ycb_map_path)) as f:
        model_name_data = json.load(f)
    scene_key = ''.join([i for i in scene_id if not i.isdigit()])
    ycb_model_filename = os.path.join(ycb_models_dir, 'models', model_name_data[scene_key]['ycbv'], 'mesh.ply')
    target = o3d.io.read_point_cloud(ycb_model_filename)

    if visualize:
        print('=================== Visualizing source and YCB model')
        target.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([source, target])

    # Run ICP to align the mesh to the model
    threshold = 0.02
    print(o3d.__version__)
    if o3d.__version__ == '0.7.0.0':
        o3d.geometry.estimate_normals(source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.geometry.estimate_normals(target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2l = o3d.registration.registration_icp(
            source, target, threshold, np.eye(4),
            o3d.registration.TransformationEstimationPointToPlane())
    else:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        if o3d.__version__ == '0.10.0.0':
            reg_p2l = o3d.registration.registration_icp(
                source, target, threshold, np.eye(4),
                o3d.registration.TransformationEstimationPointToPlane())
        else:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source, target, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())

    source.transform(reg_p2l.transformation)
    print('TF 2:')
    print(reg_p2l.transformation)
    tfs['2'] = list(reg_p2l.transformation.flatten())

    if visualize:
        print('=================== Visualizing source and YCB model after ICP')
        target.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([source, target])

    # Transform to BOP
    t_mat = np.eye(4)
    t_mat[:3, 3] = -np.asarray(model_name_data[scene_key]['offset_bop'])
    source.transform(t_mat)
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) * 1000)
    print('TF 3:')
    print(t_mat)
    tfs['3'] = list(t_mat.flatten())

    ycb_model_filename = os.path.join(bop_model_path, 'obj_' + model_name_data[scene_key]['bop'].zfill(6) + '.ply')
    mesh_ycb = o3d.io.read_triangle_mesh(ycb_model_filename)

    if visualize:
        print('=================== Visualizing source and BOP model')
        o3d.visualization.draw_geometries([mesh_ycb, source])

    new_source = o3d.geometry.PointCloud()
    new_source.points = o3d.utility.Vector3dVector(np.asarray(mesh_tsdf.vertices))
    tf_mat = np.matmul(np.asarray(tfs['1']).reshape((4, 4)), np.asarray(tfs['0']).reshape((4, 4)))
    tf_mat = np.matmul(np.asarray(tfs['2']).reshape((4, 4)), tf_mat)
    tf_mat = np.matmul(np.asarray(tfs['3']).reshape((4, 4)), tf_mat)
    new_source.transform(tf_mat)
    print('TF Final:')
    print(tf_mat)
    tfs['tf'] = list(tf_mat.flatten())
    new_source.points = o3d.utility.Vector3dVector(np.asarray(new_source.points) * 1000)

    if visualize:
        print('=================== Visualizing source and BOP model after combined transformation')
        o3d.visualization.draw_geometries([mesh_ycb, new_source])

    if save_to_file:
        tf_file = os.path.join(ho3d_dir, 'reconstructions', scene_id + '_tf.json')
        with open(tf_file, 'w') as file:
            json.dump(tfs, file)

    # Convert final model to the original
    if o3d.__version__ != '0.7.0.0':
        mesh_final = o3d.io.read_triangle_mesh(os.path.join(ho3d_dir, 'reconstructions', scene_id + '_merged.ply'))
        #o3d.visualization.draw_geometries([mesh_tsdf])
        mesh_final.scale(0.001, center=(0, 0, 0))
        mesh_final.transform(np.linalg.inv(np.asarray(tfs['tf']).reshape(4, 4)))

        '''
        init_tf = np.asarray(tfs['tf']).reshape(4, 4)
        axis_conv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        tf_mat = np.matmul(axis_conv, np.linalg.inv(init_tf))
        mesh_final.transform(tf_mat)
        '''

        print('=================== Visualizing final model transformed back to original TSDF')
        o3d.visualization.draw_geometries([mesh_tsdf, mesh_final])
        if save_to_file:
            o3d.io.write_triangle_mesh(os.path.join(ho3d_dir, 'reconstructions', scene_id + '_mesh_for_rendering.ply'),
                                       mesh_final)
    else:
        print('Transforming model to original frame unavailable with Open3D 0.7.0.0.')
        print('Run with Open3D >= 0.9.0.0')

