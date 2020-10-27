import os
import numpy as np
import open3d as o3d
import pickle

wrist_indices = [0]
thumb_indices = [13, 14, 15, 16]
index_indices = [1, 2, 3, 17]
middle_indices = [4, 5, 6, 18]
ring_indices = [10, 11, 12, 19]
pinky_indices = [7, 8, 9, 20]
all_indices = [wrist_indices, thumb_indices, index_indices, middle_indices, ring_indices, pinky_indices]
joint_sizes = [0.008, 0.006, 0.004, 0.004]
finger_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

data_dir = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/'
# mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, 'GPMF10_AnnoPoses_inPaintRend_tsdf_aligned_clean_poisson.ply'))
mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, 'GPMF10_AnnoPoses_tsdf_clean_poisson.ply'))
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
mesh_o = o3d.io.read_triangle_mesh(os.path.join(data_dir, 'GPMF10_AnnoPoses_inPaintRend_tsdf_aligned_clean.ply'))

vertices = np.asarray(mesh.vertices)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
colors = np.zeros(vertices.shape)
pcd.colors = o3d.utility.Vector3dVector(colors)
kd_tree = o3d.geometry.KDTreeFlann(pcd)

vertices_o = np.asarray(mesh_o.vertices)
pcd_o = o3d.geometry.PointCloud()
pcd_o.points = o3d.utility.Vector3dVector(vertices_o)
colors_o = np.zeros(vertices_o.shape)
for i in range(colors_o.shape[0]):
    colors_o[i] = [0, 255, 0]
pcd_o.colors = o3d.utility.Vector3dVector(colors_o)

# o3d.visualization.draw_geometries([pcd, pcd_o])

annotated_frames = ['0000', '0070', '0078', '0088', '0214', '0411', '0477', '0493', '0655', '0663', '0668', '0805',
                    '0823', '0843', '0867']

# Load hand joints
vertices_to_deform = set()
for anno in annotated_frames:
    meta_filename = os.path.join(data_dir, 'GPMF10', 'meta', anno + '.pkl')
    with open(meta_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    joints3d_anno = pickle_data['handJoints3D']
    # Iterate through fingertips and transform to object frame
    # finger_positions = []
    for i in range(len(all_indices)):
        for j in range(len(all_indices[i])):
            trans3d = joints3d_anno[all_indices[i][j]]
            [_, idx, _] = kd_tree.search_knn_vector_3d(trans3d, 1)
            obj_pt = np.asarray(pcd.points)[idx[0], :]
            dd = np.linalg.norm(trans3d - obj_pt)
            # finger_positions.append(trans3d)
            # finger_to_point_distances.append(dd)
            if dd < 0.05:
                # Find all points in radius
                vertices_to_deform.add(idx[0])
                [num_pts, idx, _] = kd_tree.search_radius_vector_3d(obj_pt, 0.005)
                for k in range(num_pts):
                    vertices_to_deform.add(idx[k])

'''
# Plot the finger joints
vis_meshes = [mesh] #[pcd]
for i in range(len(all_indices)):
    for j in range(len(all_indices[i])):
        mm = o3d.geometry.TriangleMesh.create_sphere(radius=joint_sizes[j])
        mm.compute_vertex_normals()
        mm.paint_uniform_color(finger_colors[i])
        trans3d = joints3d_anno[all_indices[i][j]]
        tt = np.eye(4)
        tt[0:3, 3] = trans3d
        mm.transform(tt)
        #mm.scale(1000, center=(0, 0, 0))
        # mm.rotate(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32), center=(0, 0, 0))
        #mm.rotate(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32), center=(0, 0, 0))
        #vis.add_geometry(mm)
        vis_meshes.append(mm)

o3d.visualization.draw_geometries(vis_meshes)
sys.exit(0)
'''

# Statistical outlier removal
outlier_rm_nb_neighbors = 20
outlier_rm_std_ratio = 0.5
_, inliers = pcd.remove_statistical_outlier(nb_neighbors=outlier_rm_nb_neighbors, std_ratio=outlier_rm_std_ratio)
inliers = set(inliers)
all_indices = set([i for i in range(vertices.shape[0])])
outliers = all_indices - inliers

mesh_center = pcd.get_center()
vertices_to_deform = list(vertices_to_deform.union(outliers))
# vertices_to_deform = list(vertices_to_deform)
vertices_to_deform_pos = np.zeros((len(vertices_to_deform), 3))
new_vertices = np.copy(vertices)
for i in range(len(vertices_to_deform)):
    curr_pt = vertices[vertices_to_deform[i]].reshape((3, ))
    # vec_to_centroid = mesh_center - curr_pt
    # vec_to_centroid = vec_to_centroid / np.linalg.norm(vec_to_centroid)
    # new_pos = curr_pt + 0.005 * vec_to_centroid
    # new_vertices[vertices_to_deform[i]] = new_pos.reshape(vertices[i].shape)
    # vertices_to_deform_pos[i, :] = new_pos

    [num_pts, idx, _] = kd_tree.search_radius_vector_3d(curr_pt, 0.01)
    add_new_vert = False
    if num_pts > 1:
        mean_of_pts = np.mean(vertices[idx[1:], :], axis=0)
        if not np.isnan(mean_of_pts[0]):
            new_vertices[vertices_to_deform[i]] = mean_of_pts.reshape(vertices[i].shape)
            vertices_to_deform_pos[i, :] = mean_of_pts
            add_new_vert = True
    if not add_new_vert:
        new_vertices[vertices_to_deform[i]] = curr_pt
        vertices_to_deform_pos[i, :] = curr_pt


pcd.points = o3d.utility.Vector3dVector(new_vertices)
colors = np.zeros(vertices.shape)
for v in vertices_to_deform:
    colors[v] = [255, 0, 0]
for o in outliers:
    colors[o] = [0, 0, 255]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
# sys.exit(0)

#'''
vertices_to_deform = np.array(vertices_to_deform, dtype=np.int32).reshape((len(vertices_to_deform), ))
constraint_ids = o3d.utility.IntVector(vertices_to_deform)
constraint_pos = o3d.utility.Vector3dVector(vertices_to_deform_pos)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)

mesh_prime.compute_vertex_normals()
mesh_prime.translate((0.3, 0.0, 0.0))
mesh_prime.paint_uniform_color([0.2, 0.2, 0.7])
o3d.visualization.draw_geometries([mesh, mesh_prime])
#'''
