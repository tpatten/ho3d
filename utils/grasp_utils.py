from utils.vis_utils import *
from os.path import exists
import numpy as np

wrist_indices = [0]
#thumb_indices = [1, 6, 7, 8]
#index_indices = [2, 9, 10, 11]
#middle_indices = [3, 12, 13, 14]
#ring_indices = [4, 15, 16, 17]
#pinky_indices = [5, 18, 19, 20]
thumb_indices = [13, 14, 15, 16]
index_indices = [1, 2, 3, 17]
middle_indices = [4, 5, 6, 18]
ring_indices = [10, 11, 12, 19]
pinky_indices = [7, 8, 9, 20]
all_indices = [wrist_indices, thumb_indices, index_indices, middle_indices, ring_indices, pinky_indices]
tip_indices = [4, 8, 12, 16, 20]
finger_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
joint_sizes = [0.008, 0.006, 0.004, 0.004]
finger_weights = [1.0, 1.0, 0.75, 0.5, 0.25]
joint_weights = [0.05, 0.05, 1.0, 1.0]
MIN_TIP_DISTANCE = 0.005
MAX_RADIUS_SEARCH = 0.015
LEFT_TIP_IN_CLOUD = 3221
RIGHT_TIP_IN_CLOUD = 5204


def read_hand_annotations(filename):
    '''
    # Read annotations and images
    args.frame_id = frame_id
    frame_name, annotations = read_anno(args)
    frame_path = join(args.frame_root_path, frame_name)
    assert (exists(frame_path)), "RGB frame doesn't exist at %s" % frame_path

    # Hand joint annotations
    joints3d_anno = annotations[0]
    joints3d_anno = joints3d_anno.reshape(21, 3)

    # estimated joint annotations
    joints_monohand_anno = None
    if hasattr(args, 'joint_monohand_path'):
        joints_monohand_anno = annotations[4]
        joints_monohand_anno = joints_monohand_anno.reshape(21, 3)
        if hasattr(args, 'use_gt_wrist') and args.use_gt_wrist:
            wrist_diff = joints_monohand_anno[0, :] - joints3d_anno[0, :]
            joints_monohand_anno -= wrist_diff
        joints3d_anno = joints_monohand_anno

    # Object pose annotations
    obj_anno = annotations[2]
    obj_rot = obj_anno[:3]
    obj_trans = obj_anno[3:6]
    obj_id = annotations[3]

    return joints3d_anno, obj_anno, obj_rot, obj_trans, obj_id
    '''

    pkl_data = load_pickle_data(filename)
    #return pkl_data['handJoints3D'], pkl_data['objRot'], pkl_data['objTrans'], pkl_data['objName'], pkl_data['camMat']
    return pkl_data


def read_grasp_annotations(filename):
    transforms = []
    scores = []

    # Check that the filename exists
    if not exists(filename):
        return transforms, scores

    with open(filename, 'r') as file:
        # read each line
        for anno in file:
            # Split the line into words
            anno = anno.split(' ')
            if anno[-1] == '\n':
                anno = anno[:-1]
            # Get the 4x4 transformation
            tf = np.asarray(anno[:-1]).reshape((4, 4)).astype(np.float)
            transforms.append(tf)
            scores.append(float(anno[-1]))

    # Return the transforms and scores
    return transforms, scores


def save_grasp_annotations(filename, transforms, scores):
    f = open(filename, "w")
    for i in range(len(transforms)):
        for t in transforms[i].flatten():
            f.write('{} '.format(t))
        f.write('{} '.format(scores[i]))
        f.write('\n')
    f.close()


def save_grasp_annotation_pkl(filename, transform):
    with open(filename, 'wb') as f:
        pickle.dump(transform, f)


def save_grasp_results(filename, grasp_results):
    f = open(filename, "w")
    for g in grasp_results:
        f.write('{} '.format(g))
        f.write('\n')
    f.close()


def quaternion_weighted_average_markley(quats, weights):
    '''
    Averaging Quaternions.
    http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf

    Arguments:
        quats: Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.
    '''
    # Form the symmetric accumulator matrix
    a = np.zeros((4, 4))
    m = quats.shape[0]
    w_sum = 0

    for i in range(m):
        q = quats[i, :]
        w_i = weights[i]
        a += w_i * (np.outer(q, q))  # rank 1 update
        w_sum += w_i
    # scale
    a /= w_sum
    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(a)[1][:, -1]


def vectors_to_rotation_matrix(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinity series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised)
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!)
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements
    R = np.zeros((3, 3))
    R[0, 0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0, 1] = -z*sa + (1.0 - ca)*x*y
    R[0, 2] = y*sa + (1.0 - ca)*x*z
    R[1, 0] = z*sa+(1.0 - ca)*x*y
    R[1, 1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1, 2] = -x*sa+(1.0 - ca)*y*z
    R[2, 0] = -y*sa+(1.0 - ca)*x*z
    R[2, 1] = x*sa+(1.0 - ca)*y*z
    R[2, 2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

    return R


def nearest_point_on_line(line_pt_1, line_pt_2, pt):
    direction = line_pt_2 - line_pt_1
    length = np.linalg.norm(direction)
    direction /= length
    min_dist = 1000
    best_point = line_pt_1
    for i in np.arange(0.0, length, length / 50):
        test_pt = line_pt_1 + i*direction
        d = np.linalg.norm(pt - test_pt)
        if d < min_dist:
            min_dist = d
            best_point = test_pt

    return best_point


def rigid_transform(a, b):
    assert len(a) == len(b)

    num_rows, num_cols = a.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = b.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_a = np.mean(a, axis=1)
    centroid_b = np.mean(b, axis=1)

    # print(centroid_A)

    # subtract mean
    # Am = A - np.tile(centroid_A, (1, num_cols))
    # Bm = B - np.tile(centroid_B, (1, num_cols))
    a_m = a - np.tile(centroid_a, (1, num_cols)).reshape(num_rows, num_cols)
    b_m = b - np.tile(centroid_b, (1, num_cols)).reshape(num_rows, num_cols)

    # dot is matrix multiplication for array
    # H = Am * np.transpose(Bm)
    h = np.matmul(a_m, b_m.T)

    # find rotation
    u, s, v_t = np.linalg.svd(h)
    rot = v_t.T * u.T

    # special reflection case
    if np.linalg.det(rot) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        v_t[2, :] *= -1
        rot = v_t.T * u.T

    tra = -rot * centroid_a + centroid_b

    return rot, tra
