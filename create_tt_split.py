import argparse
from os.path import join
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Robot grasp extraction')
    # parser.add_argument('--object-model', type=str, default='003_cracker_box', required=False,
    #                     help='Name of the object model')
    # parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.subject_name = 'ABF'
    #args.subject_name = ''
    args.test_proportion = 0.2

    # Read the lines in the train.txt file
    ho3d_train_file = join(args.ho3d_path, 'train.txt')
    f = open(ho3d_train_file, "r")
    file_list = [line[:-1] for line in f]
    f.close()

    # Check if grasp is successfully annotated for each file
    grasp_score_filename = join(args.ho3d_path, 'grasp_success.txt')
    f = open(grasp_score_filename, "r")
    grasp_success = [bool(line[:-1]) for line in f]
    f.close()

    print('Training set contains {} files and {} grasps'.format(len(file_list), len(grasp_success)))
    print('Grasp success is {} from {}'.format(sum(grasp_success), len(grasp_success)))

    # Create a dictionary for the training files
    file_dict = {}
    for i in range(len(file_list)):
        if grasp_success[i]:
            # Split the subject name and the frame id
            str_split = file_list[i].split('/')
            subset = str_split[0]
            frame_id = str_split[1]
            # Get the subject from the subject+camera
            counter = 0
            for c in subset:
                if not c.isalpha():
                    break
                else:
                    counter += 1
            subject_name = subset[0:counter]
            camera_id = subset[counter:]
            # Add to dictionary
            if not file_dict.has_key(subject_name):
                file_dict[subject_name] = []
            file_dict[subject_name].append((camera_id, frame_id))

    # Create the split
    files_for_selection = []
    if not args.subject_name:
        for i in range(len(file_list)):
            if grasp_success[i]:
                files_for_selection.append(file_list[i])
    else:
        for f in file_dict[args.subject_name]:
            files_for_selection.append(args.subject_name + str(f[0]) + "/" + str(f[1]))
    num_files_for_selection = len(files_for_selection)
    print('Files for train and test {}'.format(num_files_for_selection))

    test_indices = np.random.random_integers(0, len(files_for_selection) - 1,
                                             size=(int(len(files_for_selection) * args.test_proportion), 1))
    test_indices = test_indices.flatten()
    test_samples = []
    for i in test_indices:
        test_samples.append(files_for_selection[i])
    test_indices = np.flip(np.sort(test_indices))
    for i in test_indices:
        del files_for_selection[i]
    train_samples = files_for_selection

    print('Test set {} ({}) Train set {} ({})'.format(len(test_samples),
                                                      float(len(test_samples)) / float(num_files_for_selection),
                                                      len(train_samples),
                                                      float(len(train_samples)) / float(num_files_for_selection)))

    # Write the splits to file
    train_file = join(args.ho3d_path, 'grasp_train.txt')
    f = open(train_file, "w")
    for s in train_samples:
        f.write("{}\n".format(s))
    f.close()
    test_file = join(args.ho3d_path, 'grasp_test.txt')
    f = open(test_file, "w")
    for s in test_samples:
        f.write("{}\n".format(s))
    f.close()
