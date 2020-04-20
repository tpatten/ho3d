import argparse
import sys
from os.path import join, isdir
from os import makedirs
import numpy as np
import random


ALL_CODE = 'ALL'
SUBSET_CODE = 'SUBSET'
LOO_CODE = 'LOO'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Robot grasp train/test splits generator')
    args = parser.parse_args()
    # args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.ho3d_path = '/home/tpatten/Data/Hands/HO3D/'
    args.subject_name = ['ABF', 'BB', 'GPMF', 'GSF', 'MDF', 'ShSu']
    args.save_dir = 'splits_new'
    args.generation_mode = LOO_CODE
    args.balance_sets = True
    args.test_proportion = 0.2
    args.reject_grasp_fails = False
    args.save = True

    random.seed()

    # Read the lines in the train.txt file
    ho3d_train_file = join(args.ho3d_path, 'train.txt')
    f = open(ho3d_train_file, "r")
    file_list = [line[:-1] for line in f]
    f.close()

    # Create the directory to save the data
    if args.save:
        save_dir = join(args.ho3d_path, args.save_dir)
        if isdir(save_dir):
            print('Save directory {} already exists!'.format(save_dir))
            sys.exit(0)
        try:
            makedirs(save_dir)
        except OSError:
            pass

    # Check if grasp is successfully annotated for each file
    if args.reject_grasp_fails:
        grasp_score_filename = join(args.ho3d_path, 'grasp_success.txt')
        f = open(grasp_score_filename, "r")
        grasp_success = [bool(line[:-1]) for line in f]
        f.close()
    else:
        grasp_success = [True for _ in file_list]

    print('Training set contains {} files and {} grasps'.format(len(file_list), len(grasp_success)))
    print('Grasp success is {} from {}'.format(sum(grasp_success), len(grasp_success)))

    # Try to maintain the same split structure
    if args.generation_mode == LOO_CODE:
        subject_indices = {}
        for i in range(len(file_list)):
            if grasp_success[i]:
                subset = file_list[i].split('/')[0]
                # Get the subject from the subject+camera
                counter = 0
                for c in subset:
                    if not c.isalpha():
                        break
                    else:
                        counter += 1
                subject_name = subset[0:counter]
                if subject_name in args.subject_name:
                    # Add to dictionary
                    if subject_indices.get(subject_name) is None:
                        subject_indices[subject_name] = []
                    subject_indices[subject_name].append(i)

        # Get the maximum number of files per subject
        number_files = [len(subject_indices[i]) for i in args.subject_name]
        max_files = min(number_files)
        # Randomly select files
        for i in range(len(args.subject_name)):
            if number_files[i] != max_files:
                keep_indices = np.random.choice(number_files[i], size=(max_files, 1), replace=False).flatten()
                reduced_samples = []
                for j in keep_indices:
                    reduced_samples.append(subject_indices[args.subject_name[i]][j])
                # Set the subject files to the selected ones
                subject_indices[args.subject_name[i]] = reduced_samples
        # Get all indices
        all_indices = []
        for s in args.subject_name:
            all_indices.extend(subject_indices[s])
        # Sort the indices
        all_indices.sort()

        print('Files for train and test {}'.format(len(all_indices)))

        # Create the final list of file and save
        if args.save:
            for subn in args.subject_name:
                train_samples = []
                test_samples = []
                for i in all_indices:
                    subset = file_list[i].split('/')[0]
                    counter = 0
                    for c in subset:
                        if not c.isalpha():
                            break
                        else:
                            counter += 1
                    subject_name = subset[0:counter]
                    if subject_name == subn:
                        test_samples.append(file_list[i])
                    else:
                        train_samples.append(file_list[i])

                num_files_for_selection = float(len(all_indices))
                print('Test set {} ({:.2f}) Train set {} ({:.2f})'.format(
                    len(test_samples), float(len(test_samples)) / num_files_for_selection,
                    len(train_samples), float(len(train_samples)) / num_files_for_selection))

                train_file = join(args.ho3d_path, args.save_dir, 'X' + str(subn) + '_grasp_train.txt')
                f = open(train_file, "w")
                for s in train_samples:
                    f.write("{}\n".format(s))
                f.close()
                test_file = join(args.ho3d_path, args.save_dir, str(subn) + '_grasp_test.txt')
                f = open(test_file, "w")
                for s in test_samples:
                    f.write("{}\n".format(s))
                f.close()

        sys.exit(0)

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
            if file_dict.get(subject_name) is None:
                file_dict[subject_name] = []
            file_dict[subject_name].append((camera_id, frame_id))

    # Extract the subsets
    subject_files = []
    number_files = []
    for s in args.subject_name:
        s_files = []
        for f in file_dict[s]:
            s_files.append(s + str(f[0]) + "/" + str(f[1]))
        subject_files.append(s_files)
        number_files.append(len(s_files))
    num_files_for_selection = sum(number_files)
    if args.balance_sets:
        print('Balancing subsets...')
        # Get the maximum number of files per subject
        max_files = min(number_files)
        # Randomly select files
        for i in range(len(args.subject_name)):
            if number_files[i] != max_files:
                keep_indices = np.random.choice(number_files[i], size=(max_files, 1), replace=False)
                keep_indices = keep_indices.flatten()
                reduced_samples = []
                for j in keep_indices:
                    reduced_samples.append(subject_files[i][j])
                # Set the subject files to the selected ones
                subject_files[i] = reduced_samples
        num_files_for_selection = len(args.subject_name) * max_files

    # Create the splits
    if args.generation_mode == ALL_CODE:
        files_for_selection = [item for sublist in subject_files for item in sublist]

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

        num_files_for_selection = float(num_files_for_selection)
        print('Test set {} ({:.2f}) Train set {} ({:.2f})'.format(len(test_samples),
                                                                  float(len(test_samples)) / num_files_for_selection,
                                                                  len(train_samples),
                                                                  float(len(train_samples)) / num_files_for_selection))

        # Write the splits to file
        if args.save:
            train_file = join(args.ho3d_path, args.save_dir, 'all_grasp_train.txt')
            f = open(train_file, "w")
            for s in train_samples:
                f.write("{}\n".format(s))
            f.close()
            test_file = join(args.ho3d_path, args.save_dir, 'all_grasp_test.txt')
            f = open(test_file, "w")
            for s in test_samples:
                f.write("{}\n".format(s))
            f.close()
    elif args.generation_mode == SUBSET_CODE:
        print('Files for train and test {}'.format(num_files_for_selection))
        num_test_samples = 0
        num_train_samples = 0
        for i in range(len(args.subject_name)):
            test_indices = np.random.random_integers(0, len(subject_files[i]) - 1,
                                                     size=(int(len(subject_files[i]) * args.test_proportion), 1))
            test_indices = test_indices.flatten()
            test_samples = []
            for j in test_indices:
                test_samples.append(subject_files[i][j])
            test_indices = np.flip(np.sort(test_indices))
            for j in test_indices:
                del subject_files[i][j]
            train_samples = subject_files[i]

            # Write the splits to file
            if args.save:
                train_file = join(args.ho3d_path, args.save_dir, str(args.subject_name[i]) + '_grasp_train.txt')
                f = open(train_file, "w")
                for s in train_samples:
                    f.write("{}\n".format(s))
                f.close()
                test_file = join(args.ho3d_path, args.save_dir, str(args.subject_name[i]) + '_grasp_test.txt')
                f = open(test_file, "w")
                for s in test_samples:
                    f.write("{}\n".format(s))
                f.close()

            num_test_samples += len(test_samples)
            num_train_samples += len(train_samples)

            num_files_set = float(len(test_samples) + len(train_samples))
            print('{} \t Test {}  ({:.2f}) \t Train {}  ({:.2f})'.format(
                str(args.subject_name[i]), len(test_samples), float(len(test_samples)) / num_files_set,
                len(train_samples), float(len(train_samples)) / num_files_set))
        print('----------------------------------------------------------')
        num_files_for_selection = float(num_files_for_selection)
        print('Test {}  ({:.2f}) \t Train {}  ({:.2f})'.format(
            num_test_samples, float(num_test_samples) / num_files_for_selection,
            num_train_samples, float(num_train_samples) / num_files_for_selection))
        print('----------------------------------------------------------')
    elif args.generation_mode == LOO_CODE:
        print('Files for train and test {}'.format(num_files_for_selection))
        for i in range(len(args.subject_name)):
            # Training files are all from other subjects (not this one)
            train_samples = []
            for j in range(len(args.subject_name)):
                if not j == i:
                    train_samples.extend(subject_files[j])

            # Write the splits to file
            if args.save:
                random.shuffle(train_samples)
                train_file = join(args.ho3d_path, args.save_dir, 'X' + str(args.subject_name[i]) + '_grasp_train.txt')
                f = open(train_file, "w")
                for s in train_samples:
                    f.write("{}\n".format(s))
                f.close()
                test_file = join(args.ho3d_path, args.save_dir, str(args.subject_name[i]) + '_grasp_test.txt')
                f = open(test_file, "w")
                for s in subject_files[i]:
                    f.write("{}\n".format(s))
                f.close()

            num_files_set = float(len(subject_files[i]) + len(train_samples))
            print('{} \t Test {}  ({:.2f}) \t Train {}  ({:.2f})'.format(
                str(args.subject_name[i]), len(subject_files[i]), float(len(subject_files[i])) / num_files_set,
                len(train_samples), float(len(train_samples)) / num_files_set))
    else:
        print('Unknown generation mode {}'.format(args.generation_mode))
