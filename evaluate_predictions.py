import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt


blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[91m' + x + '\033[0m'
TARGETS = ['ABF', 'BB', 'GPMF', 'GSF', 'MDF', 'ShSu']
CAMERAS = ['10', '11', '12', '13', '14']
# TARGETS = ['ABF', 'BB']
# CAMERAS = ['10', '11']


def load_ground_truth(target_dir, file):
    anno_file_name = os.path.join(target_dir, 'meta', file)
    anno_file_name = anno_file_name.replace(".png", ".pkl")
    with open(anno_file_name, 'rb') as f:
        try:
            anno = pickle.load(f, encoding='latin1')
        except:
            anno = pickle.load(f)
    return anno['handJoints3D']


def load_prediction(target_dir, file):
    anno_file_name = os.path.join(target_dir, 'hand_tracker', file)
    anno_file_name = anno_file_name.replace(".png", ".pkl")
    if not os.path.exists(anno_file_name):
        return None

    with open(anno_file_name, 'rb') as f:
        try:
            anno = pickle.load(f, encoding='latin1')
        except:
            anno = pickle.load(f)
    return anno['handJoints3D']


def load_data(target_dir, file):
    return load_ground_truth(target_dir, file), load_prediction(target_dir, file)


def load_processed(filename):
    with open(filename, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except:
            data = pickle.load(f)
    return data


def compute_mean_joint_position_error(gt, pr):
    return np.mean(np.sqrt(np.sum(np.square(gt - pr), axis=1)))


def process(ho3d_dir, target_id, camera_id, verbose=False):
    # The target directory to load data from
    target_dir = os.path.join(ho3d_dir, 'train', target_id + camera_id)

    # Check that hand_tracker is available
    if not os.path.exists(os.path.join(target_dir, 'hand_tracker')):
        print('No hand tracking directory available')
        return np.zeros((0, 0))

    errs = []
    count = 100
    for file in sorted(os.listdir(os.path.join(target_dir, 'rgb'))):
        if verbose:
            print('--- Inspecting file {}'.format(file))
        # Load the joint values for the hand ground truth and estimated
        hand_gt, hand_pr = load_data(target_dir, file)
        if hand_pr is None:
            if verbose:
                print('%s' % (red('No estimated hand')))
            errs.append(np.nan)
        else:
            # Compute the error
            joint_err = compute_mean_joint_position_error(hand_gt, hand_pr) * 100
            if verbose:
                print('{:.2f}'.format(joint_err))
            errs.append(joint_err)

        #count -= 1
        #if count == 0:
        #    return np.array(errs)

    return np.array(errs)


def fit_function(x, A, beta, B, mu, sigma):
    return A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2))


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-((x - mean) / standard_deviation) ** 2)


def visualize_errors(errors):
    # Sum results for each target and camera
    target_results = {}
    for t in TARGETS:
        target_results[t] = []
    camera_results = {}
    for c in CAMERAS:
        camera_results[c] = []

    # Append all results for each target and camera
    for e in errors:
        if e[2].shape[0] > 0:
            target_results[e[0]].extend(e[2])
            camera_results[e[1]].extend(e[2])

    # Plot histograms for each target
    print('\n----- Each target -----')
    fig1 = plt.figure(figsize=(1, 6))
    for i in range(len(TARGETS)):
        print('%s' % (blue(TARGETS[i])))
        t_errs = np.array(target_results[TARGETS[i]]).flatten()

        if np.all(np.isnan(t_errs)):
            print('All NAN')
            continue

        ax = fig1.add_subplot(1, 6, i + 1)
        # `density=False` would make counts
        ax.hist(t_errs[np.logical_not(np.isnan(t_errs))], density=False, bins=50)
        ax.set_title(TARGETS[i])
        ax.set_xlabel('Error (cm)')
        ax.set_ylabel('Counts')

        print('Min:  {:.4f}'.format(np.nanmin(t_errs)))
        print('Max:  {:.4f}'.format(np.nanmax(t_errs)))
        print('Mean: {:.4f}'.format(np.nanmean(t_errs)))
        print('Var:  {:.4f}'.format(np.nanstd(t_errs)))
        print('Val:  {:.2f}'.format(float(np.count_nonzero(~np.isnan(t_errs))) / float(t_errs.shape[0])))

    # Plot each target for each camera
    print('\n----- Each camera -----')
    fig2 = plt.figure(figsize=(1, 5))
    for i in range(len(CAMERAS)):
        print('%s' % (blue(CAMERAS[i])))
        c_errs = np.array(camera_results[CAMERAS[i]]).flatten()

        if np.all(np.isnan(c_errs)):
            print('All NAN')
            continue

        ax = fig2.add_subplot(1, 5, i + 1)
        ax.hist(c_errs[np.logical_not(np.isnan(c_errs))], density=True, bins=50)
        ax.set_title(CAMERAS[i])
        ax.set_xlabel('Error (cm)')
        ax.set_ylabel('Counts')

        print('Min:  {:.4f}'.format(np.nanmin(c_errs)))
        print('Max:  {:.4f}'.format(np.nanmax(c_errs)))
        print('Mean: {:.4f}'.format(np.nanmean(c_errs)))
        print('Var:  {:.4f}'.format(np.nanstd(c_errs)))
        print('Val:  {:.2f}'.format(float(np.count_nonzero(~np.isnan(c_errs))) / float(c_errs.shape[0])))

    plt.show()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Evaluate the hand joint predictions')
    args = parser.parse_args()
    args.ho3d_path = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/'
    args.verbose = False
    args.visualize = True

    # Save filename
    save_filename = os.path.join(args.ho3d_path, 'results/hand_tracker_errors.pkl')
    already_processed = os.path.exists(save_filename)
    if already_processed:
        all_results = load_processed(save_filename)
    else:
        # Accumulate results
        all_results = []
        for targ in TARGETS:
            for cam in CAMERAS:
                print('%s' % (blue(targ + ' ' + cam)))
                errors = process(args.ho3d_path, targ, cam, verbose=args.verbose)
                if errors.shape[0] > 0 and not np.all(np.isnan(errors)):
                    # Print statistics
                    print('Min:  {:.4f}'.format(np.nanmin(errors)))
                    print('Max:  {:.4f}'.format(np.nanmax(errors)))
                    print('Mean: {:.4f}'.format(np.nanmean(errors)))
                    print('Var:  {:.4f}'.format(np.nanstd(errors)))
                    print('Val:  {:.2f}'.format(float(np.count_nonzero(~np.isnan(errors))) / float(errors.shape[0])))

                all_results.append((targ, cam, errors))

    print('\n----- Summary ----- \n')
    for r in all_results:
        print(r[0], r[1], r[2].shape[0])

    # Visualize the errors
    if args.visualize:
        visualize_errors(all_results)

    # Save to pickle file
    if not already_processed:
        with open(save_filename, 'wb') as f:
            pickle.dump(all_results, f)
