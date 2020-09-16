# author: Tim Patten
# contact: patten@acin.tuwien.ac.at

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import json


VALID_VAL = 255
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

bop2ho3d = {}
bop2ho3d['000001'] = 'AP10'
bop2ho3d['000002'] = 'AP11'
bop2ho3d['000003'] = 'AP12'
bop2ho3d['000004'] = 'AP13'
bop2ho3d['000005'] = 'AP14'
bop2ho3d['000006'] = 'MPM10'
bop2ho3d['000007'] = 'MPM11'
bop2ho3d['000008'] = 'MPM12'
bop2ho3d['000009'] = 'MPM13'
bop2ho3d['000010'] = 'MPM14'
bop2ho3d['000011'] = 'MPM10'
bop2ho3d['000012'] = 'SB11'
bop2ho3d['000013'] = 'SB13'
bop2ho3d['000014'] = 'SM1'

#error_metrics = ['error=mspd_ntop=-1',
#                 'error=mssd_ntop=-1',
#                 'error=vsd_ntop=-1_delta=15.000_tau=0.050',
#                 'error=vsd_ntop=-1_delta=15.000_tau=0.150',
#                 'error=vsd_ntop=-1_delta=15.000_tau=0.250',
#                 'error=vsd_ntop=-1_delta=15.000_tau=0.350',
#                 'error=vsd_ntop=-1_delta=15.000_tau=0.450']

error_metrics = ['error=mspd_ntop=-1',
                 'error=mssd_ntop=-1',
                 'error=vsd_ntop=-1_delta=15.000_tau=0.250']


class PE_VisRatio_Plotter:
    def __init__(self, args):
        # Get the ho3d id
        ho3d_seq = bop2ho3d["{}".format(str(args.seq_id).zfill(6))]

        # Load the json file of the visibility ratios
        im_ids, vis_ratios = self.load_visibility_ratios(args.vis_ratio_file, ho3d_seq)

        # Get the pose estimates for each metric
        pose_estimates = self.load_pose_estimation(args.pose_estimation_dir, args.seq_id)

        # Arrange the pose estimates for each frame
        corr_coeffs = []
        for e in error_metrics:
            pe_error = pose_estimates[e]
            pose_estimates_all = []
            for ii in im_ids:
                if ii not in pe_error.keys():
                    #pose_estimates_all.append(-0.5)
                    pose_estimates_all.append(0.0)
                else:
                    pose_estimates_all.append(pe_error[ii])
            pose_estimates[e] = pose_estimates_all
            # Compute the correlation
            vis_ratios_normalized = np.asarray(vis_ratios)
            min_vr = np.min(vis_ratios_normalized)
            max_vr = np.max(vis_ratios_normalized)
            vis_ratios_normalized = (vis_ratios_normalized - min_vr) / (max_vr - min_vr)
            r = np.corrcoef(np.asarray(pose_estimates_all), vis_ratios_normalized)
            print(e)
            print('R = {}'.format(r[0, 1]))
            corr_coeffs.append(r[0, 1])

        # Plot the visibility ratios
        figure_title = args.pose_estimation_dir[args.pose_estimation_dir.rfind("/") + 1:]
        self.plot_data(im_ids, vis_ratios_normalized, pose_estimates, corr_coeffs, figure_title)

    @staticmethod
    def load_visibility_ratios(filename, seq_id):
        with open(filename) as json_file:
            # Load the json file
            data = json.load(json_file)

            # Convert to image ids and visibility ratios for the sequence
            vis_ratios = data[seq_id]
            im_ids, vis_ratios = map(list, zip(*vis_ratios))
            im_ids = list(map(int, im_ids))

            # Return
            return im_ids, vis_ratios

    @staticmethod
    def load_pose_estimation(pe_dir, seq_id):
        # Store all the pose errors
        pose_errors = {}

        # For each directory
        for d in error_metrics:
            # Get the name of this directory
            if not os.path.isdir(os.path.join(pe_dir, d)):
                raise Exception('No directory {}'.format(d))

            # Create the name of the json file to load
            filename = os.path.join(pe_dir, d, 'errors_' + str(seq_id).zfill(6) + '.json')
            # Load the data
            with open(filename) as json_file:
                data = json.load(json_file)
            # Get the maximum error value
            max_error = 0.
            for e in data:
                err = e['errors']['0'][0]
                if not math.isinf(err) and err > max_error:
                    max_error = err
            if max_error == 0.:
                max_error = 1.
            # Create the map
            pe_errors = {}
            for e in data:
                val = 1.0 - (e['errors']['0'][0] / max_error)
                if math.isnan(val) or math.isinf(val):
                    pe_errors[int(e['im_id'])] = 0.0
                else:
                    pe_errors[int(e['im_id'])] = val
            # Append to the overall result
            pose_errors[d] = pe_errors

        # Return the results
        return pose_errors

    @staticmethod
    def plot_data(im_ids, vis_ratios, pose_estimates, corr_coeffs, figure_title=None):
        # Create window
        fig = plt.figure(figsize=(2, 5))
        if figure_title is not None:
            fig.suptitle(figure_title, fontsize=16)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())

        # Plot for each error metric
        num_error_metrics = len(error_metrics)
        for i in range(num_error_metrics):
            ax = fig.add_subplot(num_error_metrics, 1, i + 1)
            ax.plot(im_ids, vis_ratios, 'k--')
            ax.plot(im_ids, pose_estimates[error_metrics[i]], 'r')
            ax.set_title(error_metrics[i])
            ax.text(0, 0, 'R = ' + str(round(corr_coeffs[i], 2)), fontsize=12,
                    bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 1})

        # Plot for each error metric
        fig = plt.figure(figsize=(2, 5))
        if figure_title is not None:
            fig.suptitle(figure_title, fontsize=16)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())
        for i in range(num_error_metrics):
            ax = fig.add_subplot(num_error_metrics, 1, i + 1)
            ax.scatter(vis_ratios, pose_estimates[error_metrics[i]])
            ax.set_title(error_metrics[i])
            ax.text(0, 0, 'R = ' + str(round(corr_coeffs[i], 2)), fontsize=12,
                    bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 1})

        plt.show()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="HO3D object visibility ratio extractor")
    args = parser.parse_args()

    args.vis_ratio_file = '/home/tpatten/Data/Hands/HO3D_V2/visibility.json'
    args.pose_estimation_dir = '/home/tpatten/Data/testr/pix2pose-iccv19_ho3d-test'
    args.seq_id = 5

    p = PE_VisRatio_Plotter(args)
