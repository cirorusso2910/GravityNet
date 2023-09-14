import os

from matplotlib import pyplot as plt
from natsort import natsort
from pandas import read_csv

from net.initialization.header.coords import coords_header
from net.plot.coords.save_coords import save_coords
from net.utility.msg.msg_plot_complete import msg_plot_complete


def resume_FROC_plot(experiment_ID: str,
                     coords_resume_path: str,
                     coords_path: str,
                     plot_resume_path: str,
                     plot_path: str):
    """
    Resume FROC validation plot

    :param experiment_ID: experiment ID
    :param coords_resume_path: coords path from resume experiment
    :param coords_path: coords path from experiment
    :param plot_resume_path: plot path from resume experiment
    :param plot_path: plot path from experiment
    """

    coords_list_file = sorted(os.listdir(coords_resume_path))
    coords_list_file = natsort.natsorted(coords_list_file)

    plot_list_file = sorted(os.listdir(plot_resume_path))
    plot_list_file = natsort.natsorted(plot_list_file)

    tot_file = len(coords_list_file)

    for i in range(tot_file):
        prefix_coords = coords_list_file[i].split('|')[0]
        prefix_plot = plot_list_file[i].split('|')[0]
        ep = prefix_plot.split('=')[1]

        old_coords_path = os.path.join(coords_resume_path, coords_list_file[i])
        new_coords_filename = prefix_coords + "|" + experiment_ID + ".csv"
        new_coords_path = os.path.join(coords_path, new_coords_filename)

        new_plot_filename = prefix_plot + "|" + experiment_ID + ".png"
        new_plot_path = os.path.join(plot_path, new_plot_filename)

        # read coords
        header = coords_header(coords_type='FROC')
        coords = read_csv(filepath_or_buffer=old_coords_path, usecols=header, float_precision='round_trip')
        FPS = coords[header[0]].tolist()
        sens = coords[header[1]].tolist()

        # Figure: FROC
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle("FROC (VALIDATION) | EPOCH={}".format(ep), fontweight="bold", fontsize=18, y=1.0, x=0.5)
        plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
        plt.grid()
        plt.plot(FPS, sens, color='green')
        plt.xlabel('Average number of false positives per scan')
        plt.xscale('log')
        plt.xlim(0.000001, 100000)
        plt.xticks([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
        plt.ylabel('True Positive Rate')
        plt.ylim(0.0, 1.0)
        plt.savefig(new_plot_path, bbox_inches='tight')
        plt.clf()  # clear figure
        plt.close(fig)

        # Save Coords: FROC (Test)
        save_coords(x=FPS,
                    y=sens,
                    coords_type='FROC',
                    path=new_coords_path)

    msg_plot_complete(plot_type='resume FROC')


def resume_ROC_plot(experiment_ID: str,
                    coords_resume_path: str,
                    coords_path: str,
                    plot_resume_path: str,
                    plot_path: str):
    """
    Resume ROC validation plot

    :param experiment_ID: experiment ID
    :param coords_resume_path: coords path from resume experiment
    :param coords_path: coords path from experiment
    :param plot_resume_path: plot path from resume experiment
    :param plot_path: plot path from experiment
    """

    coords_list_file = sorted(os.listdir(coords_resume_path))
    coords_list_file = natsort.natsorted(coords_list_file)

    plot_list_file = sorted(os.listdir(plot_resume_path))
    plot_list_file = natsort.natsorted(plot_list_file)

    tot_file = len(coords_list_file)

    for i in range(tot_file):
        prefix_coords = coords_list_file[i].split('|')[0]
        prefix_plot = plot_list_file[i].split('|')[0]
        ep = prefix_plot.split('=')[1]

        old_coords_path = os.path.join(coords_resume_path, coords_list_file[i])
        new_coords_filename = prefix_coords + "|" + experiment_ID + ".csv"
        new_coords_path = os.path.join(coords_path, new_coords_filename)

        new_plot_filename = prefix_plot + "|" + experiment_ID + ".png"
        new_plot_path = os.path.join(plot_path, new_plot_filename)

        # read coords
        header = coords_header(coords_type='ROC')
        coords = read_csv(filepath_or_buffer=old_coords_path, usecols=header, float_precision='round_trip')
        FPR = coords[header[0]].tolist()
        TPR = coords[header[1]].tolist()

        # Figure: ROC (Validation)
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle("ROC (VALIDATION) | EPOCH={}".format(ep), fontweight="bold", fontsize=18, y=1.0, x=0.5)
        plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
        plt.grid()
        plt.plot(FPR, TPR, color='green')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(new_plot_path, bbox_inches='tight')
        plt.clf()  # clear figure
        plt.close(fig)

        # Save Coords: ROC (Validation)
        save_coords(x=FPR,
                    y=TPR,
                    coords_type='ROC',
                    path=new_coords_path)

    msg_plot_complete(plot_type='resume ROC')
