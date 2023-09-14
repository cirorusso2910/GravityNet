import numpy as np
from matplotlib import pyplot as plt

from net.plot.coords.save_coords import save_coords
from net.utility.msg.msg_plot_complete import msg_plot_complete


def FROC_plot(title: str,
              color: str,
              experiment_ID: str,
              FPS: np.ndarray,
              sens: np.ndarray,
              FROC_path: str,
              FROC_coords_path: str):
    """
    FROC plot

    :param title: plot title
    :param color: plot color
    :param experiment_ID: experiment ID
    :param FPS: false positive per scan (FPS)
    :param sens: sensitivity (sens)
    :param FROC_path: FROC path
    :param FROC_coords_path: FROC coords path
    """

    # Figure: FROC
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0, x=0.5)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(FPS, sens, color=color)
    plt.xlabel('Average number of false positives per scan')
    plt.xscale('log')
    plt.xlim(0.000001, 100000)
    plt.xticks([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    plt.ylabel('True Positive Rate')
    plt.ylim(0.0, 1.0)
    plt.savefig(FROC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # Save Coords: FROC (Test)
    save_coords(x=FPS,
                y=sens,
                coords_type='FROC',
                path=FROC_coords_path)

    # plot complete
    msg_plot_complete(plot_type='FROC')
