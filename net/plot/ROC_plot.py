import numpy as np

from matplotlib import pyplot as plt

from net.plot.coords.save_coords import save_coords
from net.utility.msg.msg_plot_complete import msg_plot_complete


def ROC_plot(title: str,
             color: str,
             experiment_ID: str,
             FPR: np.ndarray,
             TPR: np.ndarray,
             ROC_path: str,
             ROC_coords_path: str):
    """
    ROC plot

    :param title: plot title
    :param color: plot color
    :param experiment_ID: experiment ID
    :param FPR: False Positive Rate (FPR)
    :param TPR: True Positive Rate (TPR)
    :param ROC_path: ROC path
    :param ROC_coords_path: ROC coords path
    """

    # Figure: ROC
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0, x=0.5)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(FPR, TPR, color=color)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(ROC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # Save Coords: ROC
    save_coords(x=FPR,
                y=TPR,
                coords_type='ROC',
                path=ROC_coords_path)

    # plot complete
    msg_plot_complete(plot_type='ROC')
