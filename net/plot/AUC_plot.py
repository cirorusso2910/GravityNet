import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List

from net.utility.msg.msg_plot_complete import msg_plot_complete


def AUC_plot(figsize: Tuple[int, int],
             title: str,
             experiment_ID: str,
             ticks: List[int],
             epochs_ticks: np.ndarray,
             AUC: List[float],
             AUC_path: str):
    """
    AUC plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param AUC: AUC
    :param AUC_path: AUC path
    """

    # Figure: AUC
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, AUC, marker=".", color='blue')
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.savefig(AUC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='AUC')
