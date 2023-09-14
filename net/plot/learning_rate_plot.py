import numpy as np

from matplotlib import pyplot as plt
from typing import Tuple, List

from net.utility.msg.msg_plot_complete import msg_plot_complete


def learning_rate_plot(figsize: Tuple[int, int],
                       title: str,
                       experiment_ID: str,
                       ticks: List[int],
                       epochs_ticks: np.ndarray,
                       learning_rate: List[float],
                       learning_rate_path: str):
    """
    Learning rate plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param learning_rate: learning rate
    :param learning_rate_path: learning rate path
    """

    # Figure: Learning rate
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, learning_rate, marker=".", color='blue', label='Learning Rate')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=3)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.savefig(learning_rate_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='Learning Rate')
