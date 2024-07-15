from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from net.metrics.utility.best_metrics import best_metrics
from net.utility.msg.msg_plot_complete import msg_plot_complete


def AUPR_plot(figsize: Tuple[int, int],
              title: str,
              experiment_ID: str,
              ticks: List[int],
              epochs_ticks: np.ndarray,
              AUPR: List[float],
              AUPR_path: str):
    """
    AUPR plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param AUPR: AUPR
    :param AUPR_path: AUPR path
    """

    # best metrics
    max_AUPR, index_max_AUPR = best_metrics(metric=AUPR)

    # Figure: AUPR
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, AUPR, marker=".", color='blue', label='AUPR')
    plt.scatter(x=index_max_AUPR, y=max_AUPR, marker="x", color='blue', label='Best AUPR')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=4)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("AUPR")
    plt.ylim(0.0, 1.0)
    plt.savefig(AUPR_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='AUPR')
