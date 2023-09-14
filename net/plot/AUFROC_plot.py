import numpy as np

from typing import Tuple, List
from matplotlib import pyplot as plt

from net.metrics.utility.best_metrics import best_metrics
from net.utility.msg.msg_plot_complete import msg_plot_complete


def AUFROC_plot(figsize: Tuple[int, int],
                title: str,
                experiment_ID: str,
                ticks: List[int],
                epochs_ticks: np.ndarray,
                AUFROC_0_1: List[float],
                AUFROC_0_10: List[float],
                AUFROC_0_50: List[float],
                AUFROC_0_100: List[float],
                AUFROC_val_path: str):
    """
    AUFROC plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param AUFROC_0_1: AUFROC [0, 1]
    :param AUFROC_0_10: AUFROC [0, 10]
    :param AUFROC_0_50: AUFROC [0, 50]
    :param AUFROC_0_100: AUFROC [0, 100]
    :param AUFROC_val_path: AUFROC path
    """

    # best metrics
    max_AUFROC_0_1, index_max_AUFROC_0_1 = best_metrics(metric=AUFROC_0_1)
    max_AUFROC_0_10, index_max_AUFROC_0_10 = best_metrics(metric=AUFROC_0_10)
    max_AUFROC_0_50, index_max_AUFROC_0_50 = best_metrics(metric=AUFROC_0_50)
    max_AUFROC_0_100, index_max_AUFROC_0_100 = best_metrics(metric=AUFROC_0_100)

    # Figure: AUFROC
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    # AUFROC [0, 1]
    plt.plot(ticks, AUFROC_0_1, marker=".", color='red', label='AUFROC [0, 1]')
    plt.scatter(x=index_max_AUFROC_0_1, y=max_AUFROC_0_1, marker="x", color='red', label='Best AUFROC [0, 1]')
    # AUFROC [0, 10]
    plt.plot(ticks, AUFROC_0_10, marker=".", color='blue', label='AUFROC [0, 10]')
    plt.scatter(x=index_max_AUFROC_0_10, y=max_AUFROC_0_10, marker="x", color='blue', label='Best AUFROC [0, 10]')
    # AUFROC [0, 50]
    plt.plot(ticks, AUFROC_0_50, marker=".", color='green', label='AUFROC [0, 50]')
    plt.scatter(x=index_max_AUFROC_0_50, y=max_AUFROC_0_50, marker="x", color='green', label='Best AUFROC [0, 50]')
    # AUFROC [0, 100]
    plt.plot(ticks, AUFROC_0_100, marker=".", color='orange', label='AUFROC [0, 100]')
    plt.scatter(x=index_max_AUFROC_0_100, y=max_AUFROC_0_100, marker="x", color='orange', label='Best AUFROC [0, 100]')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=4)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("AUFROC")
    plt.ylim(0.0, 1.0)
    plt.savefig(AUFROC_val_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='AUFROC')
