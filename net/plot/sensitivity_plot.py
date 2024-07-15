from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from net.metrics.utility.best_metrics import best_metrics
from net.utility.msg.msg_plot_complete import msg_plot_complete


def sensitivity_plot(figsize: Tuple[int, int],
                     title: str,
                     experiment_ID: str,
                     ticks: List[int],
                     epochs_ticks: np.ndarray,
                     sensitivity_10_FPS: List[float],
                     sensitivity_max: List[float],
                     sensitivity_path: str):
    """
    Sensitivity plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param sensitivity_10_FPS: sensitivity at 10 FPS
    :param sensitivity_max: sensitivity max
    :param sensitivity_path: sensitivity path
    """

    # best metrics
    max_sensitivity_10_FPS, index_max_sensitivity_10_FPS = best_metrics(metric=sensitivity_10_FPS)

    # Figure: Sensitivity
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, sensitivity_10_FPS, marker=".", color='blue', label='Sensitivity 10 FPS')
    plt.plot(ticks, sensitivity_max, marker=".", color='red', label='Sensitivity Max')
    plt.scatter(x=index_max_sensitivity_10_FPS, y=max_sensitivity_10_FPS, marker="x", color='blue', label='Best Sensitivity 10 FPS')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("True Positive Rate")
    plt.ylim(0.0, 1.0)
    plt.savefig(sensitivity_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='Sensitivity')
